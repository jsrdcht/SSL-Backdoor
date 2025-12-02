import datetime
import logging
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.cluster import KMeans
import time
import pickle
from torch.utils.data import Dataset, DataLoader, TensorDataset


from ssl_backdoor.defenses.ssl_cleanse.methods import neg_cosine_similarity
from ssl_backdoor.utils.utils import set_seed


# 创建数据集类，直接在内部使用而不是通过adapter
class DatasetCluster(Dataset):
    """聚类数据集，用于触发器优化，随机选择目标表示"""
    def __init__(self, rep_target, x_other_sample):
        self.rep_target = rep_target
        self.x_other_sample = x_other_sample
        self.rep_target_indices = torch.randint(0, rep_target.shape[0], (x_other_sample.shape[0],))

    def __getitem__(self, idx):
        image = self.x_other_sample[idx]
        rep_target = self.rep_target[self.rep_target_indices[idx]]
        return image, rep_target

    def __len__(self):
        return self.x_other_sample.shape[0]


class DatasetClusterFix(Dataset):
    """聚类数据集，用于触发器优化，使用固定目标中心"""
    def __init__(self, rep_target_center, x_other_sample):
        self.rep_target_center = rep_target_center
        self.x_other_sample = x_other_sample

    def __getitem__(self, idx):
        image = self.x_other_sample[idx]
        return image, self.rep_target_center

    def __len__(self):
        return self.x_other_sample.shape[0]


class DatasetEval(Dataset):
    """评估数据集，用于KNN评估"""
    def __init__(self, x, sample_size):
        indices = torch.randperm(x.shape[0])[:sample_size]
        self.x_sample = x[indices]
        self.dummy_labels = torch.zeros(sample_size)

    def __getitem__(self, idx):
        return self.x_sample[idx], self.dummy_labels[idx]

    def __len__(self):
        return self.x_sample.shape[0]


def get_data(device, model, dataloader, width):
    """获取模型输出的嵌入表示和原始数据"""
    rep_all = []
    x_all = []
    y_all = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            rep = model(inputs)  # 直接使用模型进行前向传递
            rep_all.append(rep)
            x_all.append(inputs.to('cpu'))
            y_all.append(labels.to('cpu'))
    
    rep = torch.cat(rep_all, dim=0)
    x = torch.cat(x_all, dim=0)
    y_true = torch.cat(y_all, dim=0)
    
    return rep, x, y_true

def draw(images, mean, std, mask_tanh, delta_tanh):
    """在图像上应用触发器"""
    mask = mask_tanh.expand_as(images)
    delta = delta_tanh.expand_as(images)
    
    # 恢复原始图像
    unnorm_images = images.clone()
    for i in range(3):
        unnorm_images[:, i, :, :] = unnorm_images[:, i, :, :] * std[i] + mean[i]
    
    # 应用触发器
    X_poisoned = unnorm_images + mask * delta
    
    # 重新归一化
    poisoned_images = X_poisoned.clone()
    for i in range(3):
        poisoned_images[:, i, :, :] = (poisoned_images[:, i, :, :] - mean[i]) / std[i]
    
    return poisoned_images

def eval_knn(device, model, dataloader, rep_knn, y_knn, target_label):
    """评估KNN分类的攻击成功率"""
    if model.training:
        model.eval()
    
    y_knn = y_knn.to(device)
    rep_knn = rep_knn.to(device)
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            rep = model(inputs)  # 直接使用模型进行前向传递
            
            # 计算嵌入表示之间的距离
            dist = torch.cdist(rep, rep_knn)
            
            # 找到最近的索引
            _, idx = torch.topk(dist, k=1, dim=1, largest=False)
            pred = torch.gather(y_knn, dim=0, index=idx.squeeze(1))
            
            # 计算攻击成功率
            total += inputs.size(0)
            correct += (pred == target_label).sum().item()
    
    return correct / total

def outlier(reg_best_list):
    """计算异常值统计（与原始 SSL-Cleanse 实现保持一致）"""
    # 与原仓库中的实现对齐：
    # consistency_constant = 1.4826
    # median = torch.median(l1_norm_list)
    # mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median)) / 0.6745
    # min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad
    consistency_constant = 1.4826
    median = torch.median(reg_best_list)
    mad = consistency_constant * torch.median(torch.abs(reg_best_list - median)) / 0.6745
    min_mad = torch.abs(torch.min(reg_best_list) - median) / mad
    return median, mad, min_mad

def run_ssl_cleanse(args, suspicious_model, suspicious_dataset, clean_test_dataset=None, poisoned_test_dataset=None):
    """
    运行SSL-Cleanse后门检测算法
    
    参数:
        args: 配置参数
        suspicious_model: 可疑的自监督模型 (可能被后门攻击)
        suspicious_dataset: 用于清洗的数据集 (可能包含毒样本)
        clean_test_dataset: 干净的测试数据集 (可选，用于评估)
        poisoned_test_dataset: 有毒的测试数据集 (可选，用于评估)
    
    返回:
        result_dict: 包含检测结果的字典
    """
    start_time = time.time()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s => %(message)s')
    now = datetime.datetime.now()
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    log_filename = os.path.join(args.output_dir, "logs", now.strftime('%Y-%m-%d %H-%M-%S') + '.log')
    file_handler = logging.FileHandler(filename=log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s => %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    logging.info(f'参数: dataset={args.dataset_name}, num_clusters={args.num_clusters}, ratio={args.ratio}, '
                 f'attack_succ_threshold={args.attack_succ_threshold}, target_center={args.target_center}, '
                 f'weights_path={args.weights_path}, knn_center={args.knn_center}')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    suspicious_model.to(device).eval()
    
    for param in suspicious_model.parameters():
        param.requires_grad = False

    if device == "cuda":
        cudnn.benchmark = True
    
    # 数据集信息
    if args.dataset_name == "imagenet100":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        width = 224
    elif args.dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.201]
        width = 32
    else:
        # 默认值
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        width = 32
        logging.warning(f"未知数据集 {args.dataset_name}，使用默认归一化参数")

    # 创建初始数据加载器
    if args.ratio < 1.0:
        # 随机采样数据集
        total_size = len(suspicious_dataset)
        indices = torch.randperm(total_size)[:int(total_size * args.ratio)]
        subset = torch.utils.data.Subset(suspicious_dataset, indices)
        suspicious_dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        suspicious_dataloader = DataLoader(suspicious_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # KMeans聚类分析
    with torch.no_grad():
        logging.info("正在获取数据嵌入表示...")
        rep, x, y_true = get_data(device, suspicious_model, suspicious_dataloader, width)
        
        logging.info(f"对 {rep.shape[0]} 个样本进行聚类...")
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0, n_init=30).fit(rep.cpu().numpy())
        y = kmeans.labels_

        # 分析聚类纯度
        cluster_purities = {}
        first_label = {}
        second_label = {}
        counts_label = {}
        
        for i in range(np.unique(y).shape[0]):
            mask = (y == i)
            cluster_labels = y_true[mask]

            values, counts = torch.unique(cluster_labels, return_counts=True)
            if values.shape[0] == 1:
                first, second = values.item(), "None"
            else:
                second, first = values[torch.argsort(counts)][-2:].tolist()
            
            first_label[i] = first
            second_label[i] = second
            cluster_purity = torch.sum(cluster_labels == first) / len(cluster_labels)
            cluster_purities[i] = cluster_purity.item()
            counts_label[i] = mask.sum()
        
        # 计算总体聚类纯度
        summ = sum(cluster_purities.values())
        logging.info(f"总体聚类纯度: {summ / np.unique(y).shape[0]:.2f}, "
                   f"最小样本数: {min(counts_label.values())}, 最大样本数: {max(counts_label.values())}, "
                   f"总样本数: {sum(counts_label.values())}")

        # 计算聚类中心
        rep_center = torch.empty((len(np.unique(y)), rep.shape[1]))
        y_center = torch.empty(len(np.unique(y)))
        
        for label in np.unique(y):
            rep_center[label, :] = rep[y == label].mean(dim=0)
            y_center[label] = label
        
        if args.knn_center:
            rep_knn, y_knn = rep_center, y_center
        else:
            rep_knn, y_knn = rep, torch.tensor(y)

    # 初始化每个聚类的最佳正则化值列表
    reg_best_list = torch.empty(len(np.unique(y)))
    
    
    # 针对每个聚类生成触发器
    for target in np.unique(y):
        logging.info(f"聚类标签: {target}, 第1类: {first_label[target]}, "
                     f"第2类: {second_label[target]}, 聚类样本数: {counts_label[target]}, "
                     f"纯度: {cluster_purities[target]:.2f}")
        
        mask_best = None
        delta_best = None

        # 准备数据
        rep_target = rep[y == target]
        x_other = x[y != target]
        x_other_indices = torch.randperm(x_other.shape[0])[:x.shape[0] - max(counts_label.values())]
        x_other_sample = x_other[x_other_indices]

        # 初始化触发器参数
        mask = torch.arctanh((torch.rand([1, 1, width, width]) - 0.5) * 2).to(device)
        delta = torch.arctanh((torch.rand([1, 3, width, width]) - 0.5) * 2).to(device)

        mask.requires_grad = True
        delta.requires_grad = True
        opt = optim.Adam([delta, mask], lr=args.lr, betas=(0.5, 0.9))

        reg_best = torch.inf
        early_stop_reg_best = torch.inf

        # 优化参数初始化
        lam = 0
        early_stop_counter = 0
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # 创建训练数据集与数据加载器
        if args.target_center:
            # 使用目标中心
            dataset_train = DatasetClusterFix(rep_center[target], x_other_sample)
        else:
            # 使用随机目标表示
            dataset_train = DatasetCluster(rep_target, x_other_sample)

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # 触发器优化
        for ep in range(args.epoch):
            loss_asr_list, loss_reg_list, loss_list = [], [], []
            
            for n_iter, (images, target_reps) in enumerate(dataloader_train):
                images = images.to(device)
                target_reps = target_reps.to(device)
                
                mask_tanh = torch.tanh(mask) / 2 + 0.5
                delta_tanh = torch.tanh(delta) / 2 + 0.5
                
                X_R = draw(images, mean, std, mask_tanh, delta_tanh)
                z = target_reps
                zt = suspicious_model(X_R)  # 直接使用模型进行前向传递
                
                # 计算损失，使用负余弦相似度
                loss_asr = neg_cosine_similarity(zt, z)
                loss_reg = torch.mean(mask_tanh * delta_tanh)
                loss = loss_asr + lam * loss_reg
                
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()

                loss_asr_list.append(loss_asr.item())
                loss_reg_list.append(loss_reg.item())
                loss_list.append(loss.item())

            # 计算平均损失
            avg_loss_asr = torch.tensor(loss_asr_list).mean()
            avg_loss_reg = torch.tensor(loss_reg_list).mean()
            avg_loss = torch.tensor(loss_list).mean()

            # 使用优化后的最新参数重新计算 mask_tanh 和 delta_tanh，用于评估与保存
            with torch.no_grad():
                mask_tanh_eval = torch.tanh(mask) / 2 + 0.5
                delta_tanh_eval = torch.tanh(delta) / 2 + 0.5

            # 评估攻击成功率
            x_trigger = draw(x.to(device), mean, std, mask_tanh_eval, delta_tanh_eval).detach().to('cpu')
            
            # 创建评估数据加载器
            dataset_eval = DatasetEval(x_trigger, args.knn_sample_num)
            dataloader_eval = DataLoader(
                dataset_eval,
                batch_size=32,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            asr_knn = eval_knn(device, suspicious_model, dataloader_eval, rep_knn, y_knn, target)
            
            # 更新最佳触发器
            if asr_knn > args.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_best = mask_tanh_eval
                delta_best = delta_tanh_eval
                reg_best = avg_loss_reg

            logging.info('步骤: %3d, lam: %.2E, asr: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                         (ep, lam, asr_knn, avg_loss, avg_loss_asr, avg_loss_reg, reg_best))

            # 早停检查
            if args.early_stop:
                if cost_down_flag and cost_up_flag:
                    if reg_best < torch.inf:
                        if reg_best >= args.early_stop_threshold * early_stop_reg_best:
                            early_stop_counter += 1
                        else:
                            early_stop_counter = 0
                    early_stop_reg_best = min(reg_best, early_stop_reg_best)

                    if early_stop_counter >= args.early_stop_patience:
                        logging.info('早停')
                        break

                elif ep == args.start_early_stop_patience and (lam == 0 or reg_best == torch.inf):
                    logging.info('早停')
                    break

            # Lambda自适应调整
            if lam == 0 and asr_knn >= args.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= args.patience:
                    lam = args.lam  # 使用配置参数中的初始lam值
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    logging.info('初始化cost为 %.2E' % lam)
            else:
                cost_set_counter = 0

            if asr_knn >= args.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if lam != 0 and cost_up_counter >= args.patience:
                cost_up_counter = 0
                logging.info('增加cost从 %.2E 到 %.2E' % (lam, lam * args.lam_multiplier_up))
                lam *= args.lam_multiplier_up
                cost_up_flag = True
            elif lam != 0 and cost_down_counter >= args.patience:
                cost_down_counter = 0
                logging.info('减少cost从 %.2E 到 %.2E' % (lam, lam / args.lam_multiplier_up))
                lam /= args.lam_multiplier_up
                cost_down_flag = True

        reg_best_list[target] = reg_best if reg_best != torch.inf else 1

        # 保存触发器
        os.makedirs(args.trigger_path, exist_ok=True)
        torch.save({'mask': mask_best, 'delta': delta_best}, os.path.join(args.trigger_path, f'{target}.pth'))

    # 计算异常值统计
    logging.info(f'reg best list: {reg_best_list}')
    median, mad, min_mad = outlier(reg_best_list)
    logging.info(f'中位数: {median:.2f}, MAD: {mad:.2f}, 异常指数: {min_mad:.2f}')

    # 返回异常聚类（选择 reg_best 最小的聚类，与原始实现逻辑一致）
    anomaly_index = torch.argmin(reg_best_list).item()
    logging.info(f'检测到的异常聚类: {anomaly_index}')

    # 将属于异常聚类的样本标记为有毒样本
    is_poisoned = (torch.tensor(y) == anomaly_index)
    poisoned_indices = np.where(is_poisoned)[0].tolist()
    clean_indices = np.where(~is_poisoned)[0].tolist()
    
    logging.info(f'检测到 {len(poisoned_indices)} 个潜在有毒样本 ({len(poisoned_indices) / len(y) * 100:.2f}%)')
    
    # 如果有测试数据集，评估检测性能
    evaluation_results = {}
    if clean_test_dataset is not None and poisoned_test_dataset is not None:
        logging.info("评估检测性能...")
        
        # 创建测试数据加载器
        clean_dataloader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        poisoned_dataloader = DataLoader(poisoned_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        # 获取干净测试数据和有毒测试数据的嵌入表示
        clean_rep, clean_x, clean_y = get_data(device, suspicious_model, clean_dataloader, width)
        poisoned_rep, poisoned_x, poisoned_y = get_data(device, suspicious_model, poisoned_dataloader, width)
        
        # 使用之前训练的KMeans模型预测样本类别
        clean_clusters = kmeans.predict(clean_rep.cpu().numpy())
        poisoned_clusters = kmeans.predict(poisoned_rep.cpu().numpy())
        
        # 将预测为异常类别的样本标记为有毒样本
        clean_predicted_poisoned = (clean_clusters == anomaly_index)
        poisoned_predicted_poisoned = (poisoned_clusters == anomaly_index)
        
        # 计算真阳性、假阳性、假阴性
        true_positives = np.sum(poisoned_predicted_poisoned)  # 被正确标记的有毒样本数
        false_positives = np.sum(clean_predicted_poisoned)    # 被错误标记为有毒的干净样本数
        false_negatives = len(poisoned_clusters) - true_positives  # 未被检测出的有毒样本数
        
        # 计算指标
        recall = true_positives / len(poisoned_clusters) if len(poisoned_clusters) > 0 else 0  # 召回率
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0  # 精确率
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # F1分数
        
        # 干净样本被误判为有毒的比例
        false_positive_rate = false_positives / len(clean_clusters) if len(clean_clusters) > 0 else 0
        
        evaluation_results = {
            'anomaly_cluster': anomaly_index,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'recall': float(recall),
            'precision': float(precision),
            'f1_score': float(f1_score),
            'false_positive_rate': float(false_positive_rate),
            'clean_samples_total': len(clean_clusters),
            'poisoned_samples_total': len(poisoned_clusters),
        }
        
        logging.info(f"检测结果: 异常聚类 = {anomaly_index}")
        logging.info(f"召回率 (真阳性/总恶意样本) = {recall:.4f} ({true_positives}/{len(poisoned_clusters)})")
        logging.info(f"精确率 (真阳性/总召回) = {precision:.4f} ({true_positives}/{true_positives + false_positives})")
        logging.info(f"F1分数 = {f1_score:.4f}")
        logging.info(f"干净样本误判率 = {false_positive_rate:.4f} ({false_positives}/{len(clean_clusters)})")

    # 准备结果字典
    elapsed_time = time.time() - start_time
    result_dict = {
        'anomaly_cluster': anomaly_index,
        'clean_indices': clean_indices,
        'poisoned_indices': poisoned_indices,
        'poisoned_labels': first_label[anomaly_index] if anomaly_index in first_label else None,
        'clean_set_size': len(clean_indices),
        'poisoned_set_size': len(poisoned_indices),
        'reg_best_list': reg_best_list.tolist(),
        'median': median.item(),
        'mad': mad.item(), 
        'min_mad': min_mad.item(),
        'elapsed_time': elapsed_time,
        'evaluation_results': evaluation_results
    }

    # 保存结果
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    with open(os.path.join(args.output_dir, "results", "ssl_cleanse_results.pkl"), 'wb') as f:
        pickle.dump(result_dict, f)
    
    # 保存清洗后的文件列表
    if hasattr(suspicious_dataset, 'file_list'):
        filtered_file_list = [suspicious_dataset.file_list[i] for i in clean_indices]
        filtered_file_path = os.path.join(args.output_dir, 'filtered_file_list.txt')
        with open(filtered_file_path, 'w') as f:
            for line in filtered_file_list:
                f.write(f"{line}\n")
        logging.info(f"清洗后的文件列表已保存到: {filtered_file_path}")
    
    logging.info(f"SSL-Cleanse检测完成，用时 {elapsed_time:.2f} 秒")
    
    return result_dict, clean_indices, poisoned_indices 