"""No docstring provided.
    No docstring provided.

    No docstring provided.
"""

import os
import copy
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ot
import kmeans_pytorch

from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.ssl_trainers.utils import AverageMeter, ProgressMeter

from ssl_backdoor.attacks.badencoder.badencoder import train_downstream_classifier

from ssl_backdoor.attacks.drupe.metric_logger import MetricLogger, compute_linear_separability, extract_features

from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Tuple


def log_info(message, args=None):
    """No docstring provided.
        No docstring provided.
    
    Args:
        No docstring provided.
        No docstring provided.
    """
    if args is not None and hasattr(args, 'logger_file'):
        args.logger_file.write(f"{message}\n")
        args.logger_file.flush()
    print(message)


def set_seed(seed):
    """No docstring provided.."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_drupe(backdoored_encoder, clean_encoder, data_loader, train_optimizer, epoch, args, 
               warm_up=False, get_clean_dev=False, cal_cluster_based_dist=False):
    """No docstring provided.
        No docstring provided.
    
    Args:
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        
    Returns:
        No docstring provided.
    """
    global patience, cost_multiplier_up, cost_multiplier_down, init_cost, cost, cost_up_counter, cost_down_counter
    global init_cost_1, cost_1, cost_up_counter_1, cost_down_counter_1
    
    log_info(f"Current cost factors: cost={cost}, cost_1={cost_1}", args)
    

    backdoored_encoder.train()


    for module in backdoored_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


    clean_encoder.eval()


    losses = AverageMeter('Loss', '.4f')
    losses_0 = AverageMeter('Loss_0', '.4f')
    losses_1 = AverageMeter('Loss_1', '.4f')
    losses_2 = AverageMeter('Loss_2', '.4f')
    losses_3 = AverageMeter('Loss_3', '.4f')
    losses_b2c = AverageMeter('Loss_b2c', '.4f')
    losses_b2c_d_std = AverageMeter('Loss_b2c_d_std', '.4f')
    sim_backdoor2backdoor = AverageMeter('Sim_b2b', '.4f')
    sim_clean2clean = AverageMeter('Sim_c2c', '.4f')
    

    wasserstein_distances = AverageMeter('WD', '.6f')

    js_divergences = AverageMeter('JSD', '.6f') # Jensen-Shannon Divergence
    js_dims_calculated = AverageMeter('JSD_dims', '.0f') # Number of dimensions used for JSD
    
    meters = [losses, losses_0, losses_1, losses_2, losses_3, 
              losses_b2c, losses_b2c_d_std, sim_backdoor2backdoor, sim_clean2clean, wasserstein_distances,
              js_divergences, js_dims_calculated]
    progress = ProgressMeter(len(data_loader), meters, prefix=f"Epoch: [{epoch}/{args.epochs}]")
    

    if hasattr(args, 'collect_features') and args.collect_features:
        shadow_features_list = []
        target_features_list = []
    

    all_feature_raw_tensors = []
    all_feature_backdoor_tensors = []


    if get_clean_dev:
        clean_dev_list = []
        for img_clean, img_backdoor_list, reference_list, reference_aug_list in tqdm(data_loader, desc="Computing clean feature standard deviation"):
            img_clean = img_clean.cuda(non_blocking=True)
            
            with torch.no_grad():
                clean_feature_raw = clean_encoder(img_clean)
            dev_clean = clean_feature_raw.std(0).mean()
            clean_dev_list.append(dev_clean.item())
        
        args.clean_dev_mean = sum(clean_dev_list) / len(clean_dev_list)
        log_info(f"Mean clean feature std: {args.clean_dev_mean:.6f}", args)
    

    for i, (img_clean, img_backdoor_list, reference_list, reference_aug_list) in enumerate(tqdm(data_loader, desc=f"Training DRUPE: Epoch {epoch}/{args.epochs}")):

        img_clean = img_clean.cuda(non_blocking=True)
        reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
        
        for reference in reference_list:
            reference_cuda_list.append(reference.cuda(non_blocking=True))
        for reference_aug in reference_aug_list:
            reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))
        

        clean_feature_reference_list = []
        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            for img_reference in reference_cuda_list:
                clean_feature_reference = clean_encoder(img_reference)
                clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                clean_feature_reference_list.append(clean_feature_reference)


        feature_raw = backdoored_encoder(img_clean)
        feature_raw_before_normalize = feature_raw
        feature_raw = F.normalize(feature_raw, dim=-1)


        feature_backdoor_list = []
        feature_backdoor_before_normalize_list = []
        for img_backdoor in img_backdoor_cuda_list:
            feature_backdoor = backdoored_encoder(img_backdoor)
            feature_backdoor_before_normalize_list.append(feature_backdoor)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1)
            feature_backdoor_list.append(feature_backdoor)


        feature_reference_list = []
        feature_reference_before_normalize_list = []
        for img_reference in reference_cuda_list:
            feature_reference = backdoored_encoder(img_reference)
            feature_reference_before_normalize_list.append(feature_reference)
            feature_reference = F.normalize(feature_reference, dim=-1)
            feature_reference_list.append(feature_reference)


        feature_reference_aug_list = []
        for img_reference_aug in reference_aug_cuda_list:
            feature_reference_aug = backdoored_encoder(img_reference_aug)
            feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
            feature_reference_aug_list.append(feature_reference_aug)
        

        if hasattr(args, 'collect_features') and args.collect_features:

            shadow_features_list.append(feature_backdoor_before_normalize_list[0].clone().detach().cpu())

            target_features_list.append(feature_raw_before_normalize.clone().detach().cpu())
        

        all_feature_raw_tensors.append(feature_raw_before_normalize.detach().cpu())
        if len(feature_backdoor_before_normalize_list) > 0:
            all_feature_backdoor_tensors.append(feature_backdoor_before_normalize_list[0].detach().cpu())
        

        loss_0_list, loss_1_list = [], []
        loss_0_list_cal = []
        loss_b2c_list = []
        sim_backdoor2backdoor_list = []
        sim_clean2clean_list = []
        loss_local_dist_list = []
        

        if i == 0 and cal_cluster_based_dist:
            with torch.no_grad():

                feature_size = feature_raw_before_normalize.shape[1]
                

                cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(
                    X=feature_raw_before_normalize, num_clusters=2, distance='cosine',
                    tol=1e-3, device=torch.device('cuda:0')
                )
                

                cluster_ids_x = cluster_ids_x.unsqueeze(-1).bool().expand(
                    feature_raw_before_normalize.shape[0], feature_size
                ).cuda()
                cluster_ids_x2 = ~cluster_ids_x
                
                if torch.masked_select(feature_raw_before_normalize, cluster_ids_x2).reshape(-1, feature_size)[:50].shape[0] == 1:
                    continue
                

                used_num = min(
                    50,
                    torch.masked_select(feature_raw_before_normalize, cluster_ids_x).reshape(-1, feature_size).shape[0],
                    torch.masked_select(feature_raw_before_normalize, cluster_ids_x2).reshape(-1, feature_size).shape[0]
                )
                
                distance_base = ot.sliced_wasserstein_distance(
                    torch.masked_select(feature_raw_before_normalize, cluster_ids_x).reshape(-1, feature_size)[:used_num].cuda(),
                    torch.masked_select(feature_raw_before_normalize, cluster_ids_x2).reshape(-1, feature_size)[:used_num].cuda()
                )
                

                dis_GTbackdoor2clean = ot.sliced_wasserstein_distance(
                    feature_backdoor_before_normalize_list[0], feature_raw_before_normalize
                )
                

                dis_GTbackdoor2clean_cluster_based = dis_GTbackdoor2clean / distance_base
                
                log_info(f"Base distance: {distance_base:.6f}", args)
                log_info(f"Cluster-based distribution distance: {dis_GTbackdoor2clean_cluster_based:.6f}", args)
        

        for _index in range(len(feature_reference_list)):
            loss_0_list.append(torch.sum(feature_backdoor_list[_index] * feature_reference_list[_index], dim=-1).unsqueeze(0))
            loss_0_list_cal.append(torch.sum(feature_backdoor_list[_index] * feature_reference_list[_index], dim=-1).mean())

            loss_1_list.append(-torch.sum(feature_reference_aug_list[_index] * clean_feature_reference_list[_index], dim=-1).mean())
        

        loss_0_list_tensor = torch.cat(loss_0_list, 0)
        std_refs = loss_0_list_tensor.mean(-1).std()
        loss_0_list_tensor_min, index = torch.max(loss_0_list_tensor, dim=0)
        

        to_ref_list = []
        for _index in range(len(loss_0_list_tensor)):
            to_ref_list.append(torch.argwhere(index == _index).squeeze().tolist())
        

        if args.mode == "badencoder":
            loss_0 = -sum(loss_0_list_cal) / len(loss_0_list_cal)
        else:
            loss_0 = -loss_0_list_tensor_min.mean()
        

        dis_GTbackdoor2clean = ot.sliced_wasserstein_distance(
            feature_backdoor_before_normalize_list[0], feature_raw_before_normalize
        )
        total_b2c = dis_GTbackdoor2clean
        

        wasserstein_distances.update(dis_GTbackdoor2clean.item())
        

        backdoored_clean_dev = feature_raw_before_normalize.std(0).mean()
        

        loss_b2c_list.append(total_b2c)
        

        sim_matrix = torch.mm(feature_backdoor_list[0], feature_backdoor_list[0].T)
        distance = (sim_matrix - torch.diag_embed(sim_matrix.diag())).mean()
        sim_backdoor2backdoor_list.append(distance)
        

        sim_matrix = torch.mm(feature_raw, feature_raw.T)
        distance = (sim_matrix - torch.diag_embed(sim_matrix.diag())).mean()
        sim_clean2clean_list.append(distance)
        

        loss_2 = -torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()
        

        loss_1 = sum(loss_1_list) / len(loss_1_list)
        cur_sim_backdoor2backdoor = sum(sim_backdoor2backdoor_list) / len(sim_backdoor2backdoor_list)
        cur_sim_clean2clean = sum(sim_clean2clean_list) / len(sim_clean2clean_list)
        

        loss_3_list = []
        for _index in range(len(feature_reference_list)):
            for _index_2 in range(_index + 1, len(feature_reference_list)):
                loss_3_list.append(
                    torch.sum(
                        feature_reference_list[_index] * feature_reference_list[_index_2],
                        dim=-1
                    ).mean()
                )
        
        loss_3 = sum(loss_3_list) / len(loss_3_list) if loss_3_list else torch.tensor(0.0).cuda()
        

        loss_b2c = sum(loss_b2c_list) / len(loss_b2c_list)
        

        if args.mode == "drupe":
            if warm_up:

                loss = args.lambda1 * loss_1 + args.lambda2 * loss_2 + 0.5 * loss_3
                if loss_3 < 0.2:
                    loss = loss - 0.2 * loss_3
            else:

                if args.encoder_usage_info == "imagenet":
                    stage_1_epoch = 3
                else:
                    stage_1_epoch = 5

                if epoch < stage_1_epoch:

                    loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2 + 2 * std_refs
                    if loss_3 > 0.5:
                        loss = loss + 1 * loss_3
                    elif loss_3 > 0.4:
                        loss = loss + 0.2 * loss_3
                else:

                    loss = (loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2 + 
                           cost * (std_refs + cur_sim_backdoor2backdoor) + 
                           cost_1 * (loss_b2c / backdoored_clean_dev) + 
                           0.5 * std_refs)
                    

                    if std_refs > 0.1:
                        loss = loss + 1.5 * std_refs
                    

                    if loss_3 > 0.5:
                        loss = loss + 1 * loss_3
                    elif loss_3 > 0.4:
                        loss = loss + 0.2 * loss_3
        
        elif args.mode == "wb":

            loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2 + cost * loss_b2c
        
        elif args.mode == "badencoder":

            loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2
        
        else:
            raise ValueError(f"Invalid mode: {args.mode}")


        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        

        losses.update(loss.item())
        losses_0.update(loss_0.item())
        losses_1.update(loss_1.item())
        losses_2.update(loss_2.item())
        losses_3.update(loss_3.item())
        losses_b2c.update(loss_b2c.item())
        losses_b2c_d_std.update((loss_b2c / backdoored_clean_dev).item())
        sim_backdoor2backdoor.update(cur_sim_backdoor2backdoor.item())
        sim_clean2clean.update(cur_sim_clean2clean.item())
        

        if i % args.print_freq == 0:
            progress.display(i)
            

            if hasattr(args, 'logger_file'):
                args.logger_file.write(
                    f"E:[{epoch}/{args.epochs}][{i}/{len(data_loader)}],lr:{train_optimizer.param_groups[0]['lr']:.4f},"
                    f"Sb2b:{sim_backdoor2backdoor.avg:.4f},Sc2c:{sim_clean2clean.avg:.4f},"
                    f"l:{losses.avg:.4f},l0:{losses_0.avg:.4f},l1:{losses_1.avg:.4f},"
                    f"l2:{losses_2.avg:.4f},l3:{losses_3.avg:.4f},b2c:{losses_b2c.avg:.4f},"
                    f"b2c/std:{losses_b2c_d_std.avg:.4f},WD:{wasserstein_distances.avg:.6f}"
                    f",JSD:{js_divergences.avg:.6f},JSD_dims:{js_dims_calculated.avg:.0f}\n"
                )
                args.logger_file.flush()
    
    print("Start computing linear separability")

    linear_separability = 0.0
    if hasattr(args, 'collect_features') and args.collect_features:

        shadow_features = torch.cat(shadow_features_list, dim=0)
        target_features = torch.cat(target_features_list, dim=0)
        

        linear_separability = compute_linear_separability(shadow_features, target_features)
        log_info(f"Epoch {epoch}: Linear separability = {linear_separability:.4f}", args)
    


    avg_js_dist = 0.0
    num_js_dims = 0
    # if len(all_feature_raw_tensors) > 0 and len(all_feature_backdoor_tensors) > 0:

    #     all_feature_raw_np = torch.cat(all_feature_raw_tensors, dim=0).numpy()
    #     all_feature_backdoor_np = torch.cat(all_feature_backdoor_tensors, dim=0).numpy()
    #     




    #     avg_js_dist, num_js_dims = calculate_js_divergence_per_dim(
    #         all_feature_raw_np,
    #         all_feature_backdoor_np,
    #         max_dims_to_sample=max_kde_dims,
    #         random_seed_for_sampling=js_seed
    #     )
    #     
    #     js_divergences.update(avg_js_dist, n=1) # n=1 because it's an epoch-level average
    #     js_dims_calculated.update(num_js_dims, n=1)


    #     if hasattr(args, 'logger_file'):

    #         args.logger_file.flush()
    # -------------------------------------------------------


    if not warm_up and epoch > args.fix_epoch:

        if args.encoder_usage_info in ["imagenet"]:
            l0_threshold = -0.91
        else:
            l0_threshold = -0.96
        

        if (losses_0.avg < l0_threshold and losses_1.avg < -0.9 and losses_2.avg < -0.9):
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        if cost_up_counter >= patience:
            cost_up_counter = 0
            if cost == 0:
                cost = init_cost
            else:
                cost *= cost_multiplier_up
        elif cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= cost_multiplier_down
        

        if args.encoder_usage_info in ["CLIP"]:
            b2b_sim_threshold = 0.8
        else:
            b2b_sim_threshold = 0.6
            

        if (losses_0.avg < l0_threshold and 
            losses_1.avg < -0.9 and 
            losses_2.avg < -0.9 and 
            sim_backdoor2backdoor.avg < b2b_sim_threshold):
            
            cost_up_counter_1 += 1
            cost_down_counter_1 = 0

            if sim_backdoor2backdoor.avg < (b2b_sim_threshold - 0.1):
                cost_up_counter -= 1

            args.measure = loss_b2c.avg
        else:
            cost_up_counter_1 = 0
            cost_down_counter_1 += 1

        if cost_up_counter_1 >= patience:
            cost_up_counter_1 = 0
            if cost_1 == 0:
                cost_1 = init_cost_1
            else:
                cost_1 *= cost_multiplier_up
        elif cost_down_counter_1 >= patience:
            cost_down_counter_1 = 0
            cost_1 /= cost_multiplier_down
    
    return losses.avg, wasserstein_distances.avg, linear_separability, js_divergences.avg, js_dims_calculated.avg


def run_drupe(args, pretrained_encoder, shadow_dataset=None, memory_dataset=None, 
             test_data_clean=None, test_data_backdoor=None, downstream_train_dataset=None):
    """No docstring provided.
        No docstring provided.
    
    Args:
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
    
    Returns:
        No docstring provided.
    """
    start_time = time.time()
    

    set_seed(args.seed)
    

    os.makedirs(args.output_dir, exist_ok=True)
    

    train_loader = DataLoader(
        shadow_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    

    backdoored_model = copy.deepcopy(pretrained_encoder)
    


    # if args.encoder_usage_info in ['cifar10', 'stl10']:
    #     optimizer = torch.optim.SGD(backdoored_model.f.parameters(), lr=args.lr, 
    #                                weight_decay=args.weight_decay, momentum=args.momentum)
    # elif args.encoder_usage_info in ['imagenet', 'CLIP']:
    #     optimizer = torch.optim.SGD(backdoored_model.visual.parameters(), lr=args.lr, 
    #                                weight_decay=args.weight_decay, momentum=args.momentum)
    # else:

    assert not hasattr(backdoored_model, 'visual'), "backdoored_model should not expose a visual attribute yet"
    optimizer = torch.optim.SGD(backdoored_model.parameters(), lr=args.lr, 
                               weight_decay=args.weight_decay, momentum=args.momentum)
    

    

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    

    global patience, cost_multiplier_up, cost_multiplier_down, init_cost, cost, cost_up_counter, cost_down_counter
    global init_cost_1, cost_1, cost_up_counter_1, cost_down_counter_1
    
    patience = 1
    cost_multiplier_up = 1.25
    cost_multiplier_down = 1.25 ** 1.25
    

    if args.encoder_usage_info in ["CLIP"]:
        init_cost = 0.01
        init_cost_1 = 0.0001
    else:
        init_cost = 0.1
        init_cost_1 = 0.001
    
    cost = 0
    cost_up_counter = 0
    cost_down_counter = 0
    
    cost_1 = 0
    cost_up_counter_1 = 0
    cost_down_counter_1 = 0
    
    args.measure = 0
    measure_best = float('inf')
    

    args.collect_features = True
    

    metric_logger = MetricLogger()


    log_info("\n====== Initial downstream evaluation ======", args)

    if all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
        try:
            init_results = train_downstream_classifier(
                args, backdoored_model, downstream_train_dataset,
                test_data_clean, test_data_backdoor
            )
            log_info(f"Initial BA={init_results['BA']:.2f}%, ASR={init_results['ASR']:.2f}%", args)
        except Exception as e:
            log_info(f"Initial downstream evaluation failed: {e}", args)
    else:
        log_info("Downstream datasets are missing; skipping initial evaluation.", args)
    

    for epoch in range(args.epochs):
        log_info("=================================================", args)
        

        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            warm_up = (epoch < args.warm_up_epochs)
            get_clean_dev = (epoch == 0)
            cal_cluster_based_dist = (epoch % 10 == 0)
            
            train_loss, wasserstein_distance, linear_separability, js_divergence, num_js_dims = train_drupe(
                backdoored_model.f, pretrained_encoder.f, train_loader, 
                optimizer, epoch, args, warm_up, get_clean_dev, cal_cluster_based_dist
            )
            
        elif args.encoder_usage_info in ['imagenet', 'CLIP']:

            if args.encoder_usage_info == 'imagenet':
                warm_up_epoch = 2
            else:  # CLIP
                warm_up_epoch = 1
                
            warm_up = (epoch < warm_up_epoch)
            get_clean_dev = (epoch == 0)
            cal_cluster_based_dist = (epoch % 10 == 0)
            
            assert not hasattr(backdoored_model, 'visual'), "backdoored_model should not expose a visual attribute yet"
            train_loss, wasserstein_distance, linear_separability, js_divergence, num_js_dims = train_drupe(
                backdoored_model, pretrained_encoder, train_loader, 
                optimizer, epoch, args, warm_up, get_clean_dev, cal_cluster_based_dist
            )
            
        else:
            raise NotImplementedError(f"Unsupported encoder_usage_info: {args.encoder_usage_info}")
        

        metric_logger.log_epoch_metrics(epoch, wasserstein_distance, linear_separability, js_divergence, num_js_dims)
        

        log_info(f"Current metric: {args.measure}, Best metric: {measure_best}", args)
        if epoch > 24 and args.measure < measure_best:
            measure_best = args.measure
            best_checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': backdoored_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, best_checkpoint_path)
            log_info(f"Saved best checkpoint to:  {best_checkpoint_path}, Metric: {measure_best:.6f}", args)
        

        if (epoch+1) % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': backdoored_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            log_info(f"Saved checkpoint to:  {checkpoint_path}", args)


            log_info("\n====== Periodic downstream evaluation ======", args)
            if all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
                try:

                    model_results = train_downstream_classifier(
                        args, backdoored_model, downstream_train_dataset,
                        test_data_clean, test_data_backdoor
                    )
                    log_info(f"Stage {epoch}/{args.epochs} Downstream evaluation finished for stage BA={model_results['BA']:.2f}%, ASR={model_results['ASR']:.2f}%", args)
                except Exception as e:
                    log_info(f"Downstream evaluation failed: {e}", args)
            else:
                log_info("Downstream datasets are missing; skipping evaluation.", args)
    

    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        backdoored_model.load_state_dict(checkpoint['state_dict'])
        log_info("Loaded best checkpoint for final evaluation", args)
    else:
        log_info("Best checkpoint not found; using last trained model for final evaluation", args)

    

    final_results = None
    log_info("\n====== Final downstream evaluation ======", args)

    if all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
        try:

            final_results = train_downstream_classifier(
                args, backdoored_model, downstream_train_dataset, 
                test_data_clean, test_data_backdoor
            )
            log_info(f"Final results: BA={final_results['BA']:.2f}%, ASR={final_results['ASR']:.2f}%", args)
        except Exception as e:
            log_info(f"Final downstream evaluation failed: {e}", args)
    else:
        log_info("Downstream datasets are missing; cannot run final evaluation.", args)

    
    elapsed_time = time.time() - start_time
    log_info(f"DRUPE training completed, elapsed time:  {elapsed_time:.2f} seconds", args)
    
    return backdoored_model, final_results 
