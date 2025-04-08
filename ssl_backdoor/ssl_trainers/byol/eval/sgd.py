import torch
import torch.nn as nn
import torch.optim as optim

import os
from tqdm import trange, tqdm

# Modified from original
# def eval_sgd(x_train, y_train, x_test, y_test, x_test_p=None, y_test_p=None, evaluate=False, topk=[1, 5], epoch=100):
#     """ linear classifier accuracy (sgd) """

#     lr_start, lr_end = 1e-2, 1e-6
#     gamma = (lr_end / lr_start) ** (1 / epoch)
#     output_size = x_train.shape[1]
#     num_class = y_train.max().item() + 1

#     clf = nn.Linear(output_size, num_class)
#     clf.cuda()
#     clf.train()
    
#     optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
#     criterion = nn.CrossEntropyLoss()

#     from torch.utils.data import DataLoader, TensorDataset
#     # 假设 x_train 和 y_train 已经定义
#     dataset = TensorDataset(x_train, y_train)
#     dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
#     best_acc = 0.0  # 初始化最佳精度
#     best_clf_state_dict = clf.state_dict()

#     pbar = tqdm(range(epoch))
#     for ep in pbar:
#         for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
#             optimizer.zero_grad()
#             criterion(clf(batch_x), batch_y).backward()
#             optimizer.step()
#         scheduler.step()

#         # 在每个epoch结束时，计算测试集的精度
#         clf.eval()
#         with torch.no_grad():
#             y_pred = clf(x_test)
#         pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
#         acc = {
#             t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
#             for t in topk
#         }
#         # 如果当前精度高于最佳精度，则保存模型并更新最佳精度
#         if acc[1] > best_acc:
#             best_acc = acc[1]
#             best_clf_state_dict = clf.state_dict()

#         pbar.set_postfix({'best_acc': best_acc})
#         clf.train()           

#     clf.load_state_dict(best_clf_state_dict)
#     clf.eval()
#     with torch.no_grad():
#         y_pred = clf(x_test)
#     pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
#     acc = {
#         t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
#         for t in topk
#     }
    
#     if not evaluate:
#         return acc
#     else:
#         clf.eval()
#         with torch.no_grad():
#             y_pred = clf(x_test)
#         pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
#         acc = {
#             t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
#             for t in topk
#         }

#         with torch.no_grad():
#             y_pred_p = clf(x_test_p)
#         pred_top_p = y_pred_p.topk(max(topk), 1, largest=True, sorted=True).indices
#         acc_p = {
#             t: (pred_top_p[:, :t] == y_test_p[..., None]).float().sum(1).mean().cpu().item()
#             for t in topk
#         }
#         return acc, y_pred, y_test, acc_p, y_pred_p, y_test_p

def get_data(model, loader, output_size, device):
    """ encodes whole dataset into embeddings """
    n_total_samples = len(loader.dataset)
    xs = torch.empty(n_total_samples, output_size, dtype=torch.float32, device=device)
    ys = torch.empty(n_total_samples, dtype=torch.long, device=device)
    start_idx = 0
    added_count = 0  # 记录添加了多少个数据
    
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            batch_size = x.shape[0]
            end_idx = start_idx + batch_size
            
            xs[start_idx:end_idx] = model(x)
            ys[start_idx:end_idx] = y.to(device)
            
            start_idx = end_idx
            added_count += batch_size  # 更新添加了多少个数据

    # 删除未使用的部分
    xs = xs[:added_count]
    ys = ys[:added_count]

    return xs, ys

def eval_sgd(model, clean_set, test_set, test_poisoned_set, out_size, device, evaluate=True, topk=[1, 5], epoch=100):
    """ linear classifier accuracy (sgd) """

    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = out_size
    num_class = 100

    x_test, y_test = get_data(model, test_set, out_size, device)
    x_test_p, y_test_p = get_data(model, test_poisoned_set, out_size, device)

    clf = nn.Linear(output_size, num_class)
    clf.cuda()
    clf.train()
    
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    from torch.utils.data import DataLoader, TensorDataset
    # 假设 x_train 和 y_train 已经定义
    
    best_acc = 0.0  # 初始化最佳精度
    best_clf_state_dict = clf.state_dict()

    pbar = tqdm(range(epoch))
    for ep in pbar:
        x_train, y_train = get_data(model, clean_set, out_size, device)
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            criterion(clf(batch_x), batch_y).backward()
            optimizer.step()
        scheduler.step()

        # 在每个epoch结束时，计算测试集的精度
        clf.eval()
        with torch.no_grad():
            y_pred = clf(x_test)
        pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
        acc = {
            t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
            for t in topk
        }
        # 如果当前精度高于最佳精度，则保存模型并更新最佳精度
        if acc[1] > best_acc:
            best_acc = acc[1]
            best_clf_state_dict = clf.state_dict()

        pbar.set_postfix({'best_acc': best_acc})
        clf.train()           

    clf.load_state_dict(best_clf_state_dict)
    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = {
        t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }
    
    if not evaluate:
        return acc
    else:
        clf.eval()
        with torch.no_grad():
            y_pred = clf(x_test)
        pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
        acc = {
            t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
            for t in topk
        }

        with torch.no_grad():
            y_pred_p = clf(x_test_p)
        pred_top_p = y_pred_p.topk(max(topk), 1, largest=True, sorted=True).indices
        acc_p = {
            t: (pred_top_p[:, :t] == y_test_p[..., None]).float().sum(1).mean().cpu().item()
            for t in topk
        }
        return acc, y_pred, y_test, acc_p, y_pred_p, y_test_p