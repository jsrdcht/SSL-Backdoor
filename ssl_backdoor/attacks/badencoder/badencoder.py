"""No docstring provided.
    No docstring provided.

    No docstring provided.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import ot
import kmeans_pytorch

from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.ssl_trainers.utils import AverageMeter, ProgressMeter

def set_seed(seed):
    """No docstring provided.."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_badencoder(backdoored_encoder, clean_encoder, data_loader, train_optimizer, epoch, args, warm_up=False):
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

    backdoored_encoder.train()
    

    for module in backdoored_encoder.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                             nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, 
                             nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LocalResponseNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)
            module.eval()


    clean_encoder.eval()


    losses = AverageMeter('Loss', '.4f')
    losses_0 = AverageMeter('Loss_0', '.4f')
    losses_1 = AverageMeter('Loss_1', '.4f')
    losses_2 = AverageMeter('Loss_2', '.4f')
    wasserstein_distances = AverageMeter('WD', '.6f')
    
    meters = [losses, losses_0, losses_1, losses_2]
    meters.append(wasserstein_distances)
    
    progress = ProgressMeter(len(data_loader), meters, prefix=f"Epoch: [{epoch}/{args.epochs}]")
    

    for i, (img_clean, img_backdoor_list, reference_list, reference_aug_list) in enumerate(data_loader):

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

        if len(feature_backdoor_before_normalize_list) > 0:
            dis_backdoor2clean = ot.sliced_wasserstein_distance(
                feature_backdoor_before_normalize_list[0].clone().detach().cpu(), feature_raw_before_normalize.clone().detach().cpu()
            )
            wasserstein_distances.update(dis_backdoor2clean.item())
            

        loss_0_list = []
        loss_1_list = []
        

        for j in range(len(feature_reference_list)):
            loss_0_list.append(-torch.sum(feature_backdoor_list[j] * feature_reference_list[j], dim=-1).mean())

            loss_1_list.append(-torch.sum(feature_reference_aug_list[j] * clean_feature_reference_list[j], dim=-1).mean())
        
        loss_0 = sum(loss_0_list)/len(loss_0_list)
        loss_1 = sum(loss_1_list)/len(loss_1_list)
        

        loss_2 = -torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()
        
        

        loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2


        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        

        losses.update(loss.item())
        losses_0.update(loss_0.item())
        losses_1.update(loss_1.item())
        losses_2.update(loss_2.item())
        

        if i % args.print_freq == 0:
            progress.display(i)
            

            if hasattr(args, 'logger_file'):
                args.logger_file.write(f"Epoch: [{epoch}/{args.epochs}][{i}/{len(data_loader)}] "
                                     f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                                     f"Loss_0: {losses_0.val:.4f} ({losses_0.avg:.4f}) "
                                     f"Loss_1: {losses_1.val:.4f} ({losses_1.avg:.4f}) "
                                     f"Loss_2: {losses_2.val:.4f} ({losses_2.avg:.4f}) "
                                     f" WD: {wasserstein_distances.val:.6f} ({wasserstein_distances.avg:.6f})"
                                     f"\n")
                args.logger_file.flush()

    if hasattr(args, 'current_wasserstein_distance'):
        args.current_wasserstein_distance = wasserstein_distances.avg
    else:
        args.current_wasserstein_distance = wasserstein_distances.avg

    return losses.avg


class NeuralNet(nn.Module):
    """No docstring provided.."""
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            # self.layers.append(nn.ReLU())
            prev_size = hidden_size
            
        self.layers.append(nn.Linear(prev_size, output_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def predict_feature(encoder, data_loader):
    """No docstring provided.."""
    encoder.eval()
    feature_bank = []
    label_bank = []
    
    with torch.no_grad():
        for (data, target) in tqdm(data_loader, desc="Extracting features"):
            data = data.cuda(non_blocking=True)
            feature = encoder(data).flatten(start_dim=1)
            feature_bank.append(feature.cpu())
            label_bank.append(target)
            
    feature_bank = torch.cat(feature_bank, dim=0)
    label_bank = torch.cat(label_bank, dim=0)
    
    return feature_bank, label_bank


def create_torch_dataloader(features, labels, batch_size):
    """No docstring provided.."""
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    dataset = FeatureDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def net_train(model, data_loader, optimizer, epoch, criterion):
    """No docstring provided.."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (features, labels) in enumerate(data_loader):
        features = features.cuda()
        labels = labels.cuda()
        

        outputs = model(features)
        loss = criterion(outputs, labels)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(data_loader)
    train_acc = 100. * correct / total
    
    print(f'Train Epoch: {epoch} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
    return train_loss, train_acc


def net_test_with_logger(args, model, data_loader, epoch, criterion, metric_name='Accuracy'):
    """No docstring provided.."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.cuda()
            labels = labels.cuda()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(data_loader)
    test_acc = 100. * correct / total
    
    print(f'Test Epoch: {epoch} | Loss: {test_loss:.4f} | {metric_name}: {test_acc:.2f}%')
    if hasattr(args, 'logger_file'):
        args.logger_file.write(f'Test Epoch: {epoch} | Loss: {test_loss:.4f} | {metric_name}: {test_acc:.2f}%\n')
    
    return test_loss, test_acc


def train_downstream_classifier(args, model, train_data, test_data_clean, test_data_backdoor):
    """No docstring provided.."""

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size_downstream, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader_clean = DataLoader(
        test_data_clean, batch_size=args.batch_size_downstream, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader_backdoor = DataLoader(
        test_data_backdoor, batch_size=args.batch_size_downstream, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    num_of_classes = dataset_params[args.downstream_dataset]['num_classes']
    print(f"Downstream classification number of classes:  {num_of_classes}")
    

    if args.encoder_usage_info in ['CLIP', 'imagenet']:

        # feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader)
        # feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader_clean)
        # feature_bank_backdoor, label_bank_backdoor = predict_feature(model.visual, test_loader_backdoor)

        feature_bank_training, label_bank_training = predict_feature(model, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model, test_loader_backdoor)
    else:
        feature_bank_training, label_bank_training = predict_feature(model.f, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model.f, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.f, test_loader_backdoor)
    

    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size_downstream)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size_downstream)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size_downstream)
    
    input_size = feature_bank_training.shape[1]
    criterion = nn.CrossEntropyLoss()
    

    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_downstream)
    

    for epoch in range(1, args.nn_epochs + 1):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)
        clean_loss, clean_acc = net_test_with_logger(args, net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
        back_loss, back_acc = net_test_with_logger(args, net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
    
    return {
        'BA': clean_acc,
        'ASR': back_acc
    }


def run_badencoder(args, pretrained_encoder, shadow_dataset=None, memory_dataset=None, 
                  test_data_clean=None, test_data_backdoor=None,
                  downstream_train_dataset=None):
    """No docstring provided.
        No docstring provided.
    
    Args:
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
    

    log_path = os.path.join(args.output_dir, "badencoder_log.csv")
    log_exists = os.path.exists(log_path)
    

    csv_log = open(log_path, "a")
    if not log_exists:
        csv_log.write("epoch,loss,wasserstein_distance\n")
    # [WASSERSTEIN_LOG_END]
    

    train_loader = DataLoader(
        shadow_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    

    backdoored_model = copy.deepcopy(pretrained_encoder)
    

    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        optimizer = torch.optim.SGD(backdoored_model.f.parameters(), lr=args.lr, 
                                    weight_decay=args.weight_decay, momentum=args.momentum)
    else:  # 'imagenet' or 'CLIP'
        assert args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP', f"Unsupported encoder_usage_info: {args.encoder_usage_info}"

        # optimizer = torch.optim.SGD(backdoored_model.visual.parameters(), lr=args.lr, 
        #                            weight_decay=args.weight_decay, momentum=args.momentum)

        optimizer = torch.optim.Adam(backdoored_model.parameters(), lr=args.lr, 
                                   weight_decay=args.weight_decay)
    


    # if args.pretrained_encoder != '':

    #     if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
    #         checkpoint = torch.load(args.pretrained_encoder)
    #         pretrained_encoder.load_state_dict(checkpoint['state_dict'], strict=True)
    #         backdoored_model.load_state_dict(checkpoint['state_dict'], strict=True)
    #     elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
    #         checkpoint = torch.load(args.pretrained_encoder)
    #         pretrained_encoder.visual.load_state_dict(checkpoint['state_dict'], strict=True)
    #         backdoored_model.visual.load_state_dict(checkpoint['state_dict'], strict=True)
    #     else:

    

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    


    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    # )
    
    best_loss = float('inf')
    start_epoch = 0
    

    args.current_wasserstein_distance = 0.0
    # [WASSERSTEIN_LOG_END]
    

    for epoch in range(start_epoch, args.epochs):
        print("=================================================")

        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            train_loss = train_badencoder(
                backdoored_model.f, pretrained_encoder.f, train_loader, 
                optimizer, epoch, args, warm_up=(epoch < args.warm_up_epochs)
            )
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':

            # train_loss = train_badencoder(
            #     backdoored_model.visual, pretrained_encoder.visual, train_loader, 
            #     optimizer, epoch, args, warm_up=(epoch < args.warm_up_epochs)
            # )

            train_loss = train_badencoder(
                backdoored_model, pretrained_encoder, train_loader, 
                optimizer, epoch, args, warm_up=(epoch < args.warm_up_epochs)
            )
        else:
            raise NotImplementedError(f"Unsupported encoder_usage_info: {args.encoder_usage_info}")
        

        wasserstein_distance = args.current_wasserstein_distance
        print(f"Epoch {epoch}: Wasserstein Distance = {wasserstein_distance:.6f}")
        csv_log.write(f"{epoch},{train_loss:.6f},{wasserstein_distance:.6f}\n")
        csv_log.flush()
        # [WASSERSTEIN_LOG_END]
        


        # scheduler.step()
        

        if hasattr(args, 'logger_file'):
            args.logger_file.write(f"Epoch: [{epoch}/{args.epochs}] Training Loss: {train_loss:.4f} WD: {wasserstein_distance:.6f}\n")
            args.logger_file.flush()
        

        if train_loss < best_loss:
            best_loss = train_loss
            best_checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': backdoored_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss,
            }, best_checkpoint_path)
            print(f"Saved best checkpoint to:  {best_checkpoint_path}, Loss: {best_loss:.4f}")

            if hasattr(args, 'logger_file'):
                args.logger_file.write(f"Saved best checkpoint to:  {best_checkpoint_path}, Loss: {best_loss:.4f}\n")
                args.logger_file.flush()
        

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': backdoored_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to:  {checkpoint_path}")

            if hasattr(args, 'logger_file'):
                args.logger_file.write(f"Saved checkpoint to:  {checkpoint_path}\n")
                args.logger_file.flush()
        

        if (epoch + 1) % args.eval_freq == 0 and all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
            results = train_downstream_classifier(
                args, backdoored_model, downstream_train_dataset, 
                test_data_clean, test_data_backdoor
            )
            print(f"Downstream evaluation:  BA={results['BA']:.2f}%, ASR={results['ASR']:.2f}%")

            if hasattr(args, 'logger_file'):
                args.logger_file.write(f"Epoch: [{epoch}/{args.epochs}] Downstream evaluation:  BA={results['BA']:.2f}%, ASR={results['ASR']:.2f}%\n")
                args.logger_file.flush()
    

    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    backdoored_model.load_state_dict(checkpoint['state_dict'])
    

    if all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
        print("\n====== Final downstream evaluation ======")
        final_results = train_downstream_classifier(
            args, backdoored_model, downstream_train_dataset, 
            test_data_clean, test_data_backdoor
        )
        print(f"Final results: BA={final_results['BA']:.2f}%, ASR={final_results['ASR']:.2f}%")

        if hasattr(args, 'logger_file'):
            args.logger_file.write("\n====== Final downstream evaluation ======\n")
            args.logger_file.write(f"Final results: BA={final_results['BA']:.2f}%, ASR={final_results['ASR']:.2f}%\n")
            args.logger_file.flush()
    else:
        final_results = None
    

    csv_log.close()
    # [WASSERSTEIN_LOG_END]
    
    elapsed_time = time.time() - start_time
    print(f"BadEncoder training completed, elapsed time:  {elapsed_time:.2f} seconds")
    
    return backdoored_model, final_results 
