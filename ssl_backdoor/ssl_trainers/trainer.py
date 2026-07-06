import os
import sys
import argparse
import random
import time
import socket
import warnings
import math
import builtins
from pathlib import Path
import importlib.util
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import ssl_backdoor.ssl_trainers.moco.loader
import ssl_backdoor.ssl_trainers.moco.builder
import ssl_backdoor.ssl_trainers.simsiam.builder
import ssl_backdoor.ssl_trainers.byol.builder
import ssl_backdoor.ssl_trainers.simclr.builder
import wandb
import ssl_backdoor.ssl_trainers.utils as utils
from ssl_backdoor.utils.utils import set_seed
import ssl_backdoor.datasets.dataset
from ssl_backdoor.datasets import dataset_params

from ssl_backdoor.ssl_trainers.utils import (
    initialize_distributed_training, 
    load_config_from_yaml, 
    merge_configs,
    # Logger, 
    load_config, 
    adjust_learning_rate
)


def get_trainer(config_or_path):
    """
        
    
    Args:
        
        
    Returns:
        
    """
    if isinstance(config_or_path, str):
        config = load_config(config_or_path)
    elif isinstance(config_or_path, argparse.Namespace):
        config = vars(config_or_path)
    elif isinstance(config_or_path, dict):
        config = config_or_path
    else:
        raise TypeError("config_or_path must be str, dict, or argparse.Namespace")
    args = argparse.Namespace(**config)
    args.save_folder_root = getattr(args, 'save_folder_root', 'checkpoints')
    args.experiment_id = getattr(args, 'experiment_id', f"{args.method}_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}")
    args.logger_type = getattr(args, 'logger_type', 'wandb')
    args.save_folder = os.path.join(args.save_folder_root, args.experiment_id)
    config['experiment_id'] = args.experiment_id
    config['logger_type'] = args.logger_type
    os.makedirs(args.save_folder, exist_ok=True)
    args.gpus = list(range(torch.cuda.device_count()))
    args.world_size = len(args.gpus)
    args.rank = 0
    args.distributed = getattr(args, 'multiprocessing_distributed', False)
    config['distributed'] = args.distributed
    try:
        with open(os.path.join(args.save_folder, 'final_config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"Warning: failed to save final config file. Error: {e}")
    def train_func():
        print("\nargs object passed to main_worker:")
        try:
            import pprint
            pprint.pprint(vars(args))
        except ImportError:
            print(args)
        print("-"*30)

        
        if args.distributed:
            mp.spawn(main_worker, nprocs=len(args.gpus), args=(args,))
        else:
            main_worker(0, args)
            
    return train_func



def main_worker(index, args):
    initialize_distributed_training(args, index)
    global logger
    if index == 0:
        args.enable_logging = args.logger_type.lower() != 'none'
        if args.enable_logging:
            if wandb.run is None:
                wandb.init(
                    project="ssl-backdoor",
                    name=args.experiment_id,
                    config=vars(args),
                    dir=args.save_folder
                )
            logger = wandb
        else:
            logger = None
    else:
        args.enable_logging = False
        logger = None
    scaler = GradScaler(enabled=args.amp) if hasattr(args, 'amp') and args.amp else None
    if args.multiprocessing_distributed and args.index != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    print(f"Using GPUs {args.gpus} to train on {socket.gethostname()}")
    if args.seed is not None:
        args.seed = args.seed + args.rank
        set_seed(args.seed)
    print(f"=> Creating model '{args.arch}'")
    if args.method == 'moco':
        model = ssl_backdoor.ssl_trainers.moco.builder.MoCo(
            models.__dict__[args.arch], args.feature_dim, args.moco_k, args.moco_m, 
            contr_tau=args.moco_contr_tau,
            align_alpha=args.moco_align_alpha, unif_t=args.moco_unif_t,
            dataset=args.dataset)
    elif args.method == 'simsiam':
        model = ssl_backdoor.ssl_trainers.simsiam.builder.SimSiam(
            models.__dict__[args.arch], dim=args.feature_dim, pred_dim=args.pred_dim, dataset=args.dataset)
    elif args.method == 'byol':
        model = ssl_backdoor.ssl_trainers.byol.builder.BYOL(
            models.__dict__[args.arch], 
            dim=args.feature_dim, 
            proj_dim=getattr(args, 'proj_dim', None),
            pred_dim=getattr(args, 'pred_dim', None),
            tau=getattr(args, 'byol_tau', 0.99),
            dataset=args.dataset)
    elif args.method == 'simclr':
        model = ssl_backdoor.ssl_trainers.simclr.builder.SimCLR(
            models.__dict__[args.arch], 
            dim=args.feature_dim, 
            proj_dim=getattr(args, 'proj_dim', 128),
            dataset=args.dataset)
    else:
        raise ValueError(f"Unknown method '{args.method}'")

    model.cuda(args.gpu)
    if args.distributed:
        if args.method in ['simsiam', 'byol']:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            broadcast_buffers=False,
            find_unused_parameters=True
        )
    if index == 0 and logger is not None:
        logger.watch(model, log_freq=args.print_freq)
    if hasattr(args, 'fix_pred_lr') and args.fix_pred_lr: # only simsiam needs this
        if args.method == 'simsiam':
            model_ref = model.module if hasattr(model, 'module') else model
            optim_params = [
                {'params': model_ref.encoder.parameters(), 'fix_lr': False},
                {'params': model_ref.projector.parameters(), 'fix_lr': False},
                {'params': model_ref.predictor.parameters(), 'fix_lr': True}
            ]
        else:
            optim_params = model.parameters()
    else:
        optim_params = model.parameters()
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            optim_params, args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: '{args.optimizer}'")
    if hasattr(args, 'lr_schedule'):
        has_fixed_lr_groups = any('fix_lr' in pg and pg['fix_lr'] for pg in optimizer.param_groups)
        print(f"has_fixed_lr_groups: {has_fixed_lr_groups}")
        
        if args.lr_schedule.lower() == 'cos':
            if has_fixed_lr_groups:
                lr_lambdas = []
                for param_group in optimizer.param_groups:
                    if 'fix_lr' in param_group and param_group['fix_lr']:
                        lr_lambdas.append(lambda _: 1.0)
                    else:
                        lr_lambdas.append(lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)))
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
                print(f"=> Using cosine annealing LR schedule with fixed-lr support (T_max={args.epochs})")
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=0
                )
                print(f"=> Using cosine annealing LR schedule (T_max={args.epochs})")
                
        elif args.lr_schedule.lower() == 'step':
            milestones = args.lr_drops if hasattr(args, 'lr_drops') else [args.epochs // 2]
            gamma = args.lr_drop_gamma if hasattr(args, 'lr_drop_gamma') else 0.1
            
            if has_fixed_lr_groups:
                raise ValueError("Step LR schedule does not support fixed-lr parameter groups (fix_lr)")
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, gamma=gamma
                )
                print(f"=> Using step LR schedule (milestones={milestones}, gamma={gamma})")
        else:
            print(f"Warning: unknown LR schedule '{args.lr_schedule}', using constant learning rate")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        print("Warning: LR schedule is not set, using constant learning rate")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device('cuda', args.gpu))
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            model_has_module = hasattr(model, 'module')
            state_has_module = any(k.startswith('module.') for k in state_dict.keys())

            if model_has_module and not state_has_module:
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}
            elif state_has_module and not model_has_module:
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> Restored LR scheduler state")
            if 'scaler' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
                print("=> Restored GradScaler state")
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> Checkpoint not found: '{args.resume}'")
    train_loader = create_data_loader(args)
    do_eval = hasattr(args, 'test_config') and isinstance(args.test_config, dict)

    
    if do_eval and args.rank == 0:
        print(f"Model evaluation enabled, frequency: every {args.eval_frequency} epochs")
    best_results = {
        'clean_acc': 0.0,
        'poison_acc': 0.0,
        'asr': 0.0,
        'epoch': 0
    }
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train(train_loader, model, optimizer, epoch, args, scaler)
        scheduler.step()
        if args.rank == 0 and logger is not None:
            current_lr = optimizer.param_groups[0]['lr']
            global_step_end = (epoch + 1) * len(train_loader) - 1
            logger.log({'train/learning_rate': current_lr}, step=global_step_end)
            print(f"Epoch {epoch} learning rate: {current_lr:.8f}")
        should_save = (epoch + 1) % args.save_freq == 0
        should_eval = do_eval and (epoch + 1) % args.eval_frequency == 0
        save_filename = None
        if (args.distributed and args.rank == 0) or (args.index == 0):
            if should_save:
                save_dir = args.save_folder
                save_filename = os.path.join(save_dir, f'checkpoint_{epoch:04d}.pth.tar')
                save_dict = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                if scaler is not None:
                    save_dict['scaler'] = scaler.state_dict()
                torch.save(save_dict, save_filename)
                print(f"Saved checkpoint to '{save_filename}'")
        if args.distributed:
            try:
                dist.barrier()
            except Exception as e:
                print(f"[WARNING][rank {args.rank}] Failed to sync checkpoint save: {e}")
        if should_eval:
            from tools.test import test_model
            # ---------------------
            eval_step = (epoch + 1) * len(train_loader) - 1
            setattr(args, 'current_global_step', eval_step)
            print(f"rank {args.rank} evaluating model at epoch {epoch+1}...")
            eval_backbone = None
            model_for_eval = model.module if hasattr(model, 'module') else model
            if hasattr(model_for_eval, 'encoder'):
                eval_backbone = model_for_eval.encoder
            setattr(args, 'eval_backbone', eval_backbone)

            eval_config = dict(getattr(args, 'test_config', {}) or {})
            if 'distributed' not in eval_config:
                eval_config['distributed'] = bool(getattr(args, 'distributed', False))

            clean_acc, poison_acc, asr = test_model(
                save_filename,
                epoch + 1,
                args,
                logger,
                config=eval_config
            )
            if (args.distributed and args.rank == 0) or (args.index == 0):
                if should_eval and not should_save and save_filename and os.path.exists(save_filename):
                    try:
                        os.remove(save_filename)
                        print(f"Deleted temporary evaluation checkpoint: {save_filename}")
                        tmp_dir = os.path.dirname(save_filename)
                        if os.path.exists(tmp_dir) and not os.listdir(tmp_dir):
                            os.rmdir(tmp_dir)
                            print(f"Deleted empty temporary directory: {tmp_dir}")
                    except Exception as e:
                        print(f"Error while deleting temporary evaluation checkpoint: {e}")
            if args.rank == 0:
                if logger is not None:
                    global_step_end = (epoch + 1) * len(train_loader) - 1
                    log_message = f"Evaluation result - Epoch {epoch+1} | Clean Acc: {clean_acc:.2f}% | Poison Acc: {poison_acc:.2f}% | Attack Success Rate: {asr:.2f}%"
                    print(log_message)

                    logger.log({
                        "eval/epoch": epoch + 1,
                        "eval/clean_acc": clean_acc,
                        "eval/poison_acc": poison_acc,
                        "eval/attack_success_rate": asr,
                        "eval/summary": log_message
                    }, step=global_step_end)
                if clean_acc > best_results['clean_acc']:
                    best_results['clean_acc'] = clean_acc
                    best_results['poison_acc'] = poison_acc
                    best_results['asr'] = asr
                    best_results['epoch'] = epoch + 1
                print(f"Current eval: Clean Acc: {clean_acc:.2f}%, Poison Acc: {poison_acc:.2f}%, ASR: {asr:.2f}%")
                print(f"Best eval: Clean Acc: {best_results['clean_acc']:.2f}%, "
                      f"Poison Acc: {best_results['poison_acc']:.2f}%, "
                      f"ASR: {best_results['asr']:.2f}% (Epoch {best_results['epoch']})")
                with open(os.path.join(args.save_folder, 'eval_summary.txt'), 'w') as f:
                    f.write(f"Experiment ID: {args.experiment_id}\n")
                    f.write(f"Best clean accuracy: {best_results['clean_acc']:.2f}% (Epoch {best_results['epoch']})\n")
                    f.write(f"Corresponding poison accuracy: {best_results['poison_acc']:.2f}%\n")
                    f.write(f"Corresponding attack success rate: {best_results['asr']:.2f}%\n")
                    f.write(f"Last evaluation (Epoch {epoch+1}):\n")
                    f.write(f"  Clean Accuracy: {clean_acc:.2f}%\n")
                    f.write(f"  Poison Accuracy: {poison_acc:.2f}%\n")
                    f.write(f"  Attack Success Rate: {asr:.2f}%\n")
        if should_eval and args.distributed:
            if args.rank == 0:
                print(f"rank 0 evaluation finished, waiting for other processes to synchronize (Barrier)...")
            try:
                dist.barrier()
            except Exception as e:
                print(f"[WARNING][rank {args.rank}] Post-evaluation synchronization failed: {e}")

            if args.rank == 0:
                print(f"All process synchronization complete; continue training.")
    if args.index == 0 and logger is not None:
        if hasattr(logger, 'finish'):
            logger.finish()
        elif hasattr(logger, 'close'):
            logger.close()


def create_data_loader(args):
    """
        
    
    Args:
        
        
    Returns:
        
    """
    if args.dataset not in dataset_params:
        raise ValueError(f"Unsupported dataset: '{args.dataset}'")
    params = dataset_params[args.dataset]
    args.image_size = params['image_size']
    min_scale = getattr(args, 'min_crop_scale', 0.2)
    print(f"RandomResizedCrop minimum scale: {min_scale}")
    augmentation = [
        transforms.RandomResizedCrop(args.image_size, scale=(min_scale, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([ssl_backdoor.ssl_trainers.moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        params['normalize'],
    ]
    composed_transforms = ssl_backdoor.ssl_trainers.moco.loader.TwoCropsTransform(
        transforms.Compose(augmentation)
    )
    dataset_classes = {
        'corruptencoder': ssl_backdoor.datasets.dataset.CorruptEncoderTrainDataset,
        'sslbkd': ssl_backdoor.datasets.dataset.SSLBackdoorTrainDataset,
        'ctrl': ssl_backdoor.datasets.dataset.CTRLTrainDataset,
        'clean': ssl_backdoor.datasets.dataset.FileListDataset,
        'blto': ssl_backdoor.datasets.dataset.BltoPoisoningPoisonedTrainDataset,
        'external_backdoor': ssl_backdoor.datasets.dataset.ExternalBackdoorTrainDataset,
    }

    if args.attack_algorithm not in dataset_classes:
        raise ValueError(f"Unsupported attack algorithm: '{args.attack_algorithm}'")
    train_dataset = dataset_classes[args.attack_algorithm](args, args.data, composed_transforms)
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )


def train(train_loader, model, optimizer, epoch, args, scaler):
    batch_time = utils.AverageMeter('Time', '6.3f')
    data_time = utils.AverageMeter('Data', '6.3f')
    if args.dataset not in dataset_params:
        raise ValueError(f"Unsupported dataset: '{args.dataset}'")
    normalize = dataset_params[args.dataset]['normalize']
    mean = normalize.mean
    std = normalize.std
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)], 
        std=[1/s for s in std]
    )
    inv_transform = transforms.Compose([inv_normalize, transforms.ToPILImage()])
    img_save_dir = os.path.join(args.save_folder, "train_images")
    os.makedirs(img_save_dir, exist_ok=True)
    img_ctr = 0

    contr_meter = utils.AverageMeter('Contr-Loss', '.4e')
    if args.method == 'moco':
        acc1 = utils.AverageMeter('Contr-Acc1', '6.2f')
        acc5 = utils.AverageMeter('Contr-Acc5', '6.2f')
        loss_meters = [contr_meter, acc1, acc5, utils.ProgressMeter.BR]
    elif args.method in ['simsiam', 'byol']:
        loss_meters = [contr_meter, utils.ProgressMeter.BR]
    else:
        loss_meters = [contr_meter]
    if loss_meters and loss_meters[-1] == utils.ProgressMeter.BR:
        loss_meters = loss_meters[:-1]

    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time] + loss_meters,
        prefix=f"Epoch: [{epoch}]"
    )
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)
        if len(images) > 2:
            images[2] = images[2].cuda(args.gpu, non_blocking=True)
        if epoch == 0 and i < 20:
            debug_target = getattr(args, 'debug_target', None)
            if debug_target is not None:
                debug_target = int(debug_target)
            elif hasattr(args, 'attack_target_list'):
                debug_target = args.attack_target_list[0]
            for batch_index in range(images[0].size(0)):
                if debug_target is not None and int(target[batch_index].item()) == debug_target:
                    img_ctr += 1
                    for view_idx, view in enumerate(images[:2]):
                        inv_image = inv_transform(view[batch_index].cpu())
                        save_path = os.path.join(img_save_dir, f"{img_ctr:05d}_view_{view_idx}.png")
                        inv_image.save(save_path)
        if args.amp:
            with torch.amp.autocast('cuda'):
                loss = model(images[0], images[1])
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss = model(images[0], images[1])
            loss.backward()
            optimizer.step()
        if args.method in ['byol'] and hasattr(model, 'module'):
            model.module.update_target(float(epoch) / args.epochs)
        elif args.method in ['byol']:
            model.update_target(float(epoch) / args.epochs)
        if args.index == 0:
            bs = images[0].shape[0]
            contr_meter.update(loss.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 and args.index == 0:
            progress.display(i)
            if args.enable_logging and logger is not None:
                current_step = epoch * len(train_loader) + i
                logger.log({
                    'train/batch_ssl_loss': loss.item(),
                    'train/epoch_avg_ssl_loss': contr_meter.avg,
                }, step=current_step)
    if args.index == 0 and args.enable_logging and logger is not None:
        global_step_end = (epoch + 1) * len(train_loader) - 1
        logger.log({
            "train/epoch": epoch,
            "train/ssl_loss": contr_meter.avg
        }, step=global_step_end)
            

