"""PatchSearch utility implementation."""

import os
import math
import random
import logging
import numpy as np
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, ResNet
from PIL import Image
from tqdm import tqdm

from .utils.dataset import get_transforms
from .utils.evaluation import AverageMeter, ProgressMeter, accuracy
from .utils.visualization import denormalize, show_images_grid
from ssl_backdoor.utils.utils import set_seed


class PoisonDataset(Dataset):
    """PatchSearch utility implementation."""
    def __init__(self, args, path_to_txt_file, pre_transform, post_transform, poison_dir, topk_poisons, output_type='clean'):
        self.output_type = output_type
        self.args = args
        self.poisons = []
        for i in range(topk_poisons):
            poison_file = os.path.join(poison_dir, f'{i:05d}.png')
            if os.path.exists(poison_file):
                self.poisons.append(Image.open(poison_file).convert('RGB'))
        
        self.poisons = self.poisons[:topk_poisons]
        if len(self.poisons) == 0:
            raise ValueError(f"No poison patches found in {poison_dir}")

        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def paste_poison(self, img):
        """PatchSearch utility implementation."""
        if 'imagenet' in self.args.dataset_name:
            margin = 10
            image_size = 224
            poison_size_low, poison_size_high = 20, 80
        elif 'cifar' in self.args.dataset_name:
            margin = 2
            image_size = 32
            poison_size_low, poison_size_high = 4, 16
        elif 'stl' in self.args.dataset_name:
            margin = 5
            image_size = 96
            poison_size_low, poison_size_high = 12, 40
        else:
            raise ValueError(f'Unexpected dataset: {self.args.dataset_name}')
        poison = self.poisons[np.random.randint(low=0, high=len(self.poisons))]
        new_s = np.random.randint(low=poison_size_low, high=poison_size_high)
        poison = poison.resize((new_s, new_s))
        loc_box = (margin, image_size - (new_s + margin))
        loc_h, loc_w = np.random.randint(*loc_box), np.random.randint(*loc_box)
        img.paste(poison, (loc_h, loc_w))
        return img

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        is_poisoned = 'poison' in image_path
        img = Image.open(image_path).convert('RGB')
        is_poison = np.random.rand() > 0.5

        if self.output_type == 'clean' or (self.output_type == 'rand' and not is_poison):
            target = 0
            img = self.pre_transform(img)
            img = self.post_transform(img)
        elif self.output_type == 'poisoned' or (self.output_type == 'rand' and is_poison):
            target = 1
            img = self.pre_transform(img)
            img = self.paste_poison(img)
            img = self.post_transform(img)
        else:
            raise ValueError(f'Unexpected output_type: {self.output_type}')

        return image_path, img, target, is_poisoned, idx

    def __len__(self):
        return len(self.file_list)


class ValPoisonDataset(Dataset):
    """PatchSearch utility implementation."""
    def __init__(self, path_to_txt_file, pos_inds, neg_inds, transform):
        with open(path_to_txt_file, 'r') as f:
            file_list = f.readlines()
            file_list = [row.strip().split() for row in file_list]

        pos_samples = [(file_list[i][0], 1) for i in pos_inds]
        neg_samples = [(file_list[i][0], 0) for i in neg_inds]
        self.samples = pos_samples + neg_samples
        self.transform = transform

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        is_poisoned = 'poison' in image_path
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        return image_path, img, target, is_poisoned, idx

    def __len__(self):
        return len(self.samples)


class EnsembleNet(nn.Module):
    """PatchSearch utility implementation."""
    def __init__(self, models):
        super(EnsembleNet, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        y = torch.stack([model(x) for model in self.models], dim=0)
        y = torch.einsum('kbd->bkd', y)
        y = y.mean(dim=1)
        return y


def worker_init_fn(baseline_seed, it, worker_id):
    """PatchSearch utility implementation."""
    np.random.seed(baseline_seed + it + worker_id)


def prepare_datasets(args, poison_scores):
    """PatchSearch utility implementation."""
    logger = logging.getLogger('patchsearch')
    from ssl_backdoor.datasets import dataset_params

    if args.dataset_name not in dataset_params:
        raise ValueError(f"Unknown dataset '{args.dataset_name}'")
    normalize = dataset_params[args.dataset_name]['normalize']
    args.image_size = dataset_params[args.dataset_name]['image_size']
    train_t1 = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
    ])
    train_t2 = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = PoisonDataset(
        args,
        args.train_file,
        pre_transform=train_t1, 
        post_transform=train_t2,
        poison_dir=args.poison_dir,
        topk_poisons=args.topk_poisons,
        output_type='rand',
    )
    inds = np.random.randint(low=0, high=len(train_dataset), size=40)
    train_dataset.output_type = 'clean'
    clean_images = torch.stack([train_dataset[i][1] for i in inds])
    show_images_grid(clean_images, args.output_dir, 'Train_Clean_Images', args)
    train_dataset.output_type = 'poisoned'
    poisoned_images = torch.stack([train_dataset[i][1] for i in inds])
    show_images_grid(poisoned_images, args.output_dir, 'Train_Poisoned_Images', args)
    train_dataset.output_type = 'rand'
    rand_images = torch.stack([train_dataset[i][1] for i in inds])
    show_images_grid(rand_images, args.output_dir, 'Train_Rand_Images', args)
    sorted_inds = (-poison_scores).argsort()
    pos_inds = sorted_inds[:args.topk_poisons]
    neg_inds = sorted_inds[-args.topk_poisons:]
    train_inds = sorted_inds[int(args.top_p*len(train_dataset)):-args.topk_poisons]
    logger.info(f'Training dataset size: {len(train_inds)/1000:.1f}k')
    train_dataset.output_type = 'rand'
    train_dataset = Subset(train_dataset, train_inds)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True,
        worker_init_fn=partial(worker_init_fn, args.seed, 0)
    )
    if 'imagenet' in args.dataset_name:
        val_t1 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
    elif 'cifar' in args.dataset_name:
        val_t1 = transforms.Compose([
            transforms.Resize(32),
        ])
    elif 'stl' in args.dataset_name:
        val_t1 = transforms.Compose([
            transforms.Resize(96),
        ])
    else:
        raise ValueError(f'Unexpected dataset: {args.dataset_name}')
        
    val_t2 = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    val_dataset = ValPoisonDataset(
        args.train_file,
        pos_inds=pos_inds,
        neg_inds=neg_inds,
        transform=transforms.Compose([val_t1, val_t2]),
    )
    val_images = torch.stack([val_dataset[i][1] for i in range(min(40, len(val_dataset)))])
    show_images_grid(val_images, args.output_dir, 'Val_Images', args)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
        worker_init_fn=partial(worker_init_fn, args.seed, 0)
    )
    test_dataset = PoisonDataset(
        args,
        args.train_file,
        pre_transform=val_t1, 
        post_transform=val_t2,
        poison_dir=args.poison_dir,
        topk_poisons=args.topk_poisons,
        output_type='clean'
    )
    inds = np.random.randint(low=0, high=len(test_dataset), size=min(40, len(test_dataset)))
    test_images = torch.stack([test_dataset[i][1] for i in inds])
    show_images_grid(test_images, args.output_dir, 'Test_Images', args)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def validate(val_loader, model, args):
    """PatchSearch utility implementation."""
    logger = logging.getLogger('patchsearch')
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time],
        prefix="Validation"
    )
    model.eval()
    device = next(model.parameters()).device

    pred_is_poison = np.zeros(len(val_loader.dataset))
    gt_is_poison = np.zeros(len(val_loader.dataset))

    end = time.time()
    for i, (_, images, target, _, inds) in enumerate(val_loader):
        if i == 0:
            show_images_grid(images, args.output_dir, f'eval-images-iteration-0', args)
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            output = model(images)

        pred = output.argmax(dim=1).detach().cpu()
        pred_is_poison[inds.numpy()] = pred.numpy()
        gt_is_poison[inds.numpy()] = target.numpy().astype(int)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))
    if np.sum(gt_is_poison) > 0:
        recall = pred_is_poison[np.where(gt_is_poison)[0]].astype(float).mean()
    else:
        recall = 0.0
    logger.info(f'Poison recall: {recall*100:.1f}%')
    if np.sum(pred_is_poison) > 0:
        precision = gt_is_poison[np.where(pred_is_poison)[0]].astype(float).mean()
    else:
        precision = 0.0
    logger.info(f'Poison precision: {precision*100:.1f}%')
    beta = 1
    if precision > 0 or recall > 0:
        f1_beta = (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall + 1e-10)
    else:
        f1_beta = 0.0
    logger.info(f'Poison F1_beta score (beta = {beta}): {f1_beta*100:.1f}%')

    if math.isnan(recall) or math.isnan(precision) or math.isnan(f1_beta):
        return 0., 0., 0.

    return recall, precision, f1_beta


def test(test_loader, model, args):
    """PatchSearch utility implementation."""
    logger = logging.getLogger('patchsearch')
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time],
        prefix="Test"
    )
    model.eval()
    device = next(model.parameters()).device

    pred_is_poison = np.zeros(len(test_loader.dataset))
    prob_is_poison = np.zeros(len(test_loader.dataset))
    gt_is_poison = np.zeros(len(test_loader.dataset))

    end = time.time()
    for i, (_, images, _, is_poisoned, inds) in enumerate(test_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            output = model(images)
            probs = F.softmax(output, dim=1)

        pred = output.argmax(dim=1).detach().cpu()
        pred_is_poison[inds.numpy()] = pred.numpy()
        prob_is_poison[inds.numpy()] = probs[:, 1].detach().cpu().numpy()
        gt_is_poison[inds.numpy()] = is_poisoned.numpy().astype(int)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % max(1, len(test_loader) // 20) == 0:
            logger.info(progress.display(i))

    logger.info(f'Total poisons to remove: {np.count_nonzero(pred_is_poison)}')
    
    # Calculate metrics
    if len(np.unique(gt_is_poison)) > 1:
        tn, fp, fn, tp = confusion_matrix(gt_is_poison, pred_is_poison, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        try:
            auroc = roc_auc_score(gt_is_poison, prob_is_poison)
            auprc = average_precision_score(gt_is_poison, prob_is_poison)
        except ValueError:
            auroc = 0.0
            auprc = 0.0
    else:
        # If only one class is present in ground truth
        tpr = 0.0
        fpr = 0.0
        precision = 0.0
        auroc = 0.0
        auprc = 0.0
        logger.warning("Ground truth only contains one class. Metrics might be invalid.")
        
    logger.info(f'Detection metrics:')
    logger.info(f'TPR (Recall): {tpr*100:.2f}%')
    logger.info(f'FPR: {fpr*100:.2f}%')
    logger.info(f'Precision: {precision*100:.2f}%')
    logger.info(f'AUROC: {auroc*100:.2f}%')
    logger.info(f'AUPRC (Average Precision): {auprc*100:.2f}%')

    return tpr, precision, pred_is_poison


def train(args, poison_scores, external_test_loader=None):
    """PatchSearch utility implementation."""
    if args.seed is not None:
        set_seed(args.seed)
    logger = logging.getLogger('patchsearch')
    logger.info(f"Starting poison classifier training")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    train_loader, val_loader, test_loader = prepare_datasets(args, poison_scores)
    models = []
    for model_i in range(args.model_count):
        logger.info('='*40 + f' Model {model_i} ' + '='*40)
        train_loader.worker_init_fn = partial(worker_init_fn, args.seed, model_i)
        val_loader.worker_init_fn = partial(worker_init_fn, args.seed, model_i)
        model = ResNet(block=BasicBlock, layers=[1, 1, 1, 1])
        model.fc = nn.Linear(512, 2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            args.max_iterations
        )
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        lrs = AverageMeter('LR', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            args.max_iterations,
            [batch_time, data_time, lrs, losses, top1],
            prefix="Train: "
        )
        model.train()
        it = 0
        val_metrics = []
        while it < args.max_iterations:
            end = time.time()
            for _, images, target, is_poisoned, inds in train_loader:
                if it >= args.max_iterations:
                    break
                if it < 5:
                    show_images_grid(images, args.output_dir, f'train-images-iteration-{it:05d}', args)
                data_time.update(time.time() - end)
                
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(images)
                loss = F.cross_entropy(output, target)
                losses.update(loss.item(), images.size(0))
                acc1 = accuracy(output, target, topk=(1,))[0]
                top1.update(acc1[0], images.size(0))
                lrs.update(lr_scheduler.get_last_lr()[-1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if it % args.print_freq == 0:
                    logger.info(progress.display(it))
                if it % args.eval_freq == 0:
                    recall, precision, f1_beta = validate(val_loader, model, args)
                    val_metrics.append((recall, precision, f1_beta))
                    vm = set([int(x[2]*100) for x in val_metrics[-10:]])
                    logger.info(f"Average F1 over last 10 validations: {vm}")
                    if len(vm) == 1 and len(val_metrics) > 10:
                        logger.info("F1 score stabilized; early stopping")
                        it = args.max_iterations
                        break
                    model.train()
                lr_scheduler.step()
                
                it += 1
        models.append(model)
    ensemble_model = EnsembleNet(models)
    logger.info(f'Running inference on test data')
    recall, precision, preds = test(test_loader, ensemble_model, args)

    if external_test_loader is not None:
        logger.info(f'Running inference on external test data (Balanced Clean + Poisoned)')
        ext_recall, ext_precision, ext_preds = test(external_test_loader, ensemble_model, args)
        
        # Calculate extra metrics if needed, test() already prints TPR/FPR/AUROC
        # But we might want to ensure they are logged clearly as "External Test Metrics"
        logger.info(f'External test set evaluation complete')
    filtered_file_path = os.path.join(args.output_dir, 'filtered.txt')
    with open(filtered_file_path, 'w') as f:
        for line, is_poisoned in zip(test_loader.dataset.file_list, preds):
            if not is_poisoned:
                f.write(f'{line}\n')
    
    logger.info(f"Filtered dataset saved to: {filtered_file_path}")
    
    return filtered_file_path


def run_poison_classifier(
    poison_scores,
    output_dir,
    train_file,
    poison_dir,
    dataset_name='imagenet100',
    topk_poisons=20,
    top_p=0.10,
    model_count=3,
    max_iterations=2000,
    batch_size=128,
    num_workers=8,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    print_freq=10,
    eval_freq=50,
    seed=42,
    external_test_loader=None
):
    """PatchSearch utility implementation."""
    class Args:
        pass
    
    args = Args()
    args.output_dir = output_dir
    args.train_file = train_file
    args.poison_dir = poison_dir
    args.dataset_name = dataset_name
    args.topk_poisons = topk_poisons
    args.top_p = top_p
    args.model_count = model_count
    args.max_iterations = max_iterations
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.lr = lr
    args.momentum = momentum
    args.weight_decay = weight_decay
    args.print_freq = print_freq
    args.eval_freq = eval_freq
    args.seed = seed
    dir_name = f'poison_classifier_topk_{topk_poisons}_ensemble_{model_count}_max_iter_{max_iterations}'
    output_dir = os.path.join(output_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    logger = logging.getLogger('patchsearch')
    filtered_file_path = train(args, poison_scores, external_test_loader)
    
    return filtered_file_path

import time
