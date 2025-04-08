
import numpy as np
import yaml
import argparse
import os
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn

from cfg import get_cfg
from dataset import get_ds
from methods import get_method
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

def get_scheduler(optimizer, cfg):
    if cfg.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epoch if cfg.T0 is None else cfg.T0,
            T_mult=cfg.Tmult,
            eta_min=cfg.eta_min,
        )
    elif cfg.lr_step == "step":
        m = [cfg.epoch - a for a in cfg.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=cfg.drop_gamma)
    else:
        return None

def load_config_from_yaml(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def merge_configs(defaults, overrides):
    """Merge two dictionaries, prioritizing values from 'overrides'."""
    result = defaults.copy()

    result.update({k: v for k, v in overrides.items() if not k in result.keys()})
    result.update({k: v for k, v in overrides.items() if v is not None})

    return argparse.Namespace(**result)
    
if __name__ == "__main__":
    cfg = get_cfg()
    print('cfg', cfg)

    if cfg.config:
        config_from_yaml = load_config_from_yaml(cfg.config)
    else:
        config_from_yaml = {}

    # Prepare final configuration by merging YAML config with command line arguments
    cfg = merge_configs(config_from_yaml, vars(cfg))
    print(cfg)

    if 'targeted' in cfg.attack_algorithm:
        assert cfg.downstream_data is not None

    # wandb.init(project=cfg.wandb, config=cfg)
    writer = SummaryWriter(cfg.save_folder)

    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers, bs_clf=256, bs_test=256)
    train_dataloader = ds.train

    model = get_method(cfg.method)(cfg)
    model.cuda().train()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2)
    scheduler = get_scheduler(optimizer, cfg)
    starting_epoch = 0
    eval_every = cfg.eval_every
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True

    if cfg.fname is not None:
        checkpoint = torch.load(cfg.fname)

        starting_epoch = checkpoint['epoch']  # 加载当前 epoch
        lr_warmup = checkpoint['warmup']  # 加载当前 warmup
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    

    for ep in trange(starting_epoch, cfg.epoch, position=0):
        loss_ep = []
        iters = len(train_dataloader)
        for n_iter, (samples, _) in enumerate(tqdm(train_dataloader, position=1)):
            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1

            optimizer.zero_grad()
            loss = model(samples)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            model.step(ep / cfg.epoch)
            if cfg.lr_step == "cos" and lr_warmup >= 500:
                scheduler.step(ep + n_iter / iters)

        if cfg.lr_step == "step":
            scheduler.step()


        # if (ep + 1) % eval_every == 0:
        #     acc_knn, acc_linear, asr_knn, asr_linear = model.get_acc(ds.clf, ds.test, ds.test_p)
        #     # wandb.log({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn}, commit=False)
        #     writer.add_scalar('Accuracy/acc', acc_linear[1], ep)
        #     writer.add_scalar('Accuracy/acc_5', acc_linear[5], ep)
        #     writer.add_scalar('Accuracy/acc_knn', acc_knn, ep)
        #     writer.add_scalar('Accuracy/asr', asr_linear[1], ep)
        #     writer.add_scalar('Accuracy/asr_5', asr_linear[5], ep)
        #     writer.add_scalar('Accuracy/asr_knn', asr_knn, ep)

        if (ep + 1) % cfg.save_freq == 0:
            fname = f"{cfg.save_folder}/{ep}.pth"
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            checkpoint = {
                'epoch': ep,
                'warmup': lr_warmup,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            }
            torch.save(checkpoint, fname)

        # wandb.log({"loss": np.mean(loss_ep), "ep": ep})
        writer.add_scalar('Loss/byol_pretrain', np.mean(loss_ep), ep)
