"""PatchSearch utility implementation."""

import os
import re
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import pairwise_distances, roc_auc_score

from .utils import (
    get_model, get_feats, faiss_kmeans, KMeansLinear, get_candidate_patches,
    run_gradcam, extract_max_window, save_patches, paste_patch
)
from ssl_backdoor.utils.utils import set_seed



def setup_logger(save_dir):
    """PatchSearch utility implementation."""
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger('patchsearch')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(save_dir, 'patchsearch.log'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def patchsearch_iterative(
    model, 
    train_val_loader, 
    dataset_name,
    save_dir,
    arch='resnet18',
    num_clusters=100,
    test_images_size=1000,
    window_w=60,
    repeat_patch=1,
    samples_per_iteration=2,
    remove_per_iteration=0.25,
    batch_size=64,
    num_workers=8,
    prune_clusters=True,
    topk_thresholds=None
):
    """PatchSearch utility implementation."""
    if topk_thresholds is None:
        topk_thresholds = [5, 10, 20, 50, 100, 500]
    
    save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Starting PatchSearch iterative search")
    logger.info(f"Model architecture: {arch}")
    logger.info(f"Number of clusters: {num_clusters}")
    logger.info(f"Number of test images: {test_images_size}")
    logger.info(f"Window size: {window_w}")
    logger.info(f"Patch repeats per sample: {repeat_patch}")
    logger.info(f"Samples per iteration: {samples_per_iteration}")
    logger.info(f"Cluster prune ratio per iteration: {remove_per_iteration}")
    logger.info(f"Prune clusters: {prune_clusters}")
    cache_file_path = os.path.join(save_dir, 'cached_feats.pth')
    poison_scores_file = os.path.join(save_dir, 'poison-scores.npy')
    if os.path.exists(cache_file_path):
        logger.info(f"Loading cached features from {cache_file_path}")
        train_val_feats, train_val_labels, train_val_is_poisoned, train_val_inds = torch.load(cache_file_path)
    else:
        logger.info("Extracting features...")
        train_val_feats, train_val_labels, train_val_is_poisoned, train_val_inds = get_feats(model, train_val_loader)
        logger.info(f"Saving features to cache: {cache_file_path}")
        torch.save((train_val_feats, train_val_labels, train_val_is_poisoned, train_val_inds), cache_file_path)
    scores_cached = os.path.exists(poison_scores_file)
    logger.info(f"Clustering with {num_clusters}")
    logger.info("Running k-means clustering...")
    train_d, train_a, index, centroids = faiss_kmeans(train_val_feats, num_clusters)
    train_val_dataset = train_val_loader.dataset
    train_y = train_val_labels.numpy().reshape(-1, 1)
    train_i = train_val_inds.numpy().reshape(-1, 1)
    train_p = train_val_is_poisoned.numpy().reshape(-1, 1)
    model_with_kmeans = copy.deepcopy(model)
    model_with_kmeans.fc = KMeansLinear(train_a[:, 0], train_val_feats, num_clusters)
    model_with_kmeans = model_with_kmeans.to(device)
    logger.info("Building sample queue for each cluster")
    sorted_cluster_wise_i = []
    random_cluster_wise_i = []
    for cluster_id in range(num_clusters):
        cur_d = train_d[train_a == cluster_id]
        cur_i = train_i[train_a == cluster_id]
        sorted_cluster_wise_i.append(cur_i[np.argsort(cur_d)].tolist())
        random_cluster_wise_i.append(cur_i[np.random.permutation(len(cur_i))].tolist())
    logger.info(f"Fetching test images: {test_images_size} image(s)")
    test_images_i = []
    k = test_images_size // len(sorted_cluster_wise_i)
    if k > 0:
        for inds in sorted_cluster_wise_i:
            test_images_i.extend(inds[:k])
    else:
        for clust_i in np.random.permutation(len(sorted_cluster_wise_i))[:test_images_size]:
            test_images_i.append(sorted_cluster_wise_i[clust_i][0])
    
    test_images_dataset = Subset(train_val_dataset, torch.tensor(test_images_i))
    test_images_loader = DataLoader(
        test_images_dataset,
        shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True
    )
    
    logger.info("Loading test images")
    test_images = []
    for inp, _, _, _ in tqdm(test_images_loader):
        test_images.append(inp)
    test_images = torch.cat(test_images)
    test_images_a = train_a[test_images_i, 0]
    
    torch.cuda.empty_cache()
    c = model_with_kmeans.fc.classifier.detach().cpu()
    c = (c / c.norm(2, dim=1, keepdim=True)).numpy()
    cluster_distances = pairwise_distances(c, c)
    backbone = nn.DataParallel(model) if device.type == "cuda" else model
    backbone = backbone.eval()
    poison_scores = np.zeros(len(train_val_dataset))
    candidate_clusters = list(range(num_clusters))
    cur_iter = 0
    use_cached_poison_scores = os.path.exists(poison_scores_file)
    processed_count = 0
    
    if use_cached_poison_scores:
        logger.info(f"Detected cached poison score file: {poison_scores_file}, skipping search")
        poison_scores = np.load(poison_scores_file)
    else:
        while True:
            logger.info(f"Iteration: {cur_iter}")
            candidate_poison_i = []
            for clust_id in candidate_clusters:
                clust_i = random_cluster_wise_i[clust_id]
                for _ in range(min(len(clust_i), samples_per_iteration)):
                    candidate_poison_i.append(clust_i.pop(0))
            if not len(candidate_poison_i):
                logger.info("No more candidate images found, stopping iteration")
                break
            candidate_poison_dataset = Subset(
                train_val_dataset, torch.tensor(candidate_poison_i)
            )
            candidate_poison_loader = DataLoader(
                candidate_poison_dataset,
                shuffle=False, batch_size=batch_size,
                num_workers=num_workers, pin_memory=True
            )
            processed_count += len(candidate_poison_dataset)
            logger.info("Extracting patches")
            candidate_patches = get_candidate_patches(
                model_with_kmeans, candidate_poison_loader, arch, window_w, repeat_patch
            )
            logger.info("Evaluating patches")
            for candidate_patch, patch_idx in tqdm(zip(candidate_patches, candidate_poison_i)):
                cur_scores = []
                for cur_patch in candidate_patch:
                    with torch.no_grad():
                        poisoned_test_images = paste_patch(test_images.clone(), cur_patch)
                        feats_list = []
                        for i in range(0, poisoned_test_images.size(0), batch_size):
                            batch = poisoned_test_images[i:i + batch_size].to(device)
                            feats_list.append(backbone(batch).cpu())
                        feats_poisoned_test_images = torch.cat(feats_list).numpy()
                        _, poisoned_test_images_a = index.search(feats_poisoned_test_images, 1)
                        new = np.count_nonzero(poisoned_test_images_a == train_a[patch_idx, 0])
                        orig = np.count_nonzero(test_images_a == train_a[patch_idx, 0])
                        cur_scores.append(new - orig)
                assert poison_scores[patch_idx] == 0
                poison_scores[patch_idx] += max(cur_scores)
            logger.info(f"Maximum poison score {poison_scores.argmax()} : {poison_scores.max()}")
            cluster_scores = []
            for clust_id in candidate_clusters:
                cluster_scores.append((clust_id, poison_scores[train_a[:, 0] == clust_id].max()))
            cluster_scores = np.array(cluster_scores).astype(int)
            cluster_scores = cluster_scores[cluster_scores[:, 1].argsort()][::-1]
            for clust_rank, (clust_id, clust_score) in enumerate(cluster_scores.tolist()[:10]):
                logger.info(f"Top poison clusters: rank {clust_rank:3d} Cluster ID {clust_id:3d} score {clust_score}")
            
            logger.info(f"Processed count: {processed_count:6d}/{len(train_val_dataset)} ({processed_count*100/len(train_val_dataset):.1f}%)")
            
            if prune_clusters:
                rem = int(remove_per_iteration * len(candidate_clusters))
                candidate_clusters = cluster_scores[:len(cluster_scores)-rem, 0].tolist()
                
            cur_iter += 1
    if not use_cached_poison_scores:
        logger.info(f"Saving poison scores to: {poison_scores_file}")
        np.save(poison_scores_file, poison_scores)
    save_inds = poison_scores.argsort()[::-1][:100]
    
    inp, inp_titles = [], []
    for i in save_inds:
        inp.append(train_val_dataset[i][0])
        inp_titles.append(f'poison_score={poison_scores[i]:.1f}')
    inp = torch.stack(inp, dim=0)
    logger.info("Saving top poison images and patches")
    cam_images, out = run_gradcam(arch, model_with_kmeans, inp)
    windows = extract_max_window(cam_images, inp, window_w)
    patches_dir = os.path.join(save_dir, 'all_top_poison_patches')
    save_patches(windows, patches_dir, dataset_name)
    sorted_inds = poison_scores.argsort()[::-1]
    accs = [train_p[sorted_inds[:k]].sum() * 100.0 / k for k in topk_thresholds]
    
    logger.info('Top-k accuracy | ' + ' '.join(f'{k:7d}' for k in topk_thresholds))
    logger.info('Top-k accuracy | ' + ' '.join(f'{acc:7.1f}' for acc in accs))
    try:
        auroc = roc_auc_score(train_p[:, 0], poison_scores)
        logger.info(f'AUROC: {auroc*100:.2f}%')
    except Exception as e:
        logger.warning(f'Unable to compute AUROC: {e}')
    return poison_scores, sorted_inds, train_p[:, 0] 