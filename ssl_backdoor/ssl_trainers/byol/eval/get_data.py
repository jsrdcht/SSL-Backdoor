import torch
from tqdm import tqdm
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
