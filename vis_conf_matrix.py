import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取数据
file_path = '/workspace/sync/SSL-Backdoor/results/baseline/mocov2/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar/base_eval/conf_matrix_poisoned.npy'
data = np.load(file_path,allow_pickle=True)

# 使用 seaborn 绘制热图
plt.figure(figsize=(10, 10))
sns.heatmap(data, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Attention Map')
plt.xlabel('x-axis label')
plt.ylabel('y-axis label')

# 保存图片到同级目录
save_path = os.path.join(os.path.dirname(file_path), 'attention_map.png')
plt.savefig(save_path)
plt.close()
