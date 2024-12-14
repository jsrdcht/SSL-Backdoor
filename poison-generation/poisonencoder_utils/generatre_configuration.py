import os
import re

data_dir = '/workspace/sync/SSL-Backdoor/poison-generation/poisonencoder_utils/references'  # 替换为你的数据集路径
config_file = 'data_config.txt'    # 配置文件名

with open(config_file, 'w') as f:
    category_number = 0
    for subdir in sorted(os.listdir(data_dir)):  # 对子目录进行排序
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            for subsubdir in sorted(os.listdir(subdir_path)):  # 对二级子目录进行排序
                subsubdir_path = os.path.join(subdir_path, subsubdir)
                if os.path.isdir(subsubdir_path) and re.search(r'\d', subsubdir):  # 检查名字里是否有数字
                    f.write(f'{subsubdir_path}/img.png {category_number}\n')
            category_number += 1