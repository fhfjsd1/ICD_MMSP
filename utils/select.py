'''针对ESC-50数据集的音频文件进行筛选和复制'''

import pandas as pd
import shutil
import os

# 读取CSV文件
csv_file_path = r'meta/esc50.csv'
df = pd.read_csv(csv_file_path)

# 提取category列数据为特定标签的所有行数据
category_label = 'airplane'
filtered_rows = df[df['category'] == category_label]

# 指定文件夹路径
source_folder = r'./audio'
destination_folder = f'data/{category_label}'

# 创建目标文件夹（如果不存在）
os.makedirs(destination_folder, exist_ok=True)

# 遍历所有满足条件的行数据
for index, row in filtered_rows.iterrows():
    # 获取该行数据的第一列数据（文件路径）
    file_path = row.iloc[0]

    # 构建完整的源文件路径
    source_file_path = os.path.join(source_folder, file_path)

    # 检查文件是否存在
    if os.path.exists(source_file_path):
        # 构建目标文件路径
        destination_file_path = os.path.join(destination_folder, os.path.basename(file_path))

        # 复制文件到新文件夹
        shutil.copy(source_file_path, destination_file_path)
        print(f"文件已复制到: {destination_file_path}")
    else:
        print(f"文件未找到: {source_file_path}")