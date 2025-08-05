import os
import csv
import random
from collections import defaultdict

def generate_csv(directory, output_csv, num_folds=5):
    category_files = defaultdict(list)
    
    # 遍历目录及其子目录，收集所有类别的音频文件
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file)
                if "_" in file_name:
                    category = file_name.split("_")[0]
                    category_files[category].append(file_name)
    
    # 对每个类别的文件进行随机打乱并分成5个fold
    category_folds = defaultdict(list)
    for category, files in category_files.items():
        random.shuffle(files)
        fold_size = len(files) // num_folds
        for i in range(num_folds):
            start_index = i * fold_size
            end_index = start_index + fold_size if i < num_folds - 1 else len(files)
            category_folds[category].append(files[start_index:end_index])
    
    # 确保每个fold中两个类别的数量比例相同
    balanced_folds = [[] for _ in range(num_folds)]
    for category, folds in category_folds.items():
        for i, fold in enumerate(folds):
            balanced_folds[i].extend(fold)
    
    # 打开CSV文件用于写入
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # 写入表头
        csv_writer.writerow(['wav', 'class_string', 'ID', 'Name', 'subclass','fold'])
        
        # 遍历每个fold，写入CSV文件
        for fold_index, fold_files in enumerate(balanced_folds):
            for file_name in fold_files:
                if "_" in file_name:
                    _, number_with_extension = file_name.split("_", 1)
                   # category = root.split("/")[1]
                    number = number_with_extension.replace(".wav", "")
                    name = file_name.split(".")[0]
                    subclass = name.split("_")[0]
                    csv_writer.writerow([file_name, "negative", number, name,subclass,fold_index + 1])

# 调用函数生成CSV文件
generate_csv('./Negative', 'output.csv')