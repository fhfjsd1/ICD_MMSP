# 用于数据增强的音频csv文件生成（调用speechbrain的augmenter需要）

import os
import csv
import wave
import pandas as pd

def get_wav_info(wav_path):
    with wave.open(wav_path, 'r') as wav_file:
        duration = wav_file.getnframes() / float(wav_file.getframerate())
        return duration

# 读取 CSV 文件
csv_file_path = r'../data/metadata/all28479.csv'
df = pd.read_csv(csv_file_path)

filtered_df = df[(df['fold'] == 5) & (df['class_string'] == 'negative')]

# 读取这些数据行的第一列数据作为 filename
filenames = filtered_df.iloc[:, 0].tolist()

def write_wav_info_to_csv(folder_path, csv_path):
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['ID', 'duration', 'wav', 'wav_format', 'wav_opts']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for filename in filenames:
            if filename.endswith('.wav'):
                wav_path = os.path.join(folder_path, filename)
                duration = get_wav_info(wav_path)
                writer.writerow({
                    'ID': os.path.splitext(filename)[0],
                    'duration': duration,
                    'wav': f'SoundClassification/data/audio/negative/{filename}',
                    'wav_format': 'wav',
                    'wav_opts': ''
                })
