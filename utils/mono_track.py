import os
import torch
import torchaudio

def convert_to_mono_and_save(input_folder, output_folder):
    """
    读取文件夹下的所有音频文件，将其改为单声道并保存。

    参数:
    input_folder (str): 输入文件夹路径。
    output_folder (str): 输出文件夹路径。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有音频文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取音频文件
            waveform, sample_rate = torchaudio.load(input_path)

            # 将音频转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 保存处理后的音频文件
            torchaudio.save(output_path, waveform, sample_rate)
            print(f"Processed and saved: {output_path}")
