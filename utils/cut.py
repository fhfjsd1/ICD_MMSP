import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import random
import torchaudio
import torch

# 定义增强函数
def speed_perturb(waveform, sample_rate, factor=1.2):
    """
    速度扰动: 通过改变采样率来改变播放速度
    """
    new_sample_rate = int(sample_rate * factor)
    return torchaudio.transforms.Resample(orig_freq=new_sample_rate, new_freq=sample_rate)(waveform)

def time_stretch(waveform, rate=1.1):
    """
    时间拉伸：通过修改时长，不改变音高
    """

    if not torch.is_complex(waveform):
    # 创建全零张量
        zeros = torch.zeros_like(waveform)

    # 拼接两个张量
        complex_waveform = torch.stack((waveform, zeros), dim=-1)

    # 确保输入张量是复数类型
    
        waveform = torch.view_as_complex(complex_waveform)
 
    return torchaudio.transforms.TimeStretch()(waveform, rate)

def pitch_shift(waveform, sample_rate, n_steps=2):
    """
    音高变化：通过改变音高
    """
    return torchaudio.transforms.PitchShift(sample_rate, n_steps=n_steps)(waveform)

def apply_augmentation(waveform, sample_rate):
    """
    随机选择并应用1到3种数据增强方式，返回增强后的waveform
    """
    augmentations = [speed_perturb, time_stretch, pitch_shift]
    num_augmentations = random.randint(1, 3)
    
    selected_augmentations = random.sample(augmentations, num_augmentations)
    for aug in selected_augmentations:
        if aug == speed_perturb:
            factor = random.uniform(0.9, 1.1)  # 轻微改变速度
            waveform = aug(waveform, sample_rate, factor)
        elif aug == time_stretch:
            rate = random.uniform(0.8, 1.2)  # 时间拉伸率
            waveform = aug(waveform, rate)
        elif aug == pitch_shift:
            n_steps = random.randint(-3, 3)  # 变调的音阶数
            waveform = aug(waveform, sample_rate, n_steps)
    
    return waveform

# 使用pydub重新加载，保证时长为5秒
def ensure_fixed_length(waveform, target_length_ms=3000, sample_rate=16000):
    """
    确保音频时长为固定长度（5秒），如不足则补充静音
    """
    target_length_samples = target_length_ms * sample_rate // 1000
    current_length = waveform.size(1)
    
    if current_length > target_length_samples:
        return waveform[:, :target_length_samples]
    elif current_length < target_length_samples:
        padding = torch.zeros((waveform.size(0), target_length_samples - current_length))
        return torch.cat([waveform, padding], dim=1)
    else:
        return waveform

# 读取文件夹中的音频文件并进行数据增强
def augment_audio_files(input_folder, output_folder, start_index, target_length_ms=5000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有wav文件
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    file_counter = start_index  # 从上一个文件名接着编号
    
    for wav_file in wav_files:
        # 加载音频文件
        file_path = os.path.join(input_folder, wav_file)
        waveform, sample_rate = torchaudio.load(file_path)
        
        print(f"Processing {wav_file}, waveform shape: {waveform.shape}")
        
        # 应用数据增强
        augmented_waveform = apply_augmentation(waveform, sample_rate)
        
        # 确保时长为5秒
        augmented_waveform = ensure_fixed_length(augmented_waveform, target_length_ms, sample_rate)
        
        # 保存增强后的文件
        classname = input_folder.split("_") [0]
        output_filename = os.path.join(output_folder, f"{classname}_{file_counter}.wav")
        torchaudio.save(output_filename, augmented_waveform, sample_rate)
        print(f"Saved augmented file: {output_filename}")
        file_counter += 1


# 定义切割音频的函数
def split_audio(input_folder, output_folder, chunk_length=5000, min_length=3000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有wav文件
    wav_files = [f for f in os.listdir(input_folder) if (f.endswith('.wav') or f.endswith('.ogg'))]
    
    file_counter = 1  # 用于为输出文件命名

    for wav_file in wav_files:
        # 加载音频文件
        file_path = os.path.join(input_folder, wav_file)
        audio = AudioSegment.from_wav(file_path)
        
        # 先切拼接完再切割，只保存最后的切割结果
        # 静音切除参数
        silence_thresh=-40
        min_silence_len=1000
        keep_silence=200
        # 切除静音部分
        chunks = split_on_silence(audio, 
                                min_silence_len=min_silence_len, 
                                silence_thresh=silence_thresh, 
                                keep_silence=keep_silence)
        
        # 拼接有声部分
        audio = AudioSegment.empty()
        for chunk in chunks:
            audio += chunk
        
        # 计算总时长
        total_length = len(audio)  # 单位是毫秒
        print(f"Processing {wav_file}, total length: {total_length / 1000:.2f}s")
        
        # 按chunk_length (5秒)切割音频
        for start in range(0, total_length, chunk_length):
            end = start + chunk_length
            if end <= total_length:
                # 正常切割的5秒片段
                chunk = audio[start:end]
            else:
                # 剩余部分不足5秒
                remaining = total_length - start
                if remaining < min_length:
                    # 如果剩余不足3秒，直接跳过
                    print(f"Skipping small segment of {remaining / 1000:.2f}s")
                    break
                else:
                    # 否则补充静音至5秒
                    chunk = audio[start:total_length] + AudioSegment.silent(duration=(chunk_length - remaining))
                    print(f"Padding segment from {remaining / 1000:.2f}s to 5s")
            
            # 保存切割后的文件
            output_filename = os.path.join(output_folder, f"{os.path.basename(input_folder)}_{file_counter}.wav")
            chunk.export(output_filename, format="wav")
            print(f"Saved {output_filename}")
            file_counter += 1
            
    return file_counter

def remove_silence_and_save(input_folder, output_folder, silence_thresh=-30, min_silence_len=3000, keep_silence=500):
    """
    对单个原始音频相对静音的部分进行切除，有声部分拼接成一个新的音频文件并编号保存。
    
    参数:
    - input_file: 输入音频文件路径
    - output_folder: 输出文件夹路径
    - silence_thresh: 静音阈值（单位：dB）
    - min_silence_len: 最小静音长度（单位：毫秒）
    - keep_silence: 保留的静音长度（单位：毫秒）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有wav文件
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    for wav_file in wav_files:
    # 加载音频文件
        audio = AudioSegment.from_wav(os.path.join(input_folder,wav_file))
        
        # 切除静音部分
        chunks = split_on_silence(audio, 
                                min_silence_len=min_silence_len, 
                                silence_thresh=silence_thresh, 
                                keep_silence=keep_silence)
        
        # 拼接有声部分
        combined_audio = AudioSegment.empty()
        for chunk in chunks:
            combined_audio += chunk
        
        # 保存拼接后的音频文件
        base_name = os.path.basename(wav_file).split(".")[0]
        output_file = os.path.join(output_folder, f"{base_name}.wav")
        combined_audio.export(output_file, format="wav")
        print(f"Saved {output_file}")

        

