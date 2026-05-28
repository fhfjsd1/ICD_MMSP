# Infant Cry Detection in Noisy Environments Using Blueprint Separable Convolutions and Time-Frequency Recurrent Neural Networks
# 基于蓝图可分离卷积和时频循环神经网络的嘈杂环境婴儿哭声检测

This repository provides the dataset description and code implementation as detailed in the [referenced paper](#citation--引用) on infant cry detection for audio classification tasks.
本仓库提供了 [Citation](#citation--引用) 中关于婴儿哭声检测（音频分类任务）论文里所描述的数据集说明和代码实现。

[PAPER-arXiv](https://arxiv.org/abs/2508.19308#:~:text=In%20this%20paper%2C%20we%20propose%20a%20lightweight%20and,a%20time-frequency%20recurrent%20neural%20network%20for%20adaptive%20denoising.)

[PAPER-IEEE Xplore](https://ieeexplore.ieee.org/document/11324248)

[Dataset](https://huggingface.co/datasets/hoyyu1/infant-cry-detection)


## News / 最新动态

- 🎉 Our paper "Infant Cry Detection In Noisy Environment Using Blueprint Separable Convolutions and Time-Frequency Recurrent Neural Network" has been officially accepted by **IEEE MMSP 2025**!
我们的论文已被 **IEEE MMSP 2025** 国际会议正式接收！
- 🚀 We have officially open-sourced our comprehensive experimental dataset on Hugging Face!
我们已经在 [Hugging Face](https://huggingface.co/datasets/hoyyu1/infant-cry-detection) 平台上正式开源了完整的实验数据集！

## Dataset / 数据集

The dataset used in this project is available on Hugging Face. You can access and download it from the following link:
本项目中使用的数据集已发布在 Hugging Face 上。您可以从以下链接访问并下载：

[Hugging Face Dataset: infant-cry-detection](https://huggingface.co/datasets/hoyyu1/infant-cry-detection)

### Preprocessing Pipeline / 预处理流程

1. **Audio Format Conversion / 音频格式转换** (`utils/mono_track.py`)  
   Converts stereo to mono.
   将双通道立体声转换为单声道。
2. **Audio Segmentation / 音频分割** (`utils/cut.py`)  
   Removes silence and splits audio into fixed-length segments.
   去除静音并将其分割为固定长度的片段。
3. **Noise Addition / 噪声添加** (`utils/addnoise.py`)  
   Manually adds background noise.
   手动添加背景噪声。
4. **CSV Generation / 标签生成** (`utils/make_csv_aug.py`)  
   Creates annotation files for augmented data.
   为增强后的数据创建标注文件。
5. **Data Selection / 数据筛选** (`utils/select.py`)  
   Filters specific classes from the ESC-50 dataset.
   从 ESC-50 数据集中筛选特定类别的数据。

## Usage / 使用方法

This codebase has been refactored and organized. Please note that some configurations may need to be adapted to your local environment to run successfully (using an AI assistant to help with configuration is highly recommended). We hope this repository serves as a strong starting point and framework for your research and development.
本代码仓库已经过整理。请注意，部分配置需要根据您的本地情况进行适配才能成功运行（强烈建议利用 AI 助手协助配置）。我们希望这个仓库能为您提供一个很好的代码框架起点。

1. Organize the dataset as described above.  
   按照上述说明组织数据集结构。
2. Install dependencies:
   安装依赖项：
    ```bash
    pip install -r requirements.txt
    ```
3. Edit `hparams.yaml` to set hyperparameters.  
   编辑 `hparams.yaml` 文件以设置超参数。
4. Train the model:  
   训练模型：
   ```bash
   python train_test.py hparams.yaml
   ```  
5. To run evaluation only, set `TEST = True` in `train_test.py`.
   若仅运行评估测试，请在 `train_test.py` 中设置 `TEST = True`。

## Acknowledgments / 致谢

Part of this code is based on [speechbrain](https://github.com/speechbrain/speechbrain).
本代码的部分实现基于 [speechbrain](https://github.com/speechbrain/speechbrain)。

## Citation / 引用

If you use this repository in your research, please cite the following paper: (You can find the paper in this repository)
如果您在研究中使用了该代码仓库，请引用以下论文：（您可以在这个仓库中找到该论文）

```
H. Yu and Y. Li, "Infant Cry Detection In Noisy Environment Using Blueprint Separable Convolutions and Time-Frequency Recurrent Neural Network," 2025 IEEE International Workshop on Multimedia Signal Processing (MMSP), Beijing, China, 2025, pp. 1-6, doi: 10.1109/MMSP64401.2025.11324248.
```
