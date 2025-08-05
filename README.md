# Infant Cry Detection in Noisy Environments Using Blueprint Separable Convolutions and Time-Frequency Recurrent Neural Networks

This repository provides the dataset description and code implementation as detailed in the referenced paper on infant cry detection for audio classification tasks.

## Dataset

### Statistics

| Item                   | Description                                          |
|:----------------------:|:-----------------------------------------------------|
| Audio Types            | Infant cries + negative samples (noise, human voices)|
| Number of Classes      | 2 (positive: infant cries; negative: non-infant cries)|
| Sampling Rate          | 16 kHz                                               |
| Duration per Audio     | ~5 seconds                                           |
| Data Augmentation      | Room Impulse Response (RIR), noise addition, time-frequency masking |

### Structure

```
data/
├── audio/                # Audio files
│   ├── negative/         # Negative samples (non-infant cries)
│   ├── positive/         # Positive samples (infant cries)
│   └── rir/              # Room impulse responses (for augmentation)
└── metadata/             # Annotation files
    ├── all28479.csv      # Full dataset annotations
    ├── noise.csv         # Noise annotations for training
    ├── noise_v.csv       # Noise annotations for evaluation
    └── rir.csv           # RIR annotations
```

### Preprocessing Pipeline

1. **Audio Format Conversion** (`utils/mono_track.py`)  
   Converts stereo to mono.
2. **Audio Segmentation** (`utils/cut.py`)  
   Removes silence and splits audio into fixed-length segments.
3. **Noise Addition** (`utils/addnoise.py`)  
   Manually adds background noise.
4. **CSV Generation** (`utils/make_csv_aug.py`)  
   Creates annotation files for augmented data.
5. **Data Selection** (`utils/select.py`)  
   Filters specific classes from the ESC-50 dataset.





## Usage

1. Organize the dataset as described above.  
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Edit `hparams.yaml` to set hyperparameters.  
4. Train the model:  
   ```bash
   python train_test.py hparams.yaml
   ```  
5. To run evaluation only, set `TEST = True` in `train_test.py`.

## Acknowledgments

Part of this code is based on [speechbrain](https://github.com/speechbrain/speechbrain).

## Contact

Yanxiong Li (eeyxli@scut.edu.cn)  and Haolin Yu (202230244210@mail.scut.edu.cn)  
School of Electronic and Information Engineering, South China University of Technology, Guangzhou, China

## Citation

If you use this project in your research, please cite:

H. Yu and Y. Li, “Infant cry detection in noisy environment using blueprint separable convolutions and time-frequency recurrent neural network,” in 27th IEEE International Workshop on Multimedia Signal Processing (MMSP), IEEE, 2025.
