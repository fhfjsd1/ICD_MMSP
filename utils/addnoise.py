import torch

def add_noise_with_snr(positive_sample, noise_sample, snr):
    """
    Custom noise addition function: add noise audio to the positive sample based on the specified signal-to-noise ratio (SNR).

    Parameters:
        positive_sample (Tensor): Audio data of the positive sample.
        noise_sample (Tensor): Audio data of the noise sample.
        snr (float): Signal-to-noise ratio.

    Returns:
        Tensor: Augmented audio data.
    """
    # 确保噪声音频长度与正类样本相同
    if noise_sample.shape[1] > positive_sample.shape[1]:
        noise_sample = noise_sample[:, :positive_sample.shape[1]]
    else:
        noise_sample = torch.nn.functional.pad(noise_sample, (0, positive_sample.shape[1] - noise_sample.shape[1]))

    # 计算信噪比
    positive_power = positive_sample.norm(p=2)
    noise_power = noise_sample.norm(p=2)
    scale = positive_power / (10 ** (snr / 20) * noise_power)

    # 叠加噪声音频
    augmented_sample = positive_sample + scale * noise_sample
    
    return augmented_sample
