# #!/usr/bin/env python3
# """Helper to create Confusion Matrix figure

import matplotlib.pyplot as plt
import numpy as np

def create_cm_fig(cm, display_labels):
    """
    绘制混淆矩阵并保存图像。

    参数:
    cm (numpy.ndarray): 混淆矩阵数据。
    display_labels (list): 类别标签。
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制混淆矩阵
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    
    # 添加颜色条
    plt.colorbar(cax)
    
    # 设置 x 轴和 y 轴
    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)
    
    # 在每个位置显示值
    fmt = 'd'  # 格式化字符串，整数
    thresh = cm.max() / 2.  # 阈值，用于设置文本颜色
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 设置标签
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    
    return fig