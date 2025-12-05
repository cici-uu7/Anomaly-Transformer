import os
import glob
import urllib.request
import scipy.io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ================= 配置区域 =================
# 数据集保存路径
DATA_DIR = './dataset/CWRU'
# CWRU 官网数据下载链接 (12k Fan End & Drive End)
# 97.mat: Normal Baseline (0 hp)
# 105.mat: Inner Race Fault (0.007", 0 hp)
# 118.mat: Ball Fault (0.007", 0 hp)
# 130.mat: Outer Race Fault (0.007", 0 hp, Center @ 6:00)
DATA_URLS = {
    '97.mat': 'https://engineering.case.edu/sites/default/files/97.mat',
    '105.mat': 'https://engineering.case.edu/sites/default/files/105.mat',
    '118.mat': 'https://engineering.case.edu/sites/default/files/118.mat',
    '130.mat': 'https://engineering.case.edu/sites/default/files/130.mat'
}

# 窗口大小 (需与模型 win_size 一致)
WIN_SIZE = 1000
# 滑动步长 (训练集通常不重叠，测试集可重叠)
STRIDE = 1000


def download_data(save_dir):
    """如果数据不存在，则从 CWRU 官网下载"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename, url in DATA_URLS.items():
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")
        else:
            print(f"Found {filename}")


def load_mat_data(filepath):
    """读取 .mat 文件并提取 DE (Drive End) 振动信号"""
    mat = scipy.io.loadmat(filepath)
    # 查找包含 'DE_time' 的键 (驱动端加速度数据)
    for key in mat.keys():
        if 'DE_time' in key:
            return mat[key].flatten()
    raise ValueError(f"No DE_time data found in {filepath}")


def create_segments(signal, win_size, stride, label):
    """执行滑动窗口切片"""
    n_samples = (len(signal) - win_size) // stride + 1
    segments = []
    labels = []

    for i in range(n_samples):
        start = i * stride
        end = start + win_size
        segment = signal[start:end]
        segments.append(segment)
        labels.append(label)

    return np.array(segments), np.array(labels)


def main():
    # 1. 准备目录和原始数据
    raw_data_dir = os.path.join(DATA_DIR, 'raw')
    download_data(raw_data_dir)

    # 2. 读取数据
    print("Processing data...")
    normal_signal = load_mat_data(os.path.join(raw_data_dir, '97.mat'))
    ir_signal = load_mat_data(os.path.join(raw_data_dir, '105.mat'))
    ball_signal = load_mat_data(os.path.join(raw_data_dir, '118.mat'))
    or_signal = load_mat_data(os.path.join(raw_data_dir, '130.mat'))

    # 3. 构建数据集
    # 训练集: 仅使用正常数据 (前 80%)
    train_len = int(len(normal_signal) * 0.8)
    train_segs, train_lbls = create_segments(normal_signal[:train_len], WIN_SIZE, STRIDE, label=0)

    # 验证集: 使用正常数据 (后 20%)
    val_segs, val_lbls = create_segments(normal_signal[train_len:], WIN_SIZE, STRIDE, label=0)

    # 测试集: 包含少量正常数据 + 所有类型的故障数据
    # 为了测试平衡，我们取部分正常数据作为对照
    test_norm_segs, test_norm_lbls = create_segments(normal_signal[train_len:], WIN_SIZE, STRIDE, label=0)

    # 故障数据切片 (Label=1)
    ir_segs, ir_lbls = create_segments(ir_signal, WIN_SIZE, STRIDE, label=1)
    ball_segs, ball_lbls = create_segments(ball_signal, WIN_SIZE, STRIDE, label=1)
    or_segs, or_lbls = create_segments(or_signal, WIN_SIZE, STRIDE, label=1)

    # 拼接测试集
    test_segs = np.vstack([test_norm_segs, ir_segs, ball_segs, or_segs])
    test_lbls = np.concatenate([test_norm_lbls, ir_lbls, ball_lbls, or_lbls])

    # 4. 保存为 TSV 格式
    # 格式要求: 第1列是 Label，第2列到最后是 Features (Win_Size)
    # 注意: Anomaly Transformer 代码要求无 Header

    def save_to_tsv(segments, labels, filename):
        # 拼接 Label 和 Data
        data_with_label = np.hstack([labels.reshape(-1, 1), segments])
        df = pd.DataFrame(data_with_label)
        save_path = os.path.join(DATA_DIR, filename)
        df.to_csv(save_path, sep='\t', header=False, index=False)
        print(f"Saved {filename}: {df.shape}")

    save_to_tsv(train_segs, train_lbls, 'TRAIN.tsv')
    save_to_tsv(val_segs, val_lbls, 'VAL.tsv')
    save_to_tsv(test_segs, test_lbls, 'TEST.tsv')

    print("\nData preprocessing complete!")
    print(f"Files saved to {DATA_DIR}")
    print("Structure check:")
    print(f"- TRAIN.tsv (Normal only): Used for model training")
    print(f"- VAL.tsv (Normal only): Used for early stopping")
    print(f"- TEST.tsv (Mixed): Used for anomaly detection evaluation")


if __name__ == '__main__':
    main()