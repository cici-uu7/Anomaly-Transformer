import os
import urllib.request
import scipy.io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ================= 配置区域 =================
# 数据集保存路径
DATA_DIR = './dataset/CWRU_AD'

# CWRU 官网数据下载链接
DATA_URLS = {
    '97.mat': 'https://engineering.case.edu/sites/default/files/97.mat',  # Normal
    '105.mat': 'https://engineering.case.edu/sites/default/files/105.mat',  # IR Fault
    '118.mat': 'https://engineering.case.edu/sites/default/files/118.mat',  # Ball Fault
    '130.mat': 'https://engineering.case.edu/sites/default/files/130.mat'  # OR Fault
}

# 窗口大小
WIN_SIZE = 1000
# 滑动步长
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
    try:
        mat = scipy.io.loadmat(filepath)
        # 查找包含 'DE_time' 的键
        for key in mat.keys():
            if 'DE_time' in key:
                return mat[key].flatten()
        raise ValueError(f"No DE_time data found in {filepath}")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.array([])


def create_segments(signal, win_size, stride, label):
    """执行滑动窗口切片"""
    if len(signal) < win_size:
        return np.empty((0, win_size)), np.empty((0,))

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


def save_to_tsv(segments, labels, filename):
    """保存为 TSV 格式 (第一列为 Label)"""
    # 拼接 Label 和 Data
    data_with_label = np.hstack([labels.reshape(-1, 1), segments])
    df = pd.DataFrame(data_with_label)
    save_path = os.path.join(DATA_DIR, filename)
    df.to_csv(save_path, sep='\t', header=False, index=False)
    print(f"Saved {filename}: {df.shape}")


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

    # 检查数据是否加载成功
    if len(normal_signal) == 0:
        print("Error: Normal signal is empty. Exiting.")
        return

    # 3. 数据集划分 (Train 60% / Val 20% / Test 20%)
    total_len = len(normal_signal)
    train_split = int(total_len * 0.6)
    val_split = int(total_len * 0.8)

    # 切分正常信号
    train_signal_raw = normal_signal[:train_split]
    val_signal_raw = normal_signal[train_split:val_split]
    test_norm_signal_raw = normal_signal[val_split:]

    # 4. 数据标准化 (Standardization)
    # 重要: 仅在训练集上拟合 Scaler，防止数据泄露
    print("Applying Standardization...")
    scaler = StandardScaler()

    # 以此 fit (需要 reshape 为 2D 数组)
    scaler.fit(train_signal_raw.reshape(-1, 1))

    # 转换所有数据 (Transform)
    train_signal = scaler.transform(train_signal_raw.reshape(-1, 1)).flatten()
    val_signal = scaler.transform(val_signal_raw.reshape(-1, 1)).flatten()
    test_norm_signal = scaler.transform(test_norm_signal_raw.reshape(-1, 1)).flatten()

    # 同时也标准化故障数据 (使用从正常数据学到的分布)
    ir_signal = scaler.transform(ir_signal.reshape(-1, 1)).flatten()
    ball_signal = scaler.transform(ball_signal.reshape(-1, 1)).flatten()
    or_signal = scaler.transform(or_signal.reshape(-1, 1)).flatten()

    # 5. 滑动窗口切片
    print("Segmenting data...")
    # 训练集
    train_segs, train_lbls = create_segments(train_signal, WIN_SIZE, STRIDE, label=0)

    # 验证集
    val_segs, val_lbls = create_segments(val_signal, WIN_SIZE, STRIDE, label=0)

    # 测试集 - 正常部分 (Label=0)
    test_norm_segs, test_norm_lbls = create_segments(test_norm_signal, WIN_SIZE, STRIDE, label=0)

    # 测试集 - 故障部分 (Label=1)
    ir_segs, ir_lbls = create_segments(ir_signal, WIN_SIZE, STRIDE, label=1)
    ball_segs, ball_lbls = create_segments(ball_signal, WIN_SIZE, STRIDE, label=1)
    or_segs, or_lbls = create_segments(or_signal, WIN_SIZE, STRIDE, label=1)

    # 拼接测试集
    test_segs = np.vstack([test_norm_segs, ir_segs, ball_segs, or_segs])
    test_lbls = np.concatenate([test_norm_lbls, ir_lbls, ball_lbls, or_lbls])

    # 打印统计信息
    print("\nDataset Statistics:")
    print(f"Train Segments (Normal): {len(train_segs)}")
    print(f"Val Segments   (Normal): {len(val_segs)}")
    print(f"Test Segments  (Total) : {len(test_segs)}")
    print(f"  - Normal (Unseen)    : {len(test_norm_segs)}")
    print(f"  - Anomalies          : {len(ir_segs) + len(ball_segs) + len(or_segs)}")

    # 计算测试集正常与异常的比例
    n_anom = len(test_segs) - len(test_norm_segs)
    if n_anom > 0:
        ratio = len(test_norm_segs) / n_anom
        print(f"  - Test Ratio (Norm:Anom): 1 : {1 / ratio:.2f}")

    # 6. 保存文件
    save_to_tsv(train_segs, train_lbls, 'TRAIN.tsv')
    save_to_tsv(val_segs, val_lbls, 'VAL.tsv')
    save_to_tsv(test_segs, test_lbls, 'TEST.tsv')

    print(f"\nAll files saved to {DATA_DIR}")


if __name__ == '__main__':
    main()