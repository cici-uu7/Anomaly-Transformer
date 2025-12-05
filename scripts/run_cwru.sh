#!/bin/bash

# 设置数据集名称和路径
DATASET_NAME="CWRU"
DATA_PATH="./dataset/CWRU"

echo "========================================================"
echo "Step 1: Running Data Preprocessing..."
echo "========================================================"
# 运行刚才生成的 Python 脚本
python preprocess_cwru.py

# 检查数据是否生成成功
if [ ! -f "$DATA_PATH/TRAIN.tsv" ]; then
    echo "Error: TRAIN.tsv not found. Preprocessing failed."
    exit 1
fi

echo "========================================================"
echo "Step 2: Training Anomaly Transformer on CWRU..."
echo "========================================================"

# 关键参数解释:
# --win_size 1000: 对应 CWRU 高频信号，捕捉足够长的周期
# --input_c 1: 单变量输入 (只用了 Drive End 加速度)
# --d_model 64: 降低模型复杂度，避免在小数据集上过拟合
# --batch_size 8: 防止 OOM (1000长度的Attention矩阵很大)
# --anormly_ratio 0.5: 这是一个先验估计，表示测试集中大概有多少比例是异常
# (在我们的预处理中，测试集包含了3种故障和1种正常，异常比例约为 75%，所以设 0.5-0.8 比较合适)

python run_cwru.py \
    --mode train \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --win_size 1000 \
    --input_c 1 \
    --output_c 1 \
    --d_model 64 \
    --d_ff 256 \
    --batch_size 8 \
    --num_epochs 10 \
    --lr 1e-4 \
    --anormly_ratio 0.5

echo "========================================================"
echo "Step 3: Evaluating Model..."
echo "========================================================"

python run_cwru.py \
    --mode test \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --win_size 1000 \
    --input_c 1 \
    --output_c 1 \
    --d_model 64 \
    --d_ff 256 \
    --batch_size 8 \
    --anormly_ratio 0.5

echo "Done! Check results in the 'results' directory."