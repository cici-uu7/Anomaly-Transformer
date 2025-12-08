#!/bin/bash

# ================= 配置区域 =================
# 1. 基础配置
DATASET_NAME="CWRU"
DATA_PATH="./dataset/CWRU_AD"

# 2. 定义通用的模型参数 (训练和测试必须保持一致的部分!)
# 使用变量保存，避免重复书写，也避免手误导致训练/测试参数不一致
MODEL_ARGS="--dataset $DATASET_NAME \
            --data_path $DATA_PATH \
            --win_size 1000 \
            --input_c 1 \
            --output_c 1 \
            --d_model 64 \
            --d_ff 256 \
            --batch_size 8 \
            --anormly_ratio 88.3"

# ================= 脚本执行区域 =================

echo "========================================================"
echo "Step 1: Running Data Preprocessing..."
echo "========================================================"
# 运行预处理 (如果数据已存在且不需要重新生成，可以注释掉这一行以节省时间)
python ./AD/preprocess_cwru.py

if [ ! -f "$DATA_PATH/TRAIN.tsv" ]; then
    echo "Error: TRAIN.tsv not found."
    exit 1
fi

echo "========================================================"
echo "Step 2: Training..."
echo "========================================================"
# 直接引用 $MODEL_ARGS，只需要补充训练特有的参数 (如 lr, epochs)
python run_cwru.py \
    --mode train \
    --num_epochs 10 \
    --lr 1e-4 \
    $MODEL_ARGS


echo "========================================================"
echo "Step 3: Evaluating..."
echo "========================================================"
# 直接引用 $MODEL_ARGS，参数完全复用，确保一致性
python run_cwru.py \
    --mode test \
    $MODEL_ARGS

echo "Done!"