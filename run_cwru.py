import os
import argparse
from torch.backends import cudnn
from utils.utils import *
from solver_cwru import SolverCWRU  # 导入新建的 Solver


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)

    # 调用新的 Solver
    solver = SolverCWRU(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)

    # CWRU 特有参数设置
    # TimesNet 中 seq_len = 1000，这里 win_size 对应序列长度
    parser.add_argument('--win_size', type=int, default=1000)

    # CWRU 是单变量振动数据，所以 input_c 和 output_c 设为 1
    parser.add_argument('--input_c', type=int, default=1)
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)  # 显存允许的话可以调大
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CWRU')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])


    # 数据路径：指向 CWRU TSV 文件所在的文件夹
    parser.add_argument('--data_path', type=str, default='./dataset/CWRU_10')

    parser.add_argument('--model_save_path', type=str, default='checkpoints/cwru')

    # 异常比例：假设除了 0 (Normal) 以外都是异常。
    # 如果数据集中有 10 类，大概 90% 是异常。这对于无监督检测来说是个挑战（通常假设异常很少）。
    # 这里先设一个比较高的值，或者您可以根据验证集调整。
    parser.add_argument('--anormly_ratio', type=float, default=10.0)
    # 修改 2: 新增 d_model 参数 (建议设为 64，原模型默认为 512 太大了)
    parser.add_argument('--d_model', type=int, default=64)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)