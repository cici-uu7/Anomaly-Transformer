import torch
import torch.nn as nn
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader_cwru import get_loader_cwru


def my_kl_loss(p, q):
    # p, q shape: [Batch, Heads, Win, Win]
    # small eps to prevent log(0)
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss


class SolverCWRU(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(SolverCWRU.DEFAULTS, **config)

        # 加载数据
        self.train_loader = get_loader_cwru(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                            mode='train')
        self.vali_loader = get_loader_cwru(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                           mode='val')
        self.test_loader = get_loader_cwru(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                           mode='test')
        self.thre_loader = get_loader_cwru(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                           mode='test')

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            d_model=self.d_model,
            d_ff=self.d_ff,
            e_layers=3
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                rec_loss = self.criterion(output, input)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # 优化：利用广播机制，移除 .repeat()，节省显存
                    # sum(dim=-1) 后是 [B, H, L]，keepdim=True 变为 [B, H, L, 1]
                    # prior[u] 是 [B, H, L, L]，直接除以 [B, H, L, 1] 即可
                    prior_norm = prior[u] / (torch.sum(prior[u], dim=-1, keepdim=True) + 1e-8)

                    series_loss += (torch.mean(my_kl_loss(series[u], prior_norm.detach())) +
                                    torch.mean(my_kl_loss(prior_norm.detach(), series[u])))
                    prior_loss += (torch.mean(my_kl_loss(prior_norm, series[u].detach())) +
                                   torch.mean(my_kl_loss(series[u].detach(), prior_norm)))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss_1.append((rec_loss - self.k * series_loss).item())
                loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE ======================")
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()

            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)
                rec_loss = self.criterion(output, input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # 优化：移除 .repeat()
                    prior_norm = prior[u] / (torch.sum(prior[u], dim=-1, keepdim=True) + 1e-8)

                    series_loss += (torch.mean(my_kl_loss(series[u], prior_norm.detach())) +
                                    torch.mean(my_kl_loss(prior_norm.detach(), series[u])))
                    prior_loss += (torch.mean(my_kl_loss(prior_norm, series[u].detach())) +
                                   torch.mean(my_kl_loss(series[u].detach(), prior_norm)))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

                # 显式清理缓存，防止碎片化
                if (i + 1) % 50 == 0:
                    torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature =5

        print("======================TEST MODE======================")
        criterion = nn.MSELoss(reduce=False)

        # (1) Statistic on the train set
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # 优化：移除 .repeat()
                    prior_norm = prior[u] / (torch.sum(prior[u], dim=-1, keepdim=True) + 1e-8)

                    if u == 0:
                        series_loss = my_kl_loss(series[u], prior_norm.detach()) * temperature
                        prior_loss = my_kl_loss(prior_norm, series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], prior_norm.detach()) * temperature
                        prior_loss += my_kl_loss(prior_norm, series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) Find threshold & (3) Evaluation on test set
        attens_energy = []
        test_labels_list = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # 优化：移除 .repeat()
                    prior_norm = prior[u] / (torch.sum(prior[u], dim=-1, keepdim=True) + 1e-8)

                    if u == 0:
                        series_loss = my_kl_loss(series[u], prior_norm.detach()) * temperature
                        prior_loss = my_kl_loss(prior_norm, series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], prior_norm.detach()) * temperature
                        prior_loss += my_kl_loss(prior_norm, series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels_list.append(labels.cpu().numpy())

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        test_labels = np.concatenate(test_labels_list, axis=0).reshape(-1)
        gt = (test_labels > 0).astype(int)
        gt = np.repeat(gt, self.win_size)
        pred = (test_energy > thresh).astype(int)

        print("pred shape: ", pred.shape)
        print("gt shape: ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))

        return accuracy, precision, recall, f_score