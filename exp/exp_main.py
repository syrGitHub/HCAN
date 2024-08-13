from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_all, test_params_flop
from utils.metrics import metric
from utils.confusion_matrix import Confusion_Matrix

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


def symmetric_kl_divergence(p, q):
    kl_pq = torch.nn.functional.kl_div(torch.log(p), q, reduction='batchmean')
    kl_qp = torch.nn.functional.kl_div(torch.log(q), p, reduction='batchmean')
    symmetric_kl = 0.5 * (kl_pq + kl_qp)
    return symmetric_kl

# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def un_ce_loss(p, evidence, c, global_step, annealing_step):
    alpha = softplus_evidence(evidence) + 1
    S = torch.sum(alpha, dim=-1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    bu = E / S
    lam = (1 - bu)
    A = torch.sum(lam * label * (torch.digamma(S) - torch.digamma(alpha)), dim=-1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    uncertainty = c / S
    pred = alpha / S

    final = A + B
    return torch.mean(final), pred, uncertainty


def softplus_evidence(logits):
    return F.softplus(logits)


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, reg_c, label_c, reg_f, label_f) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                reg_c = reg_c.float().to(self.device)
                label_c = label_c.long().to(self.device)
                reg_f = reg_f.float().to(self.device)
                label_f = label_f.long().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x)
                        else:
                            # if self.args.output_attention:
                            outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x)
                    else:
                        # if self.args.output_attention:
                        outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                coarse_prediction = coarse_prediction[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                reg_c = reg_c[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                coarse_Logit = coarse_Logit[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                label_c = label_c[:, -self.args.pred_len:, f_dim:].to(self.device)
                fine_prediction = fine_prediction[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                reg_f = reg_f[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                fine_Logit = fine_Logit[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                label_f = label_f[:, -self.args.pred_len:, f_dim:].to(self.device)
                fine_Logit_switch = fine_Logit_switch[:, -self.args.pred_len:, f_dim:, :].to(self.device)

                # ACL_loss
                loss_acl = symmetric_kl_divergence(F.softmax(coarse_Logit, dim=-1),
                                                   F.softmax(fine_Logit_switch, dim=-1))
                attention_mask = None
                # coarse_loss
                coarse_Logit = coarse_Logit.reshape(outputs.shape[0], -1, self.args.num_coarse)
                label_c = label_c.reshape(outputs.shape[0], -1)
                coarse_loss_edl, coarse_pred, coarse_uncertainty = un_ce_loss(label_c, coarse_Logit, self.args.num_coarse, i, self.args.train_epochs)
                mask_c = reg_c >= 0
                if mask_c.sum() != 0:
                    loss_reg_coarse = criterion(coarse_prediction[mask_c], reg_c[mask_c])  # 正确

                # fine_loss
                fine_Logit = fine_Logit.reshape(outputs.shape[0], -1, self.args.num_fine)
                label_f = label_f.reshape(outputs.shape[0], -1)
                fine_loss_edl, fine_pred, fine_uncertainty = un_ce_loss(label_f, fine_Logit, self.args.num_fine, i, self.args.train_epochs)
                mask_f = reg_f >= 0
                if mask_f.sum() != 0:
                    loss_reg_fine = F.mse_loss(fine_prediction[mask_f], reg_f[mask_f])  # 正确

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # Direct_loss
                loss_direct = criterion(pred, true)
                loss = self.args.lambda_cls * coarse_loss_edl + self.args.lambda_reg * loss_reg_coarse + \
                       self.args.lambda_cls * fine_loss_edl + self.args.lambda_reg * loss_reg_fine + \
                       self.args.lambda_acl * loss_acl + self.args.lambda_direct * loss_direct

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            loss_reg_coarse = 0.0
            loss_reg_fine = 0.0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, reg_c, label_c, reg_f, label_f) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                reg_c = reg_c.float().to(self.device)
                label_c = label_c.long().to(self.device)
                reg_f = reg_f.float().to(self.device)
                label_f = label_f.long().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x)
                        else:
                            # if self.args.output_attention:
                            outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                        outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x)
                    else:
                        # if self.args.output_attention:
                        outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]  # torch.Size([32, 24, 1])
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    coarse_prediction = coarse_prediction[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                    reg_c = reg_c[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                    coarse_Logit = coarse_Logit[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                    label_c = label_c[:, -self.args.pred_len:, f_dim:].to(self.device)
                    fine_prediction = fine_prediction[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                    reg_f = reg_f[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                    fine_Logit = fine_Logit[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                    label_f = label_f[:, -self.args.pred_len:, f_dim:].to(self.device)
                    fine_Logit_switch = fine_Logit_switch[:, -self.args.pred_len:, f_dim:, :].to(self.device)

                    # ACL_loss
                    loss_acl = symmetric_kl_divergence(F.softmax(coarse_Logit, dim=-1),
                                                       F.softmax(fine_Logit_switch, dim=-1))
                    attention_mask = None

                    # coarse_loss
                    coarse_Logit = coarse_Logit.reshape(outputs.shape[0], -1, self.args.num_coarse)
                    label_c = label_c.reshape(outputs.shape[0], -1)
                    coarse_loss_edl, coarse_pred, coarse_uncertainty = un_ce_loss(label_c, coarse_Logit, self.args.num_coarse, i, self.args.train_epochs)
                    mask_c = reg_c >= 0
                    if mask_c.sum() != 0:
                        loss_reg_coarse = criterion(coarse_prediction[mask_c], reg_c[mask_c])  # 正确

                    # fine_loss
                    fine_Logit = fine_Logit.reshape(outputs.shape[0], -1, self.args.num_fine)
                    label_f = label_f.reshape(outputs.shape[0], -1)
                    fine_loss_edl, fine_pred, fine_uncertainty = un_ce_loss(label_f, fine_Logit, self.args.num_fine, i, self.args.train_epochs)
                    mask_f = reg_f >= 0
                    if mask_f.sum() != 0:
                        loss_reg_fine = F.mse_loss(fine_prediction[mask_f], reg_f[mask_f])  # 正确

                    # Direct_loss
                    loss_direct = criterion(outputs, batch_y)
                    loss = self.args.lambda_cls * coarse_loss_edl + self.args.lambda_reg * loss_reg_coarse + \
                           self.args.lambda_cls * fine_loss_edl + self.args.lambda_reg * loss_reg_fine + \
                           self.args.lambda_acl * loss_acl + self.args.lambda_direct * loss_direct
                    train_loss.append(loss.item())
                    # print(coarse_loss_edl, loss_reg_coarse, fine_loss_edl, loss_reg_fine, loss_acl, loss_direct, loss)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} | {} cost time: {:.4f}s".format(epoch + 1, self.args.train_epochs, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, reg_c, label_c, reg_f, label_f) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x)
                        else:
                            # if self.args.output_attention:
                            outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x)
                    else:
                        # if self.args.output_attention:
                        outputs, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                coarse_Logit = coarse_Logit[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                label_c = label_c[:, -self.args.pred_len:, f_dim:].to(self.device)
                fine_Logit = fine_Logit[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                label_f = label_f[:, -self.args.pred_len:, f_dim:].to(self.device)

                coarse_loss_edl, coarse_pred, coarse_uncertainty = un_ce_loss(label_c, coarse_Logit,
                                                                              self.args.num_coarse, i,
                                                                              self.args.train_epochs)
                fine_loss_edl, fine_pred, fine_uncertainty = un_ce_loss(label_f, fine_Logit, self.args.num_fine, i,
                                                                        self.args.train_epochs)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        digits_preds = preds[:, -1, :]
        digits_trues = trues[:, -1, :]
        visual_all(digits_trues, digits_preds, os.path.join(folder_path, 'All.png'))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr_time, corr = metric(preds, trues)
        print('rmse:{}, mse:{}, mae:{}, corr:{}'.format(rmse, mse, mae, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mse:{}, mae:{}, rse:{}, corr:{}, corr_time:{}'.format(rmse, mse, mae, rse, corr, corr_time))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        with open(f'./results/' + setting + '/pred_class_coarse.pkl', 'wb') as f1:
            pickle.dump(coarse_pred, f1, pickle.HIGHEST_PROTOCOL)
        with open(f'./results/' + setting + '/pred_class_fine.pkl', 'wb') as f2:
            pickle.dump(fine_pred, f2, pickle.HIGHEST_PROTOCOL)
        # Confusion_Matrix(coarse_pred, label_c, f'./test_results/' + setting + '/cm_coarse.png')
        # Confusion_Matrix(fine_pred, label_f, f'./test_results/' + setting + '/cm_fine.png')

        return

    def predict(self, setting, load=False):
        """
        用来生成之后一段时间的预测，如生成2018-01-01 00:00:00-2018-01-01 23:00:00的预测值
        """
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                batch_y = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(batch_y)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        trues = np.array(trues)
        trues = np.concatenate(trues, axis=0)
        if (pred_data.scale):
            for data in preds:
                preds = pred_data.inverse_transform(data)
            for data in trues:
                trues = pred_data.inverse_transform(data)
        # print(preds, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds, axis=1),
                     columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
