import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1, max_norm=2, norm_type=3)
        # self.E2 = torch.nn.Embedding(len(d.entities), d1)
        self.R_low = torch.nn.Embedding(len(d.relations), d2)
        self.R_high = torch.nn.Embedding(len(d.relations), d2)
        # self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
        #                              dtype=torch.float, device="cuda", requires_grad=True))
        self.W_low = torch.nn.Parameter(torch.tensor(np.random.uniform(0, 0.5, (d2, d1, d1)),
                                                     dtype=torch.float, device="cuda", requires_grad=True))
        self.W_high = torch.nn.Parameter(torch.tensor(np.random.uniform(0.5, 1, (d2, d1, d1)),
                                                     dtype=torch.float, device="cuda", requires_grad=True))
        # self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),  # 核心张量大小 参数范围（-1，1）
        #                             dtype=torch.float, device="cuda", requires_grad=True))
        # self.W = self.W_low + self.W_high
        self.bias = kwargs["bias"]  # 作为超参 参与训练
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R_low.weight.data)
        xavier_normal_(self.R_high.weight.data)

    def forward(self, e1_idx, r_idx):
        # self.E.weight.data = self.E.weight.data + self.E2.weight.data
        # self.W = self.W_low + self.W_high

        e1 = self.E(e1_idx)
        # print(e1)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        #  self.W_low = torch.clamp(self.W_low, 0, 1)
        r_low = self.R_low(r_idx)
        # print(r)
        W_mat = torch.mm(r_low, self.W_low.view(r_low.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred_1 = torch.sigmoid(x)

        r_high = self.R_high(r_idx)

        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))
        # self.W_high = torch.clamp(self.W_high, 0, 1)  # 保证正向作用不被抵消。
        W_mat = torch.mm(r_high, self.W_high.view(r_high.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)
        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred_2 = torch.sigmoid(x)

        # 计算margin_loss
        W_temp = self.W_low - self.W_high
        W_temp += self.bias
        W_temp = torch.clamp(W_temp, -0.)
        W_temp = torch.mean(W_temp)

        # norm = 0
        # norm += 0.5 * torch.sum(torch.abs(e1) ** 3)
        # norm += 1.5 * torch.sum(torch.abs(r) ** 3)
        # DURA_loss = norm / e1.shape[0]

        return pred_1, pred_2, W_temp  # , DURA_loss
    def forward_2(self, e1_idx, r_idx):

        # self.R.weight.data = self.R.weight.data + self.R_bias.weight.data    # 给R添加bias结果很差。
        self.W = self.W_low + self.W_high
        e1 = self.E(e1_idx)  # 实体索引 [128,200]
        # x = self.bn0(e1)  # 对实体e1做归一化操作。
        # x = self.input_dropout(x)  # 对e1做dropout操作
        x = e1.view(-1, 1, e1.size(1))  # 对x的形状重新构造大小为   torch.Size([128, 1, 200])

        r1 = (r_idx - len(self.d.relations)) // len(self.d.relations)
        r2 = r_idx % len(self.d.relations)
        # print(r_idx, r1, r2)
        r_1 = self.R(r1)
        r_2 = self.R(r2)
        W_mat = torch.mm(r_1, self.W.view(r_1.size(1), -1))  # 模式一乘积结果
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))      # W *1 R的结果。
        x = torch.bmm(x, W_mat)  # torch.Size([128, 1, 200]) 模式二乘积，e1在r1和W下的嵌入结果。
        x = x.view(-1, e1.size(1))  # torch.Size([128, 200])

        e2 = x
        e2 = self.bn0(e2)  # 对实体e2做归一化操作。
        e2 = self.input_dropout(e2)  # 对e2做dropout操作
        e2 = x.view(-1, 1, e1.size(1))  # 对x的形状重新构造大小为   torch.Size([128, 1, 200])
        self.W = self.W_low + self.W_high

        W_mat_2 = torch.mm(r_2, self.W.view(r_2.size(1), -1))  # 模式一
        W_mat_2 = W_mat_2.view(-1, e1.size(1), e1.size(1))
        W_mat_2 = self.hidden_dropout1(W_mat_2)
        x = torch.bmm(e2, W_mat_2)  # torch.Size([128, 1, 200])  模式二
        x = x.view(-1, e1.size(1))  # torch.Size([128, 200])
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))  # torch.Size([128, 104])   模式三乘积  [128,200] * [200,104]
        pred = torch.sigmoid(x)

        # 计算margin_loss
        W_temp = self.W_low - self.W_high
        W_temp += self.bias
        W_temp = torch.clamp(W_temp, -0.)
        W_temp = torch.mean(W_temp)

        return pred, W_temp

