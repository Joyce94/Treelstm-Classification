import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class Attention_b(nn.Module):
    def __init__(self, input_size, attention_size, use_cuda):
        super(Attention_b, self).__init__()
        self.input_size = input_size
        self.attention_size = attention_size
        self.linear_att = nn.Linear(self.input_size*2, self.attention_size, bias=True)
        self.u = Parameter(torch.randn(attention_size, 1))
        self.use_cuda = use_cuda

    def forward(self, h_i, h_t, mask):
        # h_i: variable (batch_size, len, two_hidden_size)
        # h_t: variable (batch_size, two_hidden_size)
        # mask: variable (batch_size, len_max)
        # print(h_i)
        batch_size = h_i.size(0)
        seq_len = h_i.size(1)
        lstm_hiddens = h_i.size(2)
        # print(h_t)
        h_t = h_t.unsqueeze(1).expand(batch_size, seq_len, lstm_hiddens)
        m_combine = torch.cat([h_i, h_t], 2)
        # m_combine: variable (batch_size, len, 2*two_hidden_size)
        m_combine = F.tanh(self.linear_att(m_combine))
        # m_combine: variable (batch_size, len, attention_size)
        # self.u: variable (attention_size, 1)
        m_combine = m_combine.transpose(0, 1)
        beta = []
        if mask is not None: mask_t = torch.t(mask)
        for idx in range(seq_len):
            if mask is None:
                beta.append(torch.mm(m_combine[idx], self.u))
            else:
                base = Variable(torch.FloatTensor([-1e+20]*batch_size))
                if self.use_cuda: base = base.cuda()
                y = torch.mm(m_combine[idx], self.u).squeeze(1)
                b = base.masked_scatter(mask_t[idx], y)
                # b = torch.mul(torch.mm(m_combine[idx], self.u), mask_t[idx].unsqueeze(1))
                # b: variable (batch_size)
                beta.append(b.unsqueeze(1))
            # if idx == 0:
            #     beta = torch.mm(m_combine[idx], self.u)
            # else:
            #     beta = torch.cat([beta, torch.mm(m_combine[idx], self.u)], 1)
        beta = torch.cat(beta, 1)
        # beta: variable (batch_size, len)
        if self.use_cuda: beta = beta.cuda()

        if self.use_cuda:
            alpha = F.softmax(beta, dim=1)  # alpha: variable (1, len)
        else:
            alpha = F.softmax(beta)         # alpha: variable (1, len)
        # alpha: variable (batch_size, len)

        # h_i: variable (batch_size, len, two_hidden_size)
        s = []
        for idx in range(batch_size):
            # torch.masked_select(h_i[idx], mask[idx])
            s.append(torch.mm(alpha[idx].unsqueeze(0), h_i[idx]))
        s = torch.cat(s, 0)
        if self.use_cuda: s = s.cuda()
        # alpha: variable (batch_size, len), h_i: variable (len, two_hidden_size)
        # s: variable (batch_size, two_hidden_size)
        # print(s)
        return s