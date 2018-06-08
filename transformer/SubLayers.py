''' Define the sublayers in encoder/decoder layer '''

import torch
import torch.nn as nn
import torch.nn.init as init
from transformer.Modules import BottleLinear as Linear
from transformer.Modules import ScaledDotProductAttention
#from transformer.Modules import BottleLayerNormalization as LayerNormalization
from transformer.Modules import LayerNormalization

__author__ = "Yu-Hsiang Huang and Egor Kraev"

class MultiHeadAttentionStep(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, enc_output = None):
        '''

        :param n_head:
        :param d_model:
        :param d_k:
        :param d_v:
        :param dropout:
        '''
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention()
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def init_encoder_output(self, enc_output):
        """
        Pre-calculates k and v for encoder output, to save both time and RAM
        :param enc_output: batch_size x seq_len x d_model
        :return:
        """
        self.dec_q = []
        self.enc_output = enc_output # if that's None, that just resets the state
        if self.enc_output is None:
            self.dec_k = []
            self.dec_v = []
        else:
            self.enc_k = []
            self.enc_v = []
            # pre-calc
            for i in range(self.enc_output.size()[1]):
                _, k, v = self.vec_to_qkv(self.enc_output[:,i,:])  # go over the sequence, collect k and v
                self.enc_k.append(k)
                self.enc_v.append(v)
            self.enc_k = torch.cat(self.enc_k, 1)
            self.enc_v = torch.cat(self.enc_v, 1)

    def vec_to_qkv(self, x):
        '''
        Converts batch of vectors to batches of q,k,v
        :param x: batch_size x d_model
        :return:
        '''
        batch_size = x.size()[0]
        x_nice = x.unsqueeze(0).expand(self.n_head,-1,-1) # result is self.n_head x batch_size x d_model, and no new memory allocation
        q = torch.bmm(x_nice, self.w_qs).transpose(0, 1).reshape(self.n_head*batch_size,1, self.d_k)
        k = torch.bmm(x_nice, self.w_ks).transpose(0, 1).reshape(self.n_head*batch_size,1, self.d_k)
        v = torch.bmm(x_nice, self.w_vs).transpose(0, 1).reshape(self.n_head*batch_size,1, self.d_v)
        return q, k, v


    def forward(self, x, attn_mask=None):
        '''
        Calculates attention-based output from the next input, caching intermediate results
        :param x: batch_size x d_model
        :param attn_mask:
        :return: output same dimension as x
        '''

        self.batch_size = x.size()[0]

        q, k, v = self.vec_to_qkv(x)
        q_s = q #n_head*batch_size x 1 x d_q

        if self.enc_output is None:
            # append the new k, v to the array so far
            self.dec_k.append(k)
            self.dec_v.append(v)
            k_s = torch.cat(self.dec_k,1)  # n_head*batch_size x len_k x d_k
            v_s = torch.cat(self.dec_v,1)  # n_head*batch_size x len_v x d_v
        else: # use the pre-calc'd values
            k_s = self.enc_k
            v_s = self.enc_v

        # perform attention on the latest input, result size# batch_size*n_head x 1 x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(self.n_head, 1, 1))
        # project back to residual size
        outputs = self.proj(outputs.view(self.batch_size, -1)) # batch_size x n_head*d_v
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + x), attns

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention()
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)

class PositionwiseFeedForwardStep(nn.Module):
    """
    A two-feed-forward-layer module

    """

    def __init__(self, d_model, d_inner_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner_hid)
        self.w_2 = nn.Linear(d_inner_hid, d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward step
        :param x: batch_size x d_model
        :return: batch_size x d_model
        """
        # x is batch of vectors, so need no dimension juggling
        output = self.relu(self.w_1(x))
        output = self.w_2(output)
        output = self.dropout(output)
        return self.layer_norm(output + x)
