import numpy as np
import pandas as pd
import torch
from torch import nn

from scipy.special import softmax
import os

from tqdm.autonotebook import tqdm


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, Nh):
        super(MultiHeadAttention, self).__init__()
        assert d_model % Nh == 0
        
        self.Q_lin = nn.Linear(d_model, d_model)
        self.K_lin = nn.Linear(d_model, d_model)
        self.V_lin = nn.Linear(d_model, d_model)
        
        self.WO_lin = nn.Linear(d_model, d_model)
        
        self.d_model = d_model
        self.Nh = Nh
        self.norm_const = np.sqrt(self.d_model // self.Nh)
        
    def forward(self, query, key, value, mask=None):
        
        Q = self.Q_lin(query)
        K = self.K_lin(key)
        V = self.V_lin(value)
        
        Q = Q.view(Q.shape[0], Q.shape[1], self.Nh, self.d_model // self.Nh).transpose(1,2)
        K = K.view(K.shape[0], K.shape[1], self.Nh, self.d_model // self.Nh).transpose(1,2)
        V = V.view(V.shape[0], V.shape[1], self.Nh, self.d_model // self.Nh).transpose(1,2)
        
        pre_scores = torch.matmul(Q,K.transpose(-2,-1)) / self.norm_const
        if mask is not None:
            pre_scores = pre_scores.masked_fill(mask.unsqueeze(1), -1e9)
            
        scores = torch.softmax(pre_scores, dim=-1)
        
        attended_result = torch.matmul(scores, V)
        attended_result = attended_result.transpose(1, 2).contiguous().view(query.shape)
        
        out_return = self.WO_lin(attended_result)
        
        return out_return

def attention(Q, K, V):
    pre_scores = torch.matmul(Q,K.transpose(-2,-1)) / torch.sqrt(Q.size(-2))
    scores = torch.softmax(pre_scores, dim=-1)
    attended_result = torch.matmul(scores, V.transpose(-2,-1))
    return attended_result

class FeedForward(nn.Module):
    def __init__(self, d_model, middle_dim):
        super(FeedForward, self).__init__()
        
        self.d_model = d_model
        self.middle_dim = middle_dim
        
        self.transormation = nn.Sequential(
            nn.Linear(d_model, middle_dim),
            nn.ReLU(True),
            nn.Linear(middle_dim, d_model)
        )
        
    def forward(self, X):
        return self.transormation(X)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, middle_dim, Nh):
        super(EncoderLayer, self).__init__()
        
        self.d_model = d_model
        self.middle_dim = middle_dim
        self.Nh = Nh
        
        self.attention = MultiHeadAttention(d_model, Nh)
        self.linear_part = FeedForward(d_model, middle_dim)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self,X,mask):
        
        part1 = self.layer_norm1(X + self.attention(X,X,X,mask))
        part2 = self.layer_norm2(part1 + self.linear_part(part1))
        
        return part2

class DecoderLayer(nn.Module):
    def __init__(self, d_model, middle_dim, Nh):
        super(DecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.middle_dim = middle_dim
        self.Nh = Nh
        
        self.attention1 = MultiHeadAttention(d_model, Nh)
        self.attention2 = MultiHeadAttention(d_model, Nh)
        self.linear_part = FeedForward(d_model, middle_dim)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
    def forward(self, Y, Encoded_X, mask_Y, mask_X):
        
        part1 = self.layer_norm1(Y + self.attention1(Y,Y,Y,mask_Y))
        part2 = self.layer_norm2(part1 + self.attention2(part1,Encoded_X,Encoded_X,mask_X))
        part3 = self.layer_norm3(part2 + self.linear_part(part2))
        
        return part3        

class Encoder(nn.Module):
    def __init__(self, d_model, middle_dim, Nh, N_layers=6):
        super(Encoder, self).__init__()
        
        self.d_model, self.middle_dim, self.Nh = d_model, middle_dim, Nh
        self.layer_list = nn.ModuleList([EncoderLayer(d_model, middle_dim, Nh) for i in range(N_layers)])
        
    def forward(self, X, input_mask):
        output = self.layer_list[0](X, input_mask)
        for layer in self.layer_list[1:]:
            output = layer(output, input_mask)
        return output
    
class Decoder(nn.Module):
    def __init__(self, d_model, middle_dim, Nh, N_layers=6):
        super(Decoder, self).__init__()
        
        self.d_model, self.middle_dim, self.Nh = d_model, middle_dim, Nh
        self.layer_list = nn.ModuleList([DecoderLayer(d_model, middle_dim, Nh) for i in range(N_layers)])
        
    def forward(self, Y, Encoded_X, output_mask, input_mask):
        output = self.layer_list[0](Y, Encoded_X, output_mask, input_mask)
        for layer in self.layer_list[1:]:
            output = layer(output, Encoded_X, output_mask, input_mask)        
        return output
    
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        
        self.transforms = nn.Sequential(
            nn.Linear(d_model,vocab),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, X):
        return self.transforms(X)

class Embedder(nn.Module):
    def __init__(self, vocab, d_model, max_len):
        super(Embedder, self).__init__()
        
        self.embedder = nn.Embedding(vocab,d_model)
        self.d_model = d_model
        
        self.pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float, requires_grad=False).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, requires_grad=False) *
                             -(np.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = nn.Parameter(self.pe)
        
    def forward(self, input_seqs):
        
        element_embedding = self.embedder(input_seqs) / np.sqrt(self.d_model)
        element_embedding = element_embedding + self.pe[:input_seqs.shape[-1],:]
        
        return element_embedding

class LossCompute(object):
    def __init__(self, d_model, label_smoothing, vocab_size, padding_index, device="cpu"):
        
        self.d_model = d_model
        self.label_smoothing = label_smoothing
        self.remainer_mass = (1 - label_smoothing) / (vocab_size - 2)
        self.vocab_size = vocab_size
        self.padding_index = padding_index
        self.loss = nn.KLDivLoss(reduction='sum')
        self.device=device
        
    def evaluate(self, true_indices, pred_probs):
        
        smooth_labels = self.smooth_code(true_indices)
        return self.loss(pred_probs, smooth_labels)
        
    def smooth_code(self, input_indices):

        smooth_labels = torch.zeros(
            (input_indices.shape[0],input_indices.shape[1],self.vocab_size), 
            requires_grad=False,
            device=self.device
        ) + self.remainer_mass
        
        smooth_labels.scatter_(-1, input_indices.unsqueeze(-1), self.label_smoothing)
        smooth_labels[:,:,self.padding_index] = 0.0
        smooth_labels[input_indices == self.padding_index] = 0.0

        return smooth_labels

def make_pad_mask(input_indices, pad_index=0):
    pad_mask = (input_indices == pad_index).unsqueeze(1)
    return pad_mask

def make_autoreg_mask(pad_mask):
    max_len = pad_mask.shape[-1]
    autoreg_mask = np.triu(np.ones((max_len,max_len)), 1)
    torch_autoreg_mask = torch.tensor(autoreg_mask, dtype=torch.uint8, requires_grad=False)
    torch_autoreg_mask = torch_autoreg_mask.unsqueeze(0)
    full_mask = torch_autoreg_mask | pad_mask
    return full_mask

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

if __name__ == "__main__":

    embdr = Embedder(11, 100, 9)

    encdr = Encoder(100, 200, 5)
    dcdr = Decoder(100, 200, 5)
    gnrtr = Generator(100, 11)

    lc = LossCompute(100, 0.8, 11, 0, device="cuda")

    embdr.cuda()
    encdr.cuda()
    dcdr.cuda()
    gnrtr.cuda()

    all_params = list(embdr.parameters()) + \
        list(encdr.parameters()) + \
        list(dcdr.parameters()) + \
        list(gnrtr.parameters())
    optimizer = NoamOpt(100, 1, 400,
            torch.optim.Adam(all_params, lr=0, betas=(0.9, 0.98), eps=1e-9))

    progress_bar = tqdm(range(5000))

    total_loss = 0
    n_tokens = 0

    for i in progress_bar:
        test_data_np = np.random.randint(0, 11, size=(300, 10))
        test_data_np[:,0] = 1
        test_data_np[:,-1] = 1
        src = torch.tensor(test_data_np[:,:-1].copy(), requires_grad=False).detach()
        tgt = torch.tensor(test_data_np[:,::-1][:,:-1].copy(), requires_grad=False).detach()

        src_mask = make_pad_mask(src).cuda()
        tgt_mask = make_pad_mask(tgt[:,:-1])
        tgt_mask = make_autoreg_mask(tgt_mask).cuda()
        
        src = src.cuda()
        tgt = tgt.cuda()
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()

        src_embedded = embdr(src)
        tgt_embedded = embdr(tgt)

        src_encoded = encdr(src_embedded, input_mask=src_mask)
        tgt_encoded = dcdr(tgt_embedded[:,:-1], src_encoded, tgt_mask, src_mask)
        output_preds = gnrtr(tgt_encoded)

        loss = lc.evaluate(tgt[:,1:], output_preds)
        
        total_loss += loss
        n_tokens += (test_data_np != 0).sum()

        loss.backward()
        optimizer.step()
        optimizer.optimizer.zero_grad()
        
        if i % 100 == 0:
            progress_bar.write("Loss per token: {}".format(total_loss / n_tokens))
            total_loss = 0
            n_tokens = 0
