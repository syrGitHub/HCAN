import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_coarse = configs.num_coarse
        self.num_fine = configs.num_fine
        self.d_l = configs.hidden_dim
        self.channels = configs.enc_in

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # g_Linear layers
        self.g_proj1 = nn.Linear(self.pred_len, self.d_l)
        self.g_proj2 = nn.Linear(self.d_l, self.pred_len)
        self.predictor = nn.Linear(self.pred_len, self.pred_len)

        # Coarse Classification layers
        self.c_proj1 = nn.Linear(self.pred_len, self.d_l)
        self.coarse_predictor = nn.Linear(self.d_l, self.pred_len * self.num_coarse)
        self.coarse_Classify_Layer = nn.Linear(self.d_l, self.pred_len * self.num_coarse)

        # Fine Classification layers
        self.f_proj1 = nn.Linear(self.pred_len, self.d_l)
        self.fine_predictor = nn.Linear(self.d_l, self.pred_len * self.num_fine)
        self.fine_Classify_Layer = nn.Linear(self.d_l, self.pred_len * self.num_fine) # [batch, Output_length, Channel * num_Classes]

    def pairwise_sum(self, tensor):
        # 将细粒度的分类结果相邻项进行相加合并，转化为与粗粒度相同的格式大小
        # 张量的维度展为 [batch_size, class_num]
        size = tensor.size()
        # 将展平后的张量分割为两部分
        even = tensor[:, :, :, 0:size[1]:2]
        odd = tensor[:, :, :, 1:size[1]:2]
        # 计算相邻项的和
        pairwise_sum = (even + odd)/2

        # print(pairwise_sum/2)
        # print(pairwise_sum, even, odd, tensor)
        return pairwise_sum

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        x = dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
        x = x.permute(0, 2, 1)  # [batch, channel, seq_len[]
        # Coarse prediction
        c_proj = self.c_proj1(x)  # torch.Size([32, 5, 300]) [batch, channel, hidden]
        coarse_prediction = self.coarse_predictor(
            c_proj)  # torch.Size([32, 5, 48])  [batch, channel, pred_len * num_class]
        coarse_prediction = coarse_prediction.view(coarse_prediction.shape[0], -1, self.channels,
                                                   self.num_coarse)  # [Batch_size, Channels, Pred_len, num_coarse]
        coarse_Logit = self.coarse_Classify_Layer(c_proj)
        coarse_Logit = coarse_Logit.view(coarse_prediction.shape[0], -1, self.channels, self.num_coarse)

        # Fine prediction
        f_proj = self.f_proj1(x)
        fine_prediction = self.fine_predictor(f_proj)  # .squeeze(-1)
        fine_prediction = fine_prediction.view(fine_prediction.shape[0], -1, self.channels, self.num_fine)
        fine_Logit = self.fine_Classify_Layer(f_proj)
        fine_Logit = fine_Logit.view(fine_prediction.shape[0], -1, self.channels, self.num_fine)

        fine_Logit_switch = self.pairwise_sum(fine_Logit)

        # Direct prediction
        g_proj = self.g_proj1(x)

        # None-local module
        g_x = g_proj.view(g_proj.shape[0], self.d_l, -1)  # [Batch, Hidden, Channel]
        g_x = g_x.permute(0, 2, 1)
        theta_x = c_proj.view(c_proj.shape[0], self.d_l, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = f_proj.view(f_proj.shape[0], self.d_l, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        W_y = self.g_proj2(y)

        # residual connection
        z = W_y + x  # [Batch, Channel, pred_len]

        direct_predictor = self.predictor(z).permute(0, 2, 1)  # [Batch, Output length, Channel]

        return direct_predictor, coarse_prediction, coarse_Logit, fine_prediction, fine_Logit, fine_Logit_switch
