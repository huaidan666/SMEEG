import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init  # 
from Models.Attention import *
from mamba_ssm import Mamba  # 
import pywt
import pywt.data
from functools import partial


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        # 
        if isinstance(dims, int):
            dims = (dims,)  # 
        
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

    
class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv2d, self).__init__()

        # 
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)
        
        # 
        self.scale = _ScaleModule(in_channels)
        
        # 
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # 
        x = self.depthwise(x)
        
        # 
        #x = self.scale(x)
        #
        x = self.pointwise(x)
        return x


class SpeMamba(nn.Module):
    def __init__(self,channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self,x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj



class DisjoinEncoder(nn.Module):
    def __init__(self, channel_size, emb_size, rep_size, kernel_size):
        super().__init__()
        self.temporal_CNN = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[1, kernel_size], padding='same'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        self.spatial_CNN = nn.Sequential(nn.Conv2d(emb_size, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                         nn.BatchNorm2d(emb_size),
                                         nn.GELU())

        self.rep_CNN = nn.Sequential(nn.Conv1d(emb_size, rep_size, kernel_size=3, padding=1),
                                     nn.BatchNorm1d(rep_size),
                                     nn.GELU())
        
        # 添加下采样层
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.initialize_weights()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_CNN(x)
        x = self.spatial_CNN(x)
        x = self.rep_CNN(x.squeeze())
        x = self.maxpool(x)  # 
        x = x.transpose(1, 2)  # 
        return x

    def initialize_weights(self):
        # Custom weight initialization, you can choose different methods
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize convolutional layer weights using Xavier/Glorot initialization
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    # Initialize biases with zeros
                    init.constant_(m.bias, 0)




def Encoder_factory(config):
    model = SMEEG(config, num_classes=config['num_labels'])
    return model


class SMEEG(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']  # d_x
        
        config['pooling_size'] = 2
        seq_len = int(seq_len / config['pooling_size'])
        self.InputEmbedding = InputEmbedding(config)
        self.InputEmbedding_f = DisjoinEncoder(channel_size, emb_size, emb_size, kernel_size=8)
        self.PositionalEncoding = PositionalEmbedding(seq_len, emb_size)
        # -------------------------------------------------------------------------
        self.momentum = config['momentum']
        self.device = config['device']
        self.mask_ratio = config['mask_ratio']
        self.mask_len = int(config['mask_ratio'] * seq_len)
        self.mask_token = nn.Parameter(torch.randn(emb_size, ))
        self.contex_encoder = Encoder(config)
        self.target_encoder = copy.deepcopy(self.contex_encoder)
        self.Predictor = Predictor(emb_size, config['num_heads'], config['dim_ff'], 1, config['pre_layers'])
        self.predict_head = nn.Linear(2 * emb_size, config['num_labels'])  
        self.Norm = nn.LayerNorm(emb_size)
        self.Norm2 = nn.LayerNorm(emb_size)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(p=0.3)  

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def linear_prob(self, x):
        with (torch.no_grad()):
            
            patches = self.InputEmbedding(x)
            
            patches = self.Norm(patches)
            patches = patches + self.PositionalEncoding(patches)
            patches = self.Norm2(patches)
            out = self.contex_encoder(patches)
            out = out.transpose(2, 1)
            out = self.gap(out)
            
            
            x_f = torch.fft.fft(x).float()
            patches_f = self.InputEmbedding_f(x_f)
            
            # 
            patches_f = self.Norm(patches_f)
            patches_f = patches_f + self.PositionalEncoding(patches_f)
            #
            patches_f = self.Norm2(patches_f)
            out_f = self.contex_encoder(patches_f)
            out_f = out_f.transpose(2, 1)
            out_f = self.gap(out_f)
            
            # 
            return torch.cat((out.squeeze(), out_f.squeeze()), dim=1)

    def pretrain_forward(self, x):
        patches = self.InputEmbedding(x)  #
        patches = self.Norm(patches)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)

        # 
        x_f = torch.fft.fft(x).float()
        patches_f = self.InputEmbedding_f(x_f)
        patches_f = self.Norm(patches_f)
        patches_f = patches_f + self.PositionalEncoding(patches_f)
        patches_f = self.Norm2(patches_f)
        
        # 
        combined_patches = torch.cat((patches, patches_f), dim=2)  # 

        rep_mask_token = self.mask_token.repeat(combined_patches.shape[0], combined_patches.shape[1], 1)
        rep_mask_token = rep_mask_token + self.PositionalEncoding(rep_mask_token)

        index = np.arange(patches.shape[1])
        index_chunk = Semantic_Subsequence_Preserving(index, 2, self.mask_ratio)
        v_index = np.ravel(index_chunk)
        m_index = np.setdiff1d(index, v_index)

        visible = patches[:, v_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]
        rep_contex = self.contex_encoder(visible)
        with torch.no_grad():
            rep_target = self.target_encoder(patches)
            rep_mask = rep_target[:, m_index, :]
        rep_mask_prediction = self.Predictor(rep_contex, rep_mask_token)
        return [rep_mask, rep_mask_prediction, rep_contex, rep_target]

    def forward(self, x):
        # 
        patches = self.InputEmbedding(x)
        patches = self.Norm(patches)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)
        out = self.contex_encoder(patches)
        out = self.dropout(out)  # 
        out = torch.mean(out, dim=1)
        
        # 
        x_f = torch.fft.fft(x).float()
        patches_f = self.InputEmbedding_f(x_f)
        patches_f = self.Norm(patches_f)
        patches_f = patches_f + self.PositionalEncoding(patches_f)
        patches_f = self.Norm2(patches_f)
        out_f = self.contex_encoder(patches_f)

        out_f = self.dropout(out_f)  # 
        out_f = torch.mean(out_f, dim=1)
        
        # 
        combined_features = torch.cat((out, out_f), dim=1)
        return self.predict_head(combined_features)


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']  # 
        k = 7
        # Embedding Layer -----------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(channel_size, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.SiLU = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, config['pooling_size']), stride=(1, config['pooling_size']))
        # 
        self.wtconv = DepthwiseSeparableConvWithWTConv2d(in_channels=emb_size, out_channels=emb_size)



    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.depthwise_conv(out)  # 
        #
        out = self.wtconv(out)  # 
       
       
        out = out.transpose(1, 2)  # (bs, 1, embedding, T)
        out = self.spatial_padding(out)        
        out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
        out = self.SiLU(out)
        out = self.maxpool(out)  # (bs, 1, embedding, T // m)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)
        out = out.squeeze(1)  # (bs, embedding, T // m)
        out = out.transpose(1, 2)  # (bs, T // m, embedding)
        patches = self.SiLU(out)
        return patches


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        d_model = config['emb_size']
        attn_heads = config['num_heads']
        # d_ffn = 4 * d_model
        d_ffn = config['dim_ff']
        layers = config['layers']
        dropout = config['dropout']
        enable_res_parameter = True
        # # TRMs
        # self.TRMs = nn.ModuleList(
        #     [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

        #---------------------
        self.TRMs = nn.ModuleList(
            [SpeMamba(channels=d_model, use_residual=True, group_num=4) for i in range(layers)])




    def forward(self, x):

        #------------------->

        # 
        x = x.transpose(1, 2).unsqueeze(-1)  # (B, L, C) -> (B, C, L, 1)
        for TRM in self.TRMs:
            x = TRM(x)
        x = x.squeeze(-1).transpose(1, 2)  # (B, C, L, 1) -> (B, L, C)



        #------------------------------<

        # for TRM in self.TRMs:
        #     x = TRM(x, mask=None)

        return x


def Semantic_Subsequence_Preserving(time_step_indices, chunk_count, target_percentage):
    # Get the total number of time steps
    total_time_steps = len(time_step_indices)
    # Calculate the desired total time steps for the selected chunks
    target_total_time_steps = int(total_time_steps * target_percentage)

    # Calculate the size of each chunk
    chunk_size = target_total_time_steps // chunk_count

    # Randomly select starting points for each chunk with minimum distance
    start_points = [random.randint(0, total_time_steps - chunk_size)]
    # Randomly select starting points for each subsequent chunk with minimum distance
    for _ in range(chunk_count - 1):
        next_start_point = random.randint(0, total_time_steps - chunk_size)
        start_points.append(next_start_point)

    # Select non-overlapping chunks using indices
    selected_chunks_indices = [time_step_indices[start:start + chunk_size] for start in start_points]

    return selected_chunks_indices


class Predictor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class MultiScaleWaveletEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 
        self.wtconv_low = DepthwiseSeparableConvWithWTConv2d(wt_levels=1)  # 
        self.wtconv_high = DepthwiseSeparableConvWithWTConv2d(wt_levels=2)  # 
        self.fusion = nn.Conv2d(emb_size*2, emb_size, 1)  # 
    
    def forward(self, x):
        low_freq = self.wtconv_low(x)
        high_freq = self.wtconv_high(x)
        fused = self.fusion(torch.cat([low_freq, high_freq], dim=1))
        return fused
