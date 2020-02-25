# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from wavenet_vocoder.upsample_modules import Decoder, ConvNorm


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale, mode="nearest"):
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


def _get_activation(upsample_activation):
    nonlinear = getattr(nn, upsample_activation)
    return nonlinear


class UpsampleNetwork(nn.Module):
    def __init__(self, upsample_scales, upsample_activation="none",
                 upsample_activation_params={}, mode="nearest",
                 freq_axis_kernel_size=1, cin_pad=0, cin_channels=80):
        super(UpsampleNetwork, self).__init__()
        self.up_layers = nn.ModuleList()
        total_scale = np.prod(upsample_scales)
        self.indent = cin_pad * total_scale
        for scale in upsample_scales:
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            k_size = (freq_axis_kernel_size, scale * 2 + 1)
            padding = (freq_axis_padding, scale)
            stretch = Stretch2d(scale, 1, mode)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / np.prod(k_size))
            conv = nn.utils.weight_norm(conv)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)
            if upsample_activation != "none":
                nonlinear = _get_activation(upsample_activation)
                self.up_layers.append(nonlinear(**upsample_activation_params))

    def forward(self, c):
        """
        Args:
            c : B x C x T
        """

        # B x 1 x C x T
        c = c.unsqueeze(1)
        for f in self.up_layers:
            c = f(c)
        # B x C x T
        c = c.squeeze(1)

        if self.indent > 0:
            c = c[:, :, self.indent:-self.indent]
        return c


class ConvInUpsampleNetwork(nn.Module):
    def __init__(self, upsample_scales, upsample_activation="none",
                 upsample_activation_params={}, mode="nearest",
                 freq_axis_kernel_size=1, cin_pad=0,
                 cin_channels=80):
        super(ConvInUpsampleNetwork, self).__init__()
        # To capture wide-context information in conditional features
        # meaningless if cin_pad == 0
        ks = 2 * cin_pad + 1
        self.conv_in = nn.Conv1d(cin_channels, cin_channels, kernel_size=ks, bias=False)
        self.upsample = UpsampleNetwork(
            upsample_scales, upsample_activation, upsample_activation_params,
            mode, freq_axis_kernel_size, cin_pad=0, cin_channels=cin_channels)

    def forward(self, c):
        c_up = self.upsample(self.conv_in(c))
        return c_up


class TextUpsampleNetwork(torch.nn.Module):
    def __init__(self, encoder_embedding_dim, kernel_size, attention_rnn_dim, decoder_rnn_dim,
                 attention_dim, attention_location_n_filters, attention_location_kernel_size,
                 p_attention_dropout, p_decoder_dropout,
                 local_conditioning_dim, upsample_scales):
        super(TextUpsampleNetwork, self).__init__()
        convolutions = []
        for _ in range(2):
            conv_layer = torch.nn.Sequential(
                ConvNorm(encoder_embedding_dim, encoder_embedding_dim,
                         kernel_size=kernel_size,
                         padding=int((kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='relu'),
                torch.nn.BatchNorm1d(encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)

        self.lstm = torch.nn.LSTM(encoder_embedding_dim,
                                  int(encoder_embedding_dim / 2), 1,
                                  batch_first=True, bidirectional=True)

        self.decoder = Decoder(encoder_embedding_dim, attention_rnn_dim, decoder_rnn_dim,
                               attention_dim, attention_location_n_filters,
                               attention_location_kernel_size, p_attention_dropout,
                               p_decoder_dropout, local_conditioning_dim)

        self.postnet = UpsampleNetwork(upsample_scales, cin_channels=local_conditioning_dim)

    def forward(self, x, lengths, max_audio_length):
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = conv(x)

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        local_conditioning, gate_outputs, alignments = self.decoder(x, lengths, max_audio_length)
        c = self.postnet(local_conditioning)

        return c, gate_outputs, alignments
