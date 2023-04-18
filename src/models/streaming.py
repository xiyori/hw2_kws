from typing import Tuple, Union, List, Callable, Optional
from tqdm import tqdm
from itertools import islice
import pathlib

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

import torchaudio

from ..configs import TaskConfig
from .crnn import CRNN


class StreamingCRNN(nn.Module):
    gru_output: torch.Tensor

    def __init__(self, path: str):
        super().__init__()

        config = TaskConfig(device=torch.device("cpu"))

        self.crnn = CRNN(config)
        self.crnn.load_state_dict(torch.load(path))
        self.crnn.eval()

        self.win_length = 400
        self.hop_length = 160
        self.kernel_time = config.kernel_size[1]
        self.stride_time = config.stride[1]
        self.input_step_size = self.hop_length * self.stride_time

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=400,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=config.n_mels
        ).to(config.device)

        d = 2 if self.crnn.gru.bidirectional else 1
        self.hidden = torch.zeros((d * self.crnn.gru.num_layers, 1, self.crnn.gru.hidden_size),
                                  dtype=torch.float32)

        self.gru_output = torch.tensor([], dtype=torch.float32,
                                       device=config.device)
        self.mel_output = torch.tensor([], dtype=torch.float32,
                                       device=config.device)
        self.current_length = 0

    def forward(self, input: torch.Tensor, max_window_length: int):
        input = input.unsqueeze(dim=0)

        mel_output = torch.log(
            self.melspec(input).clamp_(min=1e-9, max=1e9)
        ).unsqueeze(dim=1)

        self.mel_output = torch.cat([self.mel_output, mel_output], dim=-1)

        conv_output = self.crnn.conv(self.mel_output).transpose(-1, -2)
        gru_output, hidden = self.crnn.gru(conv_output, self.hidden)
        self.hidden = hidden

        if self.current_length < max_window_length:
            old_gru_output = self.gru_output
            self.current_length += 1
        else:
            step_size = gru_output.shape[1]
            old_gru_output = self.gru_output[:, step_size:]
        self.gru_output = torch.cat([old_gru_output, gru_output], dim=1)

        drop_frames = ((self.mel_output.shape[-1] - self.kernel_time) //
                       self.stride_time) * self.stride_time + self.kernel_time
        self.mel_output = self.mel_output[..., drop_frames + 1:]

        contex_vector = self.crnn.attention(self.gru_output)
        logits = self.crnn.classifier(contex_vector)
        probs = F.softmax(logits, dim=-1)
        output = probs.squeeze(dim=0)[..., 1]
        return output
