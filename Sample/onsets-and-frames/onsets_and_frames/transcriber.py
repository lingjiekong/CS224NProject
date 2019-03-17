"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/onsets_and_frames/onsets_frames_transcription/onsets_and_frames.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        fc_size = model_complexity * 16
        lstm_units = model_complexity * 8

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, fc_size),
            BiLSTM(fc_size, lstm_units),
            nn.Linear(lstm_units * 2, output_features),
            # nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, fc_size),
            BiLSTM(fc_size, lstm_units),
            nn.Linear(lstm_units * 2, output_features),
            # nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, fc_size),
            nn.Linear(fc_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            BiLSTM(output_features * 3, lstm_units),
            nn.Linear(lstm_units * 2, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, fc_size),
            nn.Linear(fc_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, mel):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        onset_time_label = batch['onset_time']
        offset_time_label = batch['offset_time']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        predictions = {
            # 'onset': onset_pred.reshape(*onset_label.shape),
            # 'offset': offset_pred.reshape(*offset_label.shape),
            'onset_time': onset_pred.reshape(*onset_time_label.shape),
            'offset_time': offset_pred.reshape(*offset_time_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            # 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            # 'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/onset_time': self.onset_time_loss(predictions['onset_time'], onset_time_label, onset_label),
            'loss/offset_time': self.offset_time_loss(predictions['offset_time'], offset_time_label, offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses

    def onset_time_loss(self, onset_pred_time, onset_label_time, onset_label):
        # denominator = onset_label.sum()
        row, col = onset_label.size()
        denominator = onset_label.sum()
        denominator = (denominator/denominator)*row*col
        # return (onset_label * (onset_label_time - onset_pred_time) ** 2).sum() / denominator
        return ((onset_label_time - onset_pred_time) ** 2).sum() / denominator


    def offset_time_loss(self, offset_pred_time, offset_label_time, offset_label):
        # denominator = offset_label.sum()
        row, col = offset_pred_time.size()
        denominator = offset_label.sum()
        denominator = (denominator/denominator)*row*col
        # return (offset_label * (offset_label_time - offset_pred_time) ** 2).sum() / denominator
        return ((offset_label_time - offset_pred_time) ** 2).sum() / denominator



    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

