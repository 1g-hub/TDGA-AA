import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transform
import numpy as np


class LSTMFourScene(nn.Module):
    def __init__(self, wrn_model, mid_units=50, out_dim=3):
        super(LSTMFourScene, self).__init__()
        self.wrn_model = wrn_model  # 特徴抽出部
        for param in self.wrn_model.parameters():  # 特徴抽出部の重みは変更しない
            param.requires_grad = False

        # last_layer = list(self.wrn_model.children())[-3:]  # ファインチューニング用
        # for l in last_layer:
        #     for param in l.parameters():
        #         param.requires_grad = True

        self.LSTM = nn.LSTM(input_size=self.wrn_model.nStages[3], hidden_size=self.wrn_model.nStages[3], num_layers=1, batch_first=True, bidirectional=False)
        # self.emb = nn.Linear(self.wrn_model.nStages[3], mid_units)
        # self.out = nn.Linear(mid_units, out_dim)

        self.out = nn.Linear(self.wrn_model.nStages[3], out_dim)

    def forward(self, images):
        images = images.permute(1, 0, 2, 3, 4)
        x = []
        for i in range(4):
            out = images[i]
            out = self.wrn_model.conv1(out)
            out = self.wrn_model.layer1(out)
            out = self.wrn_model.layer2(out)
            out = self.wrn_model.layer3(out)
            out = F.relu(self.wrn_model.bn1(out))
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = out.view(1, out.size(0), -1)
            x.append(out)
        x = torch.cat(x, dim=0)
        x = x.permute(1, 0, 2)
        x, _ = self.LSTM(x)
        x = x[:, -1, :]  # Final Layer output
        x = F.selu(x)
        x = self.out(x)
        return nn.LogSoftmax(dim=1)(x)
