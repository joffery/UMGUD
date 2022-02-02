import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu2_1 = nn.ReLU(inplace=True)

        self.fc1var = nn.Linear(2*64*28*28, 3)
        self.sig1var = nn.Sigmoid()

        self.fc1 = nn.Linear(128*5*5, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(1024, 10)

    def noise(self, mu, logvar, return_std_pre=False, return_std=False, eval=False):

        std = torch.exp(0.5*logvar)

        if return_std_pre:
            return mu, std

        std = torch.clamp(std, 0, 1)

        mu = torch.clamp(mu, -2, 2)

        if return_std:
            return mu, std

        eps = torch.randn_like(mu)

        if eval:
            return mu

        return mu + eps*std

    def forward(self, x, mix=True, return_feat=False, noise_layer=False, eval=False):

        in_size = x.size(0)

        if not noise_layer:
            out1 = self.relu1(self.mp(self.conv1(x)))
            out2 = self.relu2(self.mp(self.conv2(out1)))
        else:
            noise1 = F.softplus(self.noise(self.conv1_1(x), self.conv1_2(x), eval=eval))
            conv1 = self.conv1(x)
            out1 = self.relu1(self.mp(conv1))
            out1_noise = self.relu1_1(self.mp(conv1 + noise1))

            noise2 = F.softplus(self.noise(self.conv2_1(out1_noise), self.conv2_2(out1_noise), eval=eval))
            conv2 = self.conv2(out1)
            conv2_noise = self.conv2(out1_noise)
            out2_o = self.relu2(self.mp(conv2))
            out2_noise = self.relu2_1(self.mp(conv2_noise + noise2))

            conv1_mean, conv1_var = self.noise(self.conv1_1(x), self.conv1_2(x), return_std=True)

            conv_var_cat = torch.cat([conv1_mean.reshape(in_size, -1), conv1_var.reshape(in_size, -1)], dim=1).cuda()

            absmo= self.fc1var(conv_var_cat.detach())

            a, b, smo = torch.split(absmo, 1, dim=1)
            smo = self.sig1var(smo)

            # m = Bernoulli(torch.max(smo-0.3, torch.zeros_like(smo)))
            m = Bernoulli(smo)
            mask = m.sample().bool()

            if mix:
                # a, b = 2 + torch.clamp(a, -0.5, 0.5), 1 + torch.clamp(b, -0.5, 0.5)
                a, b = 1 + torch.clamp(a, -0.5, 1), 1 + torch.clamp(b, -0.5, 1)
                beta = Beta(a.unsqueeze(-1).unsqueeze(-1), b.unsqueeze(-1).unsqueeze(-1))
                lam = beta.sample()
                out2 = lam * out2_o.detach() + (1 - lam) * out2_noise.detach()
            else:
                lam = 0
                out2 = out2_noise

        out2_n = out2.reshape(in_size, -1)
        out3 = self.relu3(self.fc1(out2_n))
        out4 = self.relu4(self.fc2(out3))

        if return_feat:
            if not noise_layer:
                return out1, out2, out4, self.fc3(out4)
            else:
                return mask, out1_noise, out2_noise, out4, self.fc3(out4)
        else:
            if noise_layer:
                return lam, mask, self.fc3(out4)
            else:
                return self.fc3(out4)

class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target, mask, lam):

        mask = mask.squeeze()

        log_prob = F.log_softmax(input, dim=-1)

        weight = input.new_zeros(input.size())
        weight_ori = input.new_zeros(input.size())

        weight[mask] += self.smoothing / (input.size(-1) - 1.)

        weight[mask] = weight[mask].scatter_(-1, target[mask].unsqueeze(-1), (1. - self.smoothing))

        weight[~mask] = weight[~mask].scatter_(-1, target[~mask].unsqueeze(-1), 1.)
        weight_ori = weight_ori.scatter_(-1, target.unsqueeze(-1), 1.)
        final_weight = lam*weight_ori + (1-lam)*weight
        loss = (-final_weight * log_prob).sum(dim=-1).mean()

        return loss