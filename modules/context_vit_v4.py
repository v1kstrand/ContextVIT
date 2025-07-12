import torch.nn as nn

class LinearContextViTv4(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.m = nn.Identity()

    def forward(self, x):
        return self.m(x)
