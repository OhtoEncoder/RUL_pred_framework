import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride = stride,
                                padding = 0)
    
    def forward(self, x):
        #time series 양쪽 끝 단에 패딩 추가
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1)//2, 1)    #텐서의 첫 차원 반복안함, 두번째차원은 반복함, 세번째차원 반복안함
        end = x[:, -1:, :].repeat(1, (self.kernel_size-1)//2, 1)        #텐서의 첫 차원 반복안함, 두번째차원은 반복함, 세번째차원 반복안함
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0,2,1))
        x = x.permute(0,2,1)
        return x