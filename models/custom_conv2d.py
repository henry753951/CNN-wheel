import torch
import torch.nn as nn
import conv2d_cuda  # 導入編譯的擴展模組

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CustomConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        out_channels = self.weight.size(0)
        kernel_size = self.weight.size(2)

        output = []
        for b in range(batch_size):
            out_channels_list = []
            for o in range(out_channels):
                conv_sum = 0.0
                for i in range(in_channels):
                    input_slice = x[b, i].contiguous().to(device='cuda')
                    weight_slice = self.weight[o, i].contiguous().to(device='cuda')
                    conv_sum += conv2d_cuda.conv2d(input_slice, weight_slice)
                out_channels_list.append(conv_sum)
            output.append(torch.stack(out_channels_list, dim=0))
        output = torch.stack(output, dim=0)
        return output