# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

USE_ESA = True  
USE_CCA = True  
USE_BSCONV = True  

# 如果关闭 BSConv，就把 BSConv 替换成 nn.Conv2d
if not USE_BSCONV:
    class BSConvU(torch.nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
            padding_mode="zeros",
        ):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
                padding_mode="zeros",
            )

        def forward(self, fea):
            fea = self.conv(fea)
            return fea
else:
    # 点卷积+逐通道
    class BSConvU(torch.nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
            padding_mode="zeros",
            with_ln=False,
            bn_kwargs=None,
        ):
            super().__init__()
            self.with_ln = with_ln
            if bn_kwargs is None:
                bn_kwargs = {}

            # pointwise
            self.pw = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
            )

            # depthwise
            self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
            )

        def forward(self, fea):
            fea = self.pw(fea)
            fea = self.dw(fea)
            return fea

def kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1).mean()
    return kld

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def init_layer(layer, nonlinearity="leaky_relu"):
    """Initialize a Linear or Convolutional layer."""
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init_bn(bn):
    """Initialize a Batchnorm layer."""

    bn.bias.data.fill_(0.0)
    bn.running_mean.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_var.data.fill_(1.0)

class ESA(nn.Module):
    def __init__(self, num_feat, conv=BSConvU, p=0.25):
        super(ESA, self).__init__()

        f = num_feat // 4  # 8
        BSConvS_kwargs = {}

        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3_1 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_2 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = self.conv1(input)
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv3_1(v_max))
        c3_ = self.GELU(self.conv3_2(v_range))
        c3 = self.conv3_3(c3_)
        c3 = F.interpolate(
            c3, (input.size(2), input.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m

class ConvBlock_mix(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        conv=BSConvU,
    ):
        super(ConvBlock_mix, self).__init__()

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        self.conv2 = conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.esa = ESA(out_channels)

        self.init_weights()

    def init_weights(self):

        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type="max"):

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.esa(x)
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            return x
        return x

# 计算通道均值
def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

# 计算通道标准差
def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (
        F.size(2) * F.size(3)
    )
    return F_variance.pow(0.5)

# 通道注意力层
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SingleRNN(nn.Module):
    def __init__(
        self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False
    ):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.proj = nn.Linear(
            hidden_size * self.num_direction, input_size
        )  # 恢复feature_size的原始尺寸

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(output.shape)
        return rnn_output

class TFRNN(nn.Module):
    """
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """
    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        dropout=0,
        num_layers=1,
        bidirectional=False,
    ):
        super(TFRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])

        for i in range(num_layers):
            self.row_rnn.append(
                SingleRNN(
                    rnn_type, input_size, hidden_size, dropout, bidirectional=True
                )
            )  
            self.col_rnn.append(
                SingleRNN(
                    rnn_type,
                    input_size,
                    hidden_size,
                    dropout,
                    bidirectional=bidirectional,
                )
            )

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = (
                output.permute(0, 3, 2, 1)
                .contiguous()
                .view(batch_size * dim2, dim1, -1)
            )  
            row_output = self.row_rnn[i](row_input) 
            row_output = (
                row_output.view(batch_size, dim2, dim1, -1)
                .permute(0, 3, 2, 1)
                .contiguous()
            )  
            output = output + row_output

            col_input = (
                output.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size * dim1, dim2, -1)
            ) 
            col_output = self.col_rnn[i](col_input) 
            col_output = (
                col_output.view(batch_size, dim1, dim2, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )  
            output = output + col_output

        return output

# 论文模型
class proposed_model(nn.Module):
    def __init__(self, classes_num=2, activation="logsoftmax"):
        super(proposed_model, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock_mix(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock_mix(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock_mix(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock_mix(in_channels=256, out_channels=256)
        self.conv_block5 = ConvBlock_mix(in_channels=256, out_channels=256)

        self.TFrnn = TFRNN(rnn_type="LSTM", input_size=256, hidden_size=128)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, classes_num, bias=True)
        self.cca = CCALayer(256)  
        self.c5 = nn.Conv2d(256 * 3, 256, kernel_size=1)
        self.init_weights()

    def init_weights(self):

        init_layer(self.fc2)
        init_layer(self.fc)

    def forward(self, input):
        """
        Input: (batch_size, times_steps, freq_bins)
        """
        if input.dim() == 3:
            x = input.unsqueeze(1)
        else:
            x = input
        """(batch_size, 1, times_steps, freq_bins)"""
        x1 = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x2 = self.conv_block2(x1, pool_size=(2, 2), pool_type="avg")
        x3 = self.conv_block3(x2, pool_size=(2, 2), pool_type="avg")
        x4 = self.conv_block4(x3, pool_type=None)
        x5 = self.conv_block5(x4, pool_type=None)
        x = torch.cat((x3, x4, x5), dim=1)
        x = self.c5(x)
        x = self.cca(x)

        """(batch_size, feature_maps, time_steps, freq_bins)"""
        x = self.TFrnn(x)
        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = self.dropout(x)
        x = self.fc(x)

        if self.activation == "logsoftmax":
            output = F.log_softmax(x, dim=-1)

        elif self.activation == "sigmoid":
            output = torch.sigmoid(x)

        return output

if __name__ == "__main__":
    # 如果关闭 ESA，就把 ESA 替换成恒等层
    if not USE_ESA:
        class ESA(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

    # 如果关闭 CCA，就把 CCALayer 替换成恒等层
    if not USE_CCA:
        class CCALayer(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

    model = proposed_model().eval()
# %%
