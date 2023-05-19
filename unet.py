import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DoubleConv(nn.Module):  # 两次卷积封装
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),   # kernel size = 3
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),   # inplace=true覆盖原变量，节省内存
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, t):
        return self.conv(t)


class TransUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TransUnet, self).__init__()

        # Transformer parameters
        self.d_model = 1024
        self.nhead = 8
        self.num_layers = 6
        self.dim_feedforward = 2048
        self.transformer_dropout = 0.1

        # down sampling
        # 假如输入 224*224*1 的图像
        # H = ((224 - 3 + 1 + 2 - 1) / 1) + 1 = 224  unet的卷积不会改变特征图的大小
        self.conv1 = DoubleConv(in_ch, 64)
        # to increase the dimensions
        self.w1 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, stride=1)
        self.pool1 = nn.MaxPool2d(2)  # 224 -> 112

        self.conv2 = DoubleConv(64, 128)  # 不变
        # to increase the dimensions
        self.w2 = nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=1)
        self.pool2 = nn.MaxPool2d(2)  # 56

        self.conv3 = DoubleConv(128, 256)
        # to increase the dimensions
        self.w3 = nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=1)
        self.pool3 = nn.MaxPool2d(2)  # 28

        self.conv4 = DoubleConv(256, 512)
        # to increase the dimensions
        self.w4 = nn.Conv2d(256, 512, kernel_size=1, padding=0, stride=1)
        self.pool4 = nn.MaxPool2d(2)  # 14

        self.conv5 = DoubleConv(512, 1024)
        # to increase the dimensions
        self.w5 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, stride=1)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                   dim_feedforward=self.dim_feedforward, dropout=self.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers)
        # Positional Encoding
        self.positional_encoder = PositionalEncoding(
            self.d_model, dropout=self.transformer_dropout)

        # up sampling
        # H_out = (14 - 1) * 2 + 2 = 28 往上反卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # to increase the dimensions
        self.w6 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, stride=1)
        self.conv6 = DoubleConv(1024, 512)   # 28

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # to increase the dimensions
        self.w7 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # to increase the dimensions
        self.w8 = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # to increase the dimensions
        self.w9 = nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

        # 训练时尝试让神经元失活，加大泛化性，仅在训练时使用,pytorch自动补偿参数
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 下采样部分
        down0_res = self.w1(x)  # residual block
        down0 = self.conv1(x) + down0_res
        down1 = self.pool1(down0)

        down1_res = self.w2(down1)  # residual block
        down1 = self.conv2(down1) + down1_res
        down2 = self.pool2(down1)

        down2_res = self.w3(down2)
        down2 = self.conv3(down2) + down2_res
        down3 = self.pool3(down2)

        down3_res = self.w4(down3)
        down3 = self.conv4(down3) + down3_res
        down4 = self.pool4(down3)

        down4_res = self.w5(down4)
        # 5 , 连接上采样部分前，双卷积卷积操作    [14, 14, 1024]
        down5 = self.conv5(down4) + down4_res

        # Transformer Encoder
        b, c, h, w = down5.size()
        down5_flat = down5.view(b, c, h * w).permute(2,
                                                     0, 1)  # Reshape to [h*w, b, c]
        down5_pos = self.positional_encoder(down5_flat)
        down5_transformed = self.transformer_encoder(down5_pos)
        down5_transformed = down5_transformed.permute(1, 2, 0).view(
            b, c, h, w)  # Reshape back to [b, c, h, w]

        # 上采样部分
        up_6 = self.up6(down5_transformed)   # [28, 28, 512]
        merge6 = torch.cat([up_6, down3], dim=1)    # cat之后又变为[28, 28, 1024]
        up_6_res = self.w6(merge6)
        c6 = self.conv6(merge6) + up_6_res   # 重新双卷积变为[28, 28, 512]

        up_7 = self.up7(c6)   # [56, 56, 256]
        merge7 = torch.cat([up_7, down2], dim=1)
        up_7_res = self.w7(merge7)
        c7 = self.conv7(merge7) + up_7_res  # [56, 56, 256]

        up_8 = self.up8(c7)   # [112, 112, 128]
        merge8 = torch.cat([up_8, down1], dim=1)
        up_8_res = self.w8(merge8)
        c8 = self.conv8(merge8) + up_8_res  # [112, 112, 128]

        up_9 = self.up9(c8)   # [224, 224, 64]
        merge9 = torch.cat([up_9, down0], dim=1)
        up_9_res = self.w9(merge9)
        c9 = self.conv9(merge9) + up_9_res  # [224, 224, 64]

        c10 = self.conv10(c9)  # 卷积输出最终图像   [224, 224, t]

        return c10


class Res2Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res2Unet, self).__init__()

        # down sampling
        # 假如输入 224*224*1 的图像
        # H = ((224 - 3 + 1 + 2 - 1) / 1) + 1 = 224  unet的卷积不会改变特征图的大小
        self.conv1 = DoubleConv(in_ch, 64)
        # to increase the dimensions
        self.w1 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, stride=1)
        self.pool1 = nn.MaxPool2d(2)  # 224 -> 112

        self.conv2 = DoubleConv(64, 128)  # 不变
        # to increase the dimensions
        self.w2 = nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=1)
        self.pool2 = nn.MaxPool2d(2)  # 56

        self.conv3 = DoubleConv(128, 256)
        # to increase the dimensions
        self.w3 = nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=1)
        self.pool3 = nn.MaxPool2d(2)  # 28

        self.conv4 = DoubleConv(256, 512)
        # to increase the dimensions
        self.w4 = nn.Conv2d(256, 512, kernel_size=1, padding=0, stride=1)
        self.pool4 = nn.MaxPool2d(2)  # 14

        self.conv5 = DoubleConv(512, 1024)
        # to increase the dimensions
        self.w5 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, stride=1)

        # up sampling
        # H_out = (14 - 1) * 2 + 2 = 28 往上反卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # to increase the dimensions
        self.w6 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, stride=1)
        self.conv6 = DoubleConv(1024, 512)   # 28

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # to increase the dimensions
        self.w7 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # to increase the dimensions
        self.w8 = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # to increase the dimensions
        self.w9 = nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

        # 训练时尝试让神经元失活，加大泛化性，仅在训练时使用,pytorch自动补偿参数
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 下采样部分
        down0_res = self.w1(x)  # residual block
        down0 = self.conv1(x) + down0_res
        down1 = self.pool1(down0)

        down1_res = self.w2(down1)  # residual block
        down1 = self.conv2(down1) + down1_res
        down2 = self.pool2(down1)

        down2_res = self.w3(down2)
        down2 = self.conv3(down2) + down2_res
        down3 = self.pool3(down2)

        down3_res = self.w4(down3)
        down3 = self.conv4(down3) + down3_res
        down4 = self.pool4(down3)

        down4_res = self.w5(down4)
        # 5 , 连接上采样部分前，双卷积卷积操作    [14, 14, 1024]
        down5 = self.conv5(down4) + down4_res

        # 上采样部分
        up_6 = self.up6(down5)   # [28, 28, 512]
        merge6 = torch.cat([up_6, down3], dim=1)    # cat之后又变为[28, 28, 1024]
        up_6_res = self.w6(merge6)
        c6 = self.conv6(merge6) + up_6_res   # 重新双卷积变为[28, 28, 512]

        up_7 = self.up7(c6)   # [56, 56, 256]
        merge7 = torch.cat([up_7, down2], dim=1)
        up_7_res = self.w7(merge7)
        c7 = self.conv7(merge7) + up_7_res  # [56, 56, 256]

        up_8 = self.up8(c7)   # [112, 112, 128]
        merge8 = torch.cat([up_8, down1], dim=1)
        up_8_res = self.w8(merge8)
        c8 = self.conv8(merge8) + up_8_res  # [112, 112, 128]

        up_9 = self.up9(c8)   # [224, 224, 64]
        merge9 = torch.cat([up_9, down0], dim=1)
        up_9_res = self.w9(merge9)
        c9 = self.conv9(merge9) + up_9_res  # [224, 224, 64]

        c10 = self.conv10(c9)  # 卷积输出最终图像   [224, 224, t]

        return c10
    

class Res1Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res1Unet, self).__init__()

        # down sampling
        # 假如输入 224*224*1 的图像
        # H = ((224 - 3 + 1 + 2 - 1) / 1) + 1 = 224  unet的卷积不会改变特征图的大小
        self.conv1 = DoubleConv(in_ch, 64)
        # to increase the dimensions
        self.w1 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, stride=1)
        self.pool1 = nn.MaxPool2d(2)  # 224 -> 112

        self.conv2 = DoubleConv(64, 128)  # 不变
        # to increase the dimensions
        self.w2 = nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=1)
        self.pool2 = nn.MaxPool2d(2)  # 56

        self.conv3 = DoubleConv(128, 256)
        # to increase the dimensions
        self.w3 = nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=1)
        self.pool3 = nn.MaxPool2d(2)  # 28

        self.conv4 = DoubleConv(256, 512)
        # to increase the dimensions
        self.w4 = nn.Conv2d(256, 512, kernel_size=1, padding=0, stride=1)
        self.pool4 = nn.MaxPool2d(2)  # 14

        self.conv5 = DoubleConv(512, 1024)
        # to increase the dimensions
        self.w5 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, stride=1)

        # up sampling
        # H_out = (14 - 1) * 2 + 2 = 28 往上反卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)   # 28

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

        # 训练时尝试让神经元失活，加大泛化性，仅在训练时使用,pytorch自动补偿参数
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 下采样部分
        down0_res = self.w1(x)  # residual block
        down0 = self.conv1(x) + down0_res
        down1 = self.pool1(down0)

        down1_res = self.w2(down1)  # residual block
        down1 = self.conv2(down1) + down1_res
        down2 = self.pool2(down1)

        down2_res = self.w3(down2)
        down2 = self.conv3(down2) + down2_res
        down3 = self.pool3(down2)

        down3_res = self.w4(down3)
        down3 = self.conv4(down3) + down3_res
        down4 = self.pool4(down3)

        down4_res = self.w5(down4)
        # 5 , 连接上采样部分前，双卷积卷积操作    [14, 14, 1024]
        down5 = self.conv5(down4) + down4_res

        # 上采样部分
        up_6 = self.up6(down5)   # [28, 28, 512]
        merge6 = torch.cat([up_6, down3], dim=1)    # cat之后又变为[28, 28, 1024]
        c6 = self.conv6(merge6)   # 重新双卷积变为[28, 28, 512]

        up_7 = self.up7(c6)   # [56, 56, 256]
        merge7 = torch.cat([up_7, down2], dim=1)
        c7 = self.conv7(merge7) # [56, 56, 256]

        up_8 = self.up8(c7)   # [112, 112, 128]
        merge8 = torch.cat([up_8, down1], dim=1)
        c8 = self.conv8(merge8) # [112, 112, 128]

        up_9 = self.up9(c8)   # [224, 224, 64]
        merge9 = torch.cat([up_9, down0], dim=1)
        c9 = self.conv9(merge9)  # [224, 224, 64]

        c10 = self.conv10(c9)  # 卷积输出最终图像   [224, 224, t]

        return c10


class TransRes1Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TransRes1Unet, self).__init__()
        # Transformer parameters
        self.d_model = 1024
        self.nhead = 8
        self.num_layers = 6
        self.dim_feedforward = 2048
        self.transformer_dropout = 0.1

        # down sampling
        # 假如输入 224*224*1 的图像
        # H = ((224 - 3 + 1 + 2 - 1) / 1) + 1 = 224  unet的卷积不会改变特征图的大小
        self.conv1 = DoubleConv(in_ch, 64)
        # to increase the dimensions
        self.w1 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, stride=1)
        self.pool1 = nn.MaxPool2d(2)  # 224 -> 112

        self.conv2 = DoubleConv(64, 128)  # 不变
        # to increase the dimensions
        self.w2 = nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=1)
        self.pool2 = nn.MaxPool2d(2)  # 56

        self.conv3 = DoubleConv(128, 256)
        # to increase the dimensions
        self.w3 = nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=1)
        self.pool3 = nn.MaxPool2d(2)  # 28

        self.conv4 = DoubleConv(256, 512)
        # to increase the dimensions
        self.w4 = nn.Conv2d(256, 512, kernel_size=1, padding=0, stride=1)
        self.pool4 = nn.MaxPool2d(2)  # 14

        self.conv5 = DoubleConv(512, 1024)
        # to increase the dimensions
        self.w5 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, stride=1)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                   dim_feedforward=self.dim_feedforward, dropout=self.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers)
        # Positional Encoding
        self.positional_encoder = PositionalEncoding(
            self.d_model, dropout=self.transformer_dropout)
        # up sampling
        # H_out = (14 - 1) * 2 + 2 = 28 往上反卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)   # 28

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

        # 训练时尝试让神经元失活，加大泛化性，仅在训练时使用,pytorch自动补偿参数
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 下采样部分
        down0_res = self.w1(x)  # residual block
        down0 = self.conv1(x) + down0_res
        down1 = self.pool1(down0)

        down1_res = self.w2(down1)  # residual block
        down1 = self.conv2(down1) + down1_res
        down2 = self.pool2(down1)

        down2_res = self.w3(down2)
        down2 = self.conv3(down2) + down2_res
        down3 = self.pool3(down2)

        down3_res = self.w4(down3)
        down3 = self.conv4(down3) + down3_res
        down4 = self.pool4(down3)

        down4_res = self.w5(down4)
        # 5 , 连接上采样部分前，双卷积卷积操作    [14, 14, 1024]
        down5 = self.conv5(down4) + down4_res

        # Transformer Encoder
        b, c, h, w = down5.size()
        down5_flat = down5.view(b, c, h * w).permute(2,
                                                     0, 1)  # Reshape to [h*w, b, c]
        down5_pos = self.positional_encoder(down5_flat)
        down5_transformed = self.transformer_encoder(down5_pos)
        down5_transformed = down5_transformed.permute(1, 2, 0).view(
            b, c, h, w)  # Reshape back to [b, c, h, w]
        
        # 上采样部分
        up_6 = self.up6(down5)   # [28, 28, 512]
        merge6 = torch.cat([up_6, down3], dim=1)    # cat之后又变为[28, 28, 1024]
        c6 = self.conv6(merge6)   # 重新双卷积变为[28, 28, 512]

        up_7 = self.up7(c6)   # [56, 56, 256]
        merge7 = torch.cat([up_7, down2], dim=1)
        c7 = self.conv7(merge7) # [56, 56, 256]

        up_8 = self.up8(c7)   # [112, 112, 128]
        merge8 = torch.cat([up_8, down1], dim=1)
        c8 = self.conv8(merge8) # [112, 112, 128]

        up_9 = self.up9(c8)   # [224, 224, 64]
        merge9 = torch.cat([up_9, down0], dim=1)
        c9 = self.conv9(merge9)  # [224, 224, 64]

        c10 = self.conv10(c9)  # 卷积输出最终图像   [224, 224, t]

        return c10
    

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        # down sampling
        # 假如输入 224*224*1 的图像
        # H = ((224 - 3 + 1 + 2 - 1) / 1) + 1 = 224  unet的卷积不会改变特征图的大小
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)  # 224 -> 112

        self.conv2 = DoubleConv(64, 128)  # 不变
        self.pool2 = nn.MaxPool2d(2)  # 56

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)  # 28

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)  # 14

        self.conv5 = DoubleConv(512, 1024)

        # up sampling
        # H_out = (14 - 1) * 2 + 2 = 28 往上反卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)   # 28

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

        # 训练时尝试让神经元失活，加大泛化性，仅在训练时使用,pytorch自动补偿参数
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 下采样部分
        down0 = self.conv1(x)
        down1 = self.pool1(down0)

        down1 = self.conv2(down1)
        down2 = self.pool2(down1)

        down2 = self.conv3(down2)
        down3 = self.pool3(down2)

        down3 = self.conv4(down3)
        down4 = self.pool4(down3)

        # 5 , 连接上采样部分前，双卷积卷积操作    [14, 14, 1024]
        down5 = self.conv5(down4)

        # 上采样部分
        up_6 = self.up6(down5)   # [28, 28, 512]
        merge6 = torch.cat([up_6, down3], dim=1)    # cat之后又变为[28, 28, 1024]
        c6 = self.conv6(merge6)   # 重新双卷积变为[28, 28, 512]

        up_7 = self.up7(c6)   # [56, 56, 256]
        merge7 = torch.cat([up_7, down2], dim=1)
        c7 = self.conv7(merge7) # [56, 56, 256]

        up_8 = self.up8(c7)   # [112, 112, 128]
        merge8 = torch.cat([up_8, down1], dim=1)
        c8 = self.conv8(merge8) # [112, 112, 128]

        up_9 = self.up9(c8)   # [224, 224, 64]
        merge9 = torch.cat([up_9, down0], dim=1)
        c9 = self.conv9(merge9)  # [224, 224, 64]

        c10 = self.conv10(c9)  # 卷积输出最终图像   [224, 224, t]

        return c10
    