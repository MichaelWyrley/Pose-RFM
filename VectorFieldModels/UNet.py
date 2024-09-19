import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, redidual=True):
        super(ResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # changing input
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.GroupNorm(1, out_channels),
        )

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.residual = redidual

    def forward(self, x):
        if self.residual:
            if self.in_channels != self.out_channels:
                out = self.block(x) + self.skip(x)
            else:
                out = self.block(x) + x
        else:
            out = self.block(x)

        return F.gelu(out)
    
class SelfAttention(nn.Module):
    def __init__(self, channels, img_size, no_heads=8):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.img_size = img_size
        self.mha = nn.MultiheadAttention(channels, no_heads)
        self.norm = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.img_size * self.img_size).permute(0, 2, 1)
        x_ln = self.norm(x)

        attention_output, _ = self.mha(x_ln, x_ln, x_ln)

        x = x + attention_output
        x = self.ff_self(x) + x

        x = x.permute(0, 2, 1).view(-1, self.channels, self.img_size, self.img_size)
        return x

class DoubleResnet(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, kernel_size=3, stride=1, padding=1, bias=True):
        super(DoubleResnet, self).__init__()

        # Changing time embedding
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels,
            ),
        )

        self.rn1 = ResNetBlock(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.rn2 = ResNetBlock(out_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x, t):
        x = self.rn1(x)
        x = self.rn2(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        x = x + emb

        return x
    
class UNet_Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dimention, img_size, num_heads=8):
        super(UNet_Down, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        #self.resnet_block = DoubleResnet(in_channels, in_channels, emb_dimention)
        self.resnet_block2 = DoubleResnet(in_channels, out_channels, emb_dimention)
        self.self_attention = SelfAttention(out_channels, img_size, num_heads)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, t):
        #x = self.resnet_block(x, t)
        x = self.resnet_block2(x,t)
        x = self.self_attention(x)
        out = self.max_pool(x)
        return out, x

class UNet_Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dimention, img_size, num_heads=8):
        super(UNet_Up, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.resnet_block = DoubleResnet(in_channels, in_channels, emb_dimention)
        self.resnet_block2 = DoubleResnet(in_channels, out_channels, emb_dimention)
        self.self_attention = SelfAttention(out_channels, img_size, num_heads)

    def forward(self, x, x_previous, t):
        x = self.up(x)
        x = torch.cat([x, x_previous], dim=1)
        #x = self.resnet_block(x, t)
        x = self.resnet_block2(x, t)
        x = self.self_attention(x)
        return x

    

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_classes=10, img_size = 32, num_heads=8, emb_dimention=256, condition_prob = 0.25, device="cuda"):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.emb_dimention = emb_dimention
        self.device = device
        self.condition_prob = condition_prob

        
        self.down1 = UNet_Down(in_channels, 128, emb_dimention, img_size, num_heads)
        self.down2 = UNet_Down(128, 256, emb_dimention, img_size//2, num_heads)
        self.down3 = UNet_Down(256, 256, emb_dimention, img_size//4, num_heads)

        self.bottleneck1 = DoubleResnet(256, 512, emb_dimention)
        self.self_attention = SelfAttention(512, img_size//8, num_heads)
        self.bottleneck2 = DoubleResnet(512, 256, emb_dimention)

        self.up1 = UNet_Up(512, 256, emb_dimention, img_size//4, num_heads)
        self.up2 = UNet_Up(512, 128, emb_dimention, img_size//2, num_heads)
        self.up3 = UNet_Up(256, 128, emb_dimention, img_size, num_heads)

        self.out = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0)

        self.label_emb = nn.Embedding(num_classes, emb_dimention)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x, t, c):
        t = self.pos_encoding(t, self.emb_dimention)
        if (torch.rand(1).item() > self.condition_prob and self.train == True) and c is not None:
            c = self.label_emb(c)
            t = t + c

        x, x1 = self.down1(x, t)
        x, x2 = self.down2(x, t)
        x, x3 = self.down3(x, t)

        x = self.bottleneck1(x, t)
        x = self.self_attention(x)
        x = self.bottleneck2(x, t)

        x = self.up1(x, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)

        x = self.out(x)
        
        return x

        
        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # net = UNet(device="cpu")
    net = UNet(num_classes=10, device=device).to(device)

    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 32, 32).to(device)
    t = x.new_tensor([500] * x.shape[0]).unsqueeze(-1).long().to(device)
    y = x.new_tensor([1] * x.shape[0]).long().to(device)
    print(net(x, t, y).shape)
