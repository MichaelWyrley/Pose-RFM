import torch
import torch.nn as nn
import torch.nn.functional as F

class PointFF(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0):
        super(PointFF, self).__init__()

        self.activation = nn.GELU(approximate='tanh')

        self.w_1 = nn.Linear(in_features, hidden_features)
        self.w_2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        z = self.w_1(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.w_2(z)

        return z


class TransformerBlock(nn.Module):
    def __init__(self, channel, no_heads=8, mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()
        self.channel = channel
        self.no_heads = no_heads

        self.mha = nn.MultiheadAttention(channel, no_heads)
        self.norm1 = nn.LayerNorm([channel])

        mlp_hidden_dim = int(channel * mlp_ratio)
        self.mlp = PointFF(in_features=channel, hidden_features=mlp_hidden_dim, drop=0)

        self.norm2 = nn.LayerNorm([channel])



    def forward(self, x):
        
        attention_output, _ = self.mha(x,x,x)

        x = x + attention_output
        x = self.norm1(x)

        z = self.mlp(x) 
        x = z + x

        return x
    

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_size, emb_dimention):
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channels
        self.emb_dimention = emb_dimention

        self.fc = nn.Linear(out_size, emb_dimention)
        self.ln = nn.LayerNorm([emb_dimention])

    def forward(self, x):
        x = self.fc(x)
        
        x = self.ln(x)
        return x

class DiT(nn.Module):
    def __init__(self, channel=21, in_dim=6, depth=12, emb_dimention=768, num_heads=12, num_classes=10, device='cuda', condition_prob=0.1):
        super(DiT, self).__init__()
        self.channel = channel
        self.num_heads = num_heads
        self.emb_dimention = emb_dimention
        self.device = device

        self.block = nn.ModuleList([TransformerBlock(emb_dimention, num_heads) for _ in range(depth)])

        self.label_emb = nn.Embedding(num_classes, emb_dimention)
        self.patch_embedding = PatchEmbedding(channel, in_dim, emb_dimention)

        self.fc = nn.Linear(emb_dimention, in_dim )

        self.time_embedding = nn.Linear(1, self.emb_dimention)
        


    # def time_encoding(self, t, channels):
    #     inv_freq = 1.0 / (
    #         10000
    #         ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
    #     )
    #     pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    #     pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    #     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    #     return pos_enc
    

    def position_encoding(self, x):
        b, n, _ = x.shape
        pos = torch.arange(n, device=x.device).float()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.emb_dimention, 2, device=x.device).float() / self.emb_dimention))
        pos = pos[:, None] * inv_freq[None, :]
        pos_enc = torch.cat([pos.sin(), pos.cos()], dim=-1)[None, :, :].expand(b, -1, -1)
        return pos_enc

    def forward(self, x, t, c=None):


        # Encode time and condition
        t = self.time_embedding(t)
        if c is not None:
            c = self.label_emb(c)
            t = t + c
        # Patchify and encode image
        x = self.patch_embedding(x) 
        x += self.position_encoding(x)

        # add time_embedding to the tokens
        x = torch.cat((t.unsqueeze(1), x), dim=1)
 
        # Apply transformer blocks
        for block in self.block:
            x = block(x)

        # Reshape x to the original pose shape
        x = self.fc(x)[:, 1:]

        # # Reproject onto quaternion
        # x = F.normalize(x, dim=2)

        # # force w to be poseitive
        # w_neg = x[..., 0] < 0
        # x[w_neg] = -x[w_neg]
        
        return x

        
        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # net = UNet(device="cpu")
    net = DiT(num_classes=10, device=device).to(device)

    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 21, 6).to(device)
    t = x.new_tensor([0.02] * x.shape[0]).unsqueeze(-1).to(device)
    # y = x.new_tensor([1] * x.shape[0]).long().to(device)
    print(net(x, t).shape)
