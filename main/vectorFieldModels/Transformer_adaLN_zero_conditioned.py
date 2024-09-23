# MODIFIED FROM https://github.com/facebookresearch/DiT/blob/main/models.py
import sys
sys.path.append('')
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from main.vectorFieldModels.Transformer_adaLN_zero import modulate, PointFF, DiTBlock, PatchEmbedding, TimestepEmbedder

class DiT_adaLN_zero(nn.Module):
    def __init__(self, in_dim=6, depth=12, emb_dimention=768, num_heads=12, num_classes=10, device='cuda', condition_prob=0.1):
        super(DiT_adaLN_zero, self).__init__()
        self.num_heads = num_heads
        self.emb_dimention = emb_dimention
        self.device = device

        self.blocks = nn.ModuleList([DiTBlock(emb_dimention, num_heads) for _ in range(depth)])

        self.condition_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dimention, 2 * emb_dimention),
        )
        self.layernorm = nn.LayerNorm(emb_dimention, elementwise_affine=False, eps=1e-6)
        self.fc = nn.Linear(emb_dimention, in_dim )


        self.label_emb = nn.Embedding(num_classes, emb_dimention)
        self.patch_embedding = PatchEmbedding(in_dim, emb_dimention)
        self.time_embedding = TimestepEmbedder(emb_dimention)

        self.initialise_weights()
    
    def initialise_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedding.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.condition_mlp[-1].weight, 0)
        nn.init.constant_(self.condition_mlp[-1].bias, 0)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, t, c=None):


        # Encode time and condition
        t = self.time_embedding(t)
        if c is not None:
            c = self.label_emb(c)
            t = t + c
        # Patchify and encode image
        x = self.patch_embedding(x) 

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t)

        
        # Use adaptive Layer Normalization
        shift, scale = self.condition_mlp(t).chunk(2, dim=1)
        # Reshape x to the original image shape
        x = self.layernorm(x)
        x = modulate(x, shift, scale)

        x = self.fc(x)
        
        return x

        
        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # net = UNet(device="cpu")
    net = DiT_adaLN_zero(num_classes=10, device=device).to(device)

    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 21, 6).to(device)
    t = x.new_tensor([0.02] * x.shape[0]).unsqueeze(-1).to(device)
    # y = x.new_tensor([1] * x.shape[0]).long().to(device)
    print(net(x, t).shape)
