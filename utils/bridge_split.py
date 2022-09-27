import torch
from torch import nn, einsum
from einops import rearrange



class resnet2transformer(nn.Module):
    def __init__(self, dim, heads, c, dropout=0.):
        super(resnet2transformer, self).__init__()
        inner_dim = c
        dim_head = c // heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = dim_head ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
   

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
       
        x = x.contiguous().view(b, h * w, c)
        x = rearrange(x, 'b (h i) c -> b h i c', h=self.heads)
        k, v = x, x
       
        q = self.to_q(z)
        q = rearrange(q, 'b (h j) c -> b h j c', h=self.heads)
        dots = einsum('b h j c, b h i c -> b h j i', q, k) * self.scale
    
        attn = self.attend(dots)
        out = einsum('b h j i, b h i c -> b h j c', attn, v)
        out = rearrange(out, 'b h j c -> b (h j) c')
        return z + self.to_out(out)



class transformer2resnet(nn.Module):
    def __init__(self, dim, heads, c, dropout=0.):
        super(transformer2resnet, self).__init__()
        inner_dim = c
        dim_head = c // heads
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = dim_head ** -0.5

        self.to_out = nn.Identity()

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        x_ = x.contiguous().view(b, h * w, c)
        x_ = rearrange(x_, 'b (h i) c -> b h i c', h=self.heads)
        q = x_

        k = self.to_k(z)
        v = self.to_v(z)
        k = rearrange(k, 'b (h j) c -> b h j c', h=self.heads)
        v = rearrange(v, 'b (h j) c -> b h j c', h=self.heads)

        dots = einsum('b h i c, b h j c -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        out = einsum('b h i j, b h j c -> b h i c', attn, v)
        out = rearrange(out, 'b h i c -> b (h i) c')
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out
