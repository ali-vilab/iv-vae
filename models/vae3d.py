import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
__all__ = [ 
    'IV_VAE', 
] 

CACHE_T = 1

class Group_Causal_Conv3d(nn.Conv3d): 
    def __init__(self, in_dim, out_dim, kernel, stride=1, padding=1, t_c=1):
        super().__init__(in_dim, out_dim, kernel, stride=stride, padding=padding)
        self._padding = (self.padding[2],self.padding[2],self.padding[1],self.padding[1],self.padding[0],self.padding[0])
        self._padding1 = (self.padding[2],self.padding[2],self.padding[1],self.padding[1],0,0)
        self._padding2 = (0,0,0,0,0,self.padding[0])
        self.padding = (0, 0, 0) 
        self.t_c = t_c

    def forward(self, x, cache_x=None):   
        if self._padding[4] == 0:  
            return super().forward(F.pad(x, self._padding))
        else:
            assert self._padding[4] == 1
            b, c, t, h, w = x.shape 
            
            if  t == 1: 
                x = F.pad(x, self._padding1)
                x = torch.cat([x,x], 2)
                x = F.pad(x, self._padding2)                
                x = super().forward(x)  
                return x             
            else: 
                t_c = self.t_c   
                b_t = t // t_c 
                x = rearrange(x, 'b c (b_t t_c) h w -> (b b_t) c t_c h w', t_c=t_c)
                x = F.pad(x, self._padding)
                x = rearrange(x, '(b b_t) c t_c h w -> b b_t c t_c h w', b_t=b_t)
                for i in range(b_t):
                    if i != 0: 
                        x[:,i,:,:1,:,:] = x[:,i-1,:,-2:-1,:,:]
                    else:
                        assert cache_x is not None
                        cache_x = cache_x.to(x.device)   
                        x[:,i,:,:1,:,:] = F.pad(cache_x, self._padding1)
                x = rearrange(x, 'b b_t c t_c h w -> (b b_t) c t_c h w')
                x = super().forward(x)  
                x = rearrange(x, '(b b_t) c t_c h w -> b c (b_t t_c) h w', b_t=b_t)  
                return x

class First_Sep_Conv3d(nn.Module): 
    def __init__(self, in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, kernel, stride=1, padding=0, t_c=1):
        super().__init__() 
        self.in_dim_2d = in_dim_2d 
        self.conv_2d = nn.Conv2d(in_dim_2d, out_dim_2d, kernel, stride, padding)
        self.conv_3d = Group_Causal_Conv3d(in_dim_2d, out_dim_3d, kernel, stride, padding, t_c)

    def forward(self, x, cache_x=None):
        b, c, t, h, w = x.size()
        x_2d = rearrange(x, 'b c t h w -> (b t) c h w')
        x_2d = self.conv_2d(x_2d)
        x_2d = rearrange(x_2d, '(b t) c h w-> b c t h w',b=b) 
        x = self.conv_3d(x, cache_x)
        return torch.cat([x_2d, x], dim=1)

class Sep_Conv3d(nn.Module): 
    def __init__(self, in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, kernel, stride=1, padding=0, t_c=1):
        super().__init__()
        self.in_dim_2d = in_dim_2d

        self.conv_2d = nn.Conv2d(in_dim_2d+in_dim_3d, out_dim_2d, kernel, stride, padding)
        self.conv_3d = Group_Causal_Conv3d(in_dim_2d+in_dim_3d, out_dim_3d, kernel, stride, padding, t_c)

    def forward(self, x, cache_x=None):
        b, c, t, h, w = x.size() 
        x_2d = rearrange(x, 'b c t h w -> (b t) c h w')
        x_2d = self.conv_2d(x_2d)
        x_2d = rearrange(x_2d, '(b t) c h w-> b c t h w',b=b) 
        x = self.conv_3d(x, cache_x)  
        return torch.cat([x_2d, x], dim=1)

class Sep_Conv2d(nn.Module): 
    def __init__(self, in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, kernel, stride=1, padding=0):
        super().__init__()
        self.in_dim_2d = in_dim_2d

        self.conv = nn.Conv2d(in_dim_2d+in_dim_3d, out_dim_2d+out_dim_3d, kernel, stride, padding) 

    def forward(self, x, cache_x=None):
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w') 
        x = self.conv(x) 
        x = rearrange(x, '(b t) c h w-> b c t h w',b=b) 
        return x

class PAC(nn.Module):
    def __init__(self, in_dim_2d, out_dim_2d):
        super().__init__()
        self.PAC_0 = Sep_Conv2d(in_dim_2d, out_dim_2d, in_dim_2d, out_dim_2d, 1, padding=0)
        self.PAC_1 = Sep_Conv2d(in_dim_2d, out_dim_2d, in_dim_2d, out_dim_2d, 3, padding=1)
        self.PAC_3 = Sep_Conv2d_dilated(in_dim_2d, out_dim_2d, in_dim_2d, out_dim_2d, 3, padding=3, dilation=3)
        self.PAC_6 = Sep_Conv2d_dilated(in_dim_2d, out_dim_2d, in_dim_2d, out_dim_2d, 3, padding=6, dilation=6)
        self.PAC_9 = Sep_Conv2d_dilated(in_dim_2d, out_dim_2d, in_dim_2d, out_dim_2d, 3, padding=9, dilation=9)
        self.PAC_12 = Sep_Conv2d_dilated(in_dim_2d, out_dim_2d, in_dim_2d, out_dim_2d, 3, padding=12, dilation=12)
        self.PAC_conv = Sep_Conv2d(out_dim_2d*6, out_dim_2d, out_dim_2d*6, out_dim_2d, 1)
        self.out_dim = out_dim_2d
    def forward(self, x):
        out_dim = self.out_dim
        x_0 = self.PAC_0(x)
        x_1 = self.PAC_1(x)
        x_3 = self.PAC_3(x)
        x_6 = self.PAC_6(x)
        x_9 = self.PAC_9(x)
        x_12 = self.PAC_12(x)
        x = torch.cat([x_0[:,:out_dim],x_1[:,:out_dim],x_3[:,:out_dim],x_6[:,:out_dim], \
            x_9[:,:out_dim], x_12[:,:out_dim], x_0[:,out_dim:], x_1[:,out_dim:],x_3[:,out_dim:], \
            x_6[:,out_dim:], x_9[:,out_dim:], x_12[:,out_dim:]], 1) 
        return self.PAC_conv(x)

class Sep_Conv2d_dilated(nn.Module): 
    def __init__(self, in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, kernel, stride=1, padding=0, dilation=0):
        super().__init__() 
        self.conv = nn.Conv2d(in_dim_2d+in_dim_3d, out_dim_2d+out_dim_3d, kernel, dilation=dilation, padding=dilation) 

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w') 
        x = self.conv(x) 
        x = rearrange(x, '(b t) c h w-> b c t h w',b=b) 
        return x
    
class RMS_norm(nn.Module): 
    def __init__(
        self,
        dim,
        channel_first = True,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

class Sep_RMS_norm(nn.Module): 
    def __init__(
        self,
        dim_2d,
        dim_3d, 
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) 
        assert dim_2d == dim_3d
        shape = (dim_2d+dim_3d, *broadcastable_dims)   
        self.dim = dim_2d   
        self.scale = dim_2d ** 0.5 
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias= nn.Parameter(torch.zeros(shape)) if bias else 0.
        
    def forward(self, x): 
        x = rearrange(x, 'b (r c)  t h w -> b r c t h w', r=2)
        x = F.normalize(x, dim = 2)
        x = rearrange(x, 'b r c t h w -> b (r c)  t h w')    
        return x * self.scale * self.gamma + self.bias
    
class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x)


class Resample(nn.Module):

    def __init__(self, dim, mode, t_c):
        assert mode in (
            'none',
            'upsample2d',
            'upsample3d',
            'downsample2d',
            'downsample3d'
        )
        super().__init__()
        self.dim = dim
        self.mode = mode 
        self.t_c = t_c
        # layers
        if mode == 'upsample2d':
            self.resample_2d = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim//2, 3, padding=1)
            )
            self.resample_3d = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim//2, 3, padding=1)
            )
        elif mode == 'upsample3d':
            self.resample_2d = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim//2, 3, padding=1)
            )
            self.resample_3d = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim//2, 3, padding=1)
            )
            self.time_conv = nn.Conv3d(dim, dim*2, (1,1,1), padding=(0,0,0))
        elif mode == 'downsample2d':
            self.resample_2d = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.resample_3d = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == 'downsample3d':
            self.resample_2d = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.resample_3d = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = nn.Conv3d(dim, dim, (2,1,1), stride=(2, 1, 1), padding=(0,0,0))
        else:
            self.resample = nn.Identity()
    
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache!=None and feat_cache[feat_idx[0]]!=None:   
                t = x.shape[2]
                x_left = x[:,:c//2,:,:,:]   
                x_right = x[:,c//2:,:,:,]    
                x_right = self.time_conv(x_right) 
                x_right = x_right.reshape(b,2,c//2,t,h,w)
                x_right = torch.stack((x_right[:,0,:,:,:,:],x_right[:,1,:,:,:,:]),3)
                x_right = x_right.reshape(b,c//2,t*2,h,w)  
                x_left = torch.stack((x_left[:,:,:,:,:],x_left[:,:,:,:,:]),3)
                x_left = x_left.reshape(b,c//2,t*2,h,w)  
                x = torch.cat([x_left,x_right],1)  
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        
        x = torch.cat([self.resample_2d(x[:,:self.dim,:,:]), self.resample_3d(x[:,self.dim:,:,:])],1)

        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)         
        if self.mode == 'downsample3d':  
            if t != 1:   
                select_list = [i*2 for i in range(t//2)]
                if self.t_c == 4:                
                    x = torch.cat([x[:,:self.dim,select_list,:,:], self.time_conv(x[:,self.dim:,:,:,:])], 1)
                else:
                    x = torch.cat([x[:,:self.dim,select_list,:,:], self.time_conv(x[:,self.dim:,:,:,:])], 1)
        return x
    
class Sep_ResidualBlock(nn.Module): 
    def __init__(self, in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, dropout=0.0, t_c=1):
        super().__init__() 
        self.in_dim_2d = in_dim_2d
        self.in_dim_3d = in_dim_3d
        # layers
        self.residual = nn.Sequential(
            Sep_RMS_norm(in_dim_2d, in_dim_3d),
            nn.SiLU(),
            Sep_Conv3d(in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, 3, padding=1,t_c=t_c),
            Sep_RMS_norm(out_dim_2d, out_dim_3d),
            nn.SiLU(),
            nn.Dropout(dropout),
            Sep_Conv3d(out_dim_2d, out_dim_2d, out_dim_3d, out_dim_3d, 3, padding=1,t_c=t_c)
        )
        self.shortcut = Sep_Conv3d(in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, 1,t_c=t_c) \
            if in_dim_2d != out_dim_2d else nn.Identity()
    
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, Sep_Conv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone() 
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h

class Sep_ResidualBlock_2d(nn.Module):

    def __init__(self, in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, dropout=0.0):
        super().__init__() 
        self.in_dim_2d = in_dim_2d
        self.in_dim_3d = in_dim_3d
        # layers
        self.residual = nn.Sequential(
            Sep_RMS_norm(in_dim_2d, in_dim_3d),
            nn.SiLU(),
            Sep_Conv2d(in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, 3, padding=1),
            Sep_RMS_norm(out_dim_2d, out_dim_3d),
            nn.SiLU(),
            nn.Dropout(dropout),
            Sep_Conv2d(out_dim_2d, out_dim_2d, out_dim_3d, out_dim_3d, 3, padding=1)
        )
        self.shortcut = Sep_Conv2d(in_dim_2d, out_dim_2d, in_dim_3d, out_dim_3d, 1) \
            if in_dim_2d != out_dim_2d else nn.Identity()
    
    def forward(self, x, feat_cache=None, feat_idx=[0]): 
        return self.residual(x) + self.shortcut(x)

class Sep_AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """
    def __init__(self, dim_2d, dim_3d):
        super().__init__()
        self.dim_2d = dim_2d
        self.dim_3d = dim_3d

        self.Attention_2d = AttentionBlock(dim_2d)
        self.Attention_3d = AttentionBlock(dim_3d) 
    
    def forward(self, x, feat_cache=None, feat_idx=[0]): 
        assert x.shape[1] == self.dim_2d + self.dim_3d
        return torch.cat([self.Attention_2d(x[:,:self.dim_2d,:,:,:]),self.Attention_3d(x[:,self.dim_2d:,:,:,:])], 1)

class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1) 
    
    def forward(self, x): 
        identity = x
        b, c, t, h, w = x.size()
        
        x = rearrange(x, 'b c t h w -> (b t) c h w')    
        x = self.norm(x)     
        # compute query, key, value        
        q, k, v = self.to_qkv(x).reshape(
            b*t, 1, c * 3, -1
        ).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            #attn_mask=block_causal_mask(q, block_size=h * w)
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b*t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w',t=t) 
        return x + identity


class Encoder3d(nn.Module):

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
 
        self.conv1 = First_Sep_Conv3d(3, dims[0], 3, dims[0], 3, padding=1, t_c=4)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])): 
            for _ in range(num_res_blocks): 
                if i != len(dim_mult) - 1:
                    if i == len(dim_mult) - 2:
                        downsamples.append(Sep_ResidualBlock(in_dim, out_dim, in_dim, out_dim, dropout, t_c=2))
                    else:
                        downsamples.append(Sep_ResidualBlock(in_dim, out_dim, in_dim, out_dim, dropout, t_c=4))
                else:
                    downsamples.append(Sep_ResidualBlock_2d(in_dim, out_dim, in_dim, out_dim, dropout)) # 2D
                if scale in attn_scales: 
                    downsamples.append(Sep_AttentionBlock(out_dim, out_dim))
                if i == len(dim_mult) - 1:
                    downsamples.append(Sep_AttentionBlock(out_dim, out_dim))
                in_dim = out_dim
            
            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d' 
                if i == len(dim_mult) - 2:
                    downsamples.append(Resample(out_dim, mode=mode,t_c=2)) 
                else:
                    downsamples.append(Resample(out_dim, mode=mode,t_c=4)) 
                scale /= 2.0
            if i == len(dim_mult) - 2:
                downsamples.append(Sep_AttentionBlock(out_dim, out_dim)) 
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks 
        self.middle = nn.Sequential(
            Sep_ResidualBlock_2d(out_dim, out_dim, out_dim, out_dim, dropout),
            Sep_AttentionBlock(out_dim, out_dim),
            Sep_ResidualBlock_2d(out_dim, out_dim, out_dim, out_dim, dropout)
        )

        # output blocks 
        self.head = nn.Sequential(
            Sep_RMS_norm(out_dim, out_dim),
            nn.SiLU(),
            PAC(out_dim, z_dim)
        )
        
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone() 
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1 
        else:
            x = self.conv1(x) 

        ## downsamples 
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)  
            else:
                x = layer(x) 

        ## middle 
        x = self.middle(x) 
        ## head 
        x = self.head(x)
        return x


class Decoder3d(nn.Module):

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block 
        self.conv1 = Sep_Conv2d(z_dim, dims[0], z_dim, dims[0], 3, padding=1)

        # middle blocks 
        self.middle = nn.Sequential(
            Sep_ResidualBlock_2d(dims[0], dims[0], dims[0], dims[0], dropout),
            Sep_AttentionBlock(dims[0], dims[0]),
            Sep_ResidualBlock_2d(dims[0], dims[0], dims[0], dims[0], dropout)
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim //2
            for _ in range(num_res_blocks + 1): 
                if i != 0:
                    if i == 1:
                        upsamples.append(Sep_ResidualBlock(in_dim, out_dim, in_dim, out_dim, dropout, t_c=2))
                    else:
                        upsamples.append(Sep_ResidualBlock(in_dim, out_dim, in_dim, out_dim, dropout, t_c=4))
                else:
                    upsamples.append(Sep_ResidualBlock_2d(in_dim, out_dim, in_dim, out_dim, dropout)) # 2D
                if scale in attn_scales: 
                    upsamples.append(Sep_AttentionBlock(out_dim, out_dim))
                if i == 0 :
                    upsamples.append(Sep_AttentionBlock(out_dim, out_dim))
                in_dim = out_dim
            
            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                if i == 0:
                    upsamples.append(Resample(out_dim, mode=mode, t_c=1))
                else:
                    upsamples.append(Resample(out_dim, mode=mode, t_c=2))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks 
        self.head = nn.Sequential(
            Sep_RMS_norm(out_dim, out_dim),
            nn.SiLU(),
            Sep_Conv3d(out_dim, 3, out_dim, 3, 3, padding=1, t_c=4)
        )
    
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.conv1(x) 
        x = self.middle(x)
        ## upsamples 
        for layer in self.upsamples:
            if feat_cache is not None: 
                x = layer(x, feat_cache, feat_idx)  
            else: 
                x = layer(x)  
        ## head 
        for layer in self.head:
            if isinstance(layer, Sep_Conv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()                
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1 
            else:
                x = layer(x)  
        return x

def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, Sep_Conv3d): 
            count += 1
    return count

class VAutoencoder3d(nn.Module):

    def __init__(
        self,
        dim=64,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        scale_factor=1,
        shift_factor=0,
    ):
        super().__init__()
        self.dim = dim 
        z_dim = z_dim // 2 
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        
        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales,
            self.temperal_downsample, dropout
        ) 
        self.conv1 = Sep_Conv3d(z_dim * 2, z_dim * 2, z_dim * 2, z_dim * 2, 1, t_c=1)
        self.conv2 = Sep_Conv3d(z_dim, z_dim, z_dim, z_dim, 1, t_c=1)
        self.decoder = Decoder3d(
            dim, z_dim, dim_mult, num_res_blocks, attn_scales,
            self.temperal_upsample, dropout
        )
        
        # shift & scale
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor 
        
    def forward(self, x):
        b = x.shape[0]  
        z = self.encode(x) 
        x_recon = self.decode(z)    
        return x_recon
    
    def encode(self, x):     
        self.clear_cache() 
        t = x.shape[2]
        iter_ = 1 + (t-1)//4  ## clip the video to 1+4*N....
        
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:,:,:1,:,:], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:,:,1+4*(i-1):1+4*i,:,:], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)   
        x = self.conv1(out) 
        x = torch.cat([x[:,:self.z_dim,:,:,:], x[:,self.z_dim*2:self.z_dim*3,:,:,:], 
                    x[:,self.z_dim:self.z_dim*2,:,:,:], x[:,self.z_dim*3:,:,:,:]],1) 
        mu, log_var = x.chunk(2, dim=1)
        latent = self.sample(mu, log_var) 
        latent = self.scale_factor * (latent - self.shift_factor)
        return latent
    
    def decode(self, z):
        self.clear_cache() 
        z = z / self.scale_factor + self.shift_factor
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:,:,i:i+1,:,:], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:,:,i:i+1,:,:], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2) 
        
        ## output
        key_frame, res = out.chunk(2, dim=1)
        x_first = key_frame[:,:,:1,:,:]
        key_frame, res = key_frame[:,:,1:,:,:], res[:,:,1:,:,:]
        
        b, c, t, h, w = x.size()
        res = rearrange(res, 'b c (t t_c) h w -> (b t) c t_c h w', t_c=4)   
        key_frame = rearrange(key_frame, 'b c (t t_c) h w -> (b t) c t_c h w', t_c=4)   
        x_recon = torch.cat([key_frame[:,:,:1,:,:], res[:,:,1:,:,:]],2)
        x_recon = rearrange(x_recon, '(b t) c t_c h w -> b c (t t_c) h w',b=b)  
        x_recon = torch.cat([x_first, x_recon], 2)
        return x_recon   
    
    def sample(self, mu, log_var, deterministic=False): 
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num 
        self._enc_conv_num = count_conv3d(self.encoder)  
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def IV_VAE(z_dim, dim):
    scale_factor_list = {'ivvae_z4_dim64': 1.3196, 'ivvae_z8_dim64': 1.0940, 'ivvae_z16_dim64': 0.9697, 'ivvae_z16_dim96': 0.7619}
    shift_factor_list = {'ivvae_z4_dim64': 0.1250, 'ivvae_z8_dim64': -0.0322, 'ivvae_z16_dim64': -0.1553, 'ivvae_z16_dim96': -0.0483}
    
    pretrained_model = 'ivvae_z' + str(z_dim) + '_dim' + str(dim)
    pretrained_path = 'weights/' + pretrained_model + ".pth"
    print('pretrained_model:', pretrained_model)
    # params
    cfg = dict(
        z_dim=16,
        dim=96, 
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        scale_factor=scale_factor_list[pretrained_model],
        shift_factor=shift_factor_list[pretrained_model]
    )
    cfg.update(z_dim=z_dim, dim=dim) 
    model = VAutoencoder3d(**cfg) 
    state = torch.load( 
        pretrained_path, 
        map_location='cpu'
    )
    model.load_state_dict(state, strict=True) 
    del state
    return model

 
