import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

def window_partition(input, window_size: int):
    """
    TODO: 将feature map划分为没有重叠的window
    """
    B, H, W, C = input.shape
    input = input.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = input.permute(0, 1, 3, 2, 4, 5).contiguous() # [B, H//Mh, Mh, W//Mw, Mw, C] => [B, H//Mh, W//Mw, Mh, Mw, C]
    windows = windows.view(-1, window_size, window_size, C) # [B, H//Mh, W//Mw, Mh, Mw, C] => [B * num_windows, Mh, Mw, C]
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将Window还原成feature map
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    input = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1) # [B, H//Mh, W//Mw, Mh, Mw, C]
    input = input.permute(0, 1, 3, 2, 4, 5).contiguous() # [B, H//Mh, W//Mw, Mh, Mw, C] => [B, H//Mh, Mh, W//Mw, Mw, C]
    input = input.view(B, H, W, -1) # [B, H//Mh, Mh, W//Mw, Mw, C] => [B, H, W, C]
    return input

def drop_path_f(x, drop_prob: float = 0, training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, input):
        return drop_path_f(input, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size = 4, in_channel = 3, embed_dim = 96, norm_layer = None):
        super(PatchEmbed, self).__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, input):
        _, _, H, W = input.shape

        # padding 如果输入的H, W不是patch_size的整数倍需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            input = F.pad(input, (0, self.patch_size[1] - W % self.patch_size[1],
                                  0, self.patch_size[0] - H % self.patch_size[0],
                                  0, 0))
        input = self.proj(input)
        _, _, H, W = input.shape
        input = input.flatten(2).transpose(1, 2)
        input = self.norm(input)
        return input, H, W

class PatchMerging(nn.Module): # 使用的是 2 * 2的窗口
    def __init__(self, dim, norm_layer = nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias = False)
        self.norm = norm_layer(4 * dim)

    def forward(self, input, H, W):
        """
        input: B, HW, C
        """
        B, L, C = input.shape
        assert L == H * W, "input features has wrong size"

        input = input.view(B, H, W, C)

        # padding 如果输入的H, W不是2的整数倍
        pad_input = (H % 2 == 0) or (W % 2 == 0)
        if pad_input:
            input = F.pad(input, (0, 0, 0, W % 2, 0, H % 2))

        # [B, H/2, W/2, C]
        input0 = input[:, 0::2, 0::2, :]
        input1 = input[:, 1::2, 0::2, :]
        input2 = input[:, 0::2, 1::2, :]
        input3 = input[:, 1::2, 1::2, :]

        input = torch.cat([input0, input1, input2, input3], -1) # [B, H/2, W/2, 4 * C]
        input = input.view(B, -1, 4 * C) # # [B, H/2 * W/2, 4 * C]

        input = self.norm(input)
        input = self.reduction(input) # # [B, H/2 * W/2, 2C]

        return input

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, input):
        input = self.fc1(input)
        input = self.act(input)
        input = self.drop1(input)
        input = self.fc2(input)
        input = self.drop2(input)
        return input

class WindowAttention(nn.Module): # W-MSA
    def __init__(self, dim, window_size, num_heads, qkv_bias = True, attn_drop = 0., proj_drop = 0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)

        # 相关位置编码 每个head都有一个 因此是[2M-1 * 2M-1, head]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 获取索引 (0, 0), (0, 1), ...
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing = "ij")) # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1) # [2, Mh * Mw]

        # 计算相对索引
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # [2, Mh * Mw, 1] - [2, 1, Mh * Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # [Mh*Mw, Mh*Mw, 2]
        # 行列 + M-1
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # 行乘 2M - 2
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # 这里是不是要用0
        relative_position_index = relative_coords.sum(-1) # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, 3 * dim, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std = .02)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, input, mask: Optional[torch.Tensor] = None): # mask决定了 SW-MSA或者W-MSA
        B_, N, C = input.shape # [batchsize * num_windows, Mh * Mw, total_embed_dim]

        # qkv() [batchsize * num_windows, Mh * Mw, 3 * total_embed_dim]
        # reshape [batchsize * num_windows, Mh * Mw, 3, num_heads, embed_dim_per_head]
        # permute [3, batchsize * num_windows, num_heads, Mh * Mw, embed_dim_per_head]
        qkv = self.qkv(input).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # [batchsize * num_windows, num_heads, Mh * Mw, Mh * Mw]

        # 位置编码
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) # [Mh * Mw, Mh * Mw, num_head]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # [num_head, Mh * Mw, Mh * Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask [num_window, Mh * Mw, Mh * Mw]
            num_window = mask.shape[0]
            attn = attn.view(B_ // num_window, num_window, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn) #[batchsize * num_windows, num_heads, Mh * Mw, Mh * Mw]

        input = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        input = self.proj(input)
        input = self.proj_drop(input)
        return input

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size = 7, shift_size = 0,
                 mlp_ratio = 4., qkv_bias = True, drop = 0., attn_drop = 0., drop_path = 0.,
                 act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0 - window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size = (self.window_size, self.window_size), num_heads = num_heads, qkv_bias = qkv_bias,
            attn_drop = attn_drop, proj_drop = drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)

    def forward(self, input, attn_mask):
        H, W = self.H, self.W
        B, L, C = input.shape
        assert L == H * W, "input features has wrong size"

        shortcut = input
        input = self.norm1(input)
        input = input.view(B, H, W, C)

        # 把feature map pad到windowsize的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        input = F.pad(input, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = input.shape

        # cyclic shift SW-MSA操作
        if self.shift_size > 0:
            shifted_input = torch.roll(input, shifts = (-self.shift_size, -self.shift_size), dims = (1, 2))
        else:
            shifted_input = input
            attn_mask = None

        # partition windows 分窗口
        input_windows = window_partition(shifted_input, self.window_size) #[B * num_windows, Mh, Mw, C]
        input_windows = input_windows.view(-1, self.window_size * self.window_size, C) # [B * num_windows, Mh * Mw, C]

        # W-MSA / SW-MSA
        attn_windows = self.attn(input_windows, mask = attn_mask) # [B * num_windows, Mh * Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # [B * num_windows, Mh, Mw, C]
        shifted_input = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # 移回去
        if self.shift_size > 0:
            input = torch.roll(shifted_input, shifts = (self.shift_size, self.shift_size), dims = (1, 2))
        else:
            input = shifted_input

        # 把pad移去
        if pad_r > 0 or pad_b > 0:
            input = input[:, :H, :W, :].contiguous()

        input = input.view(B, H * W, C)

        input = shortcut + self.drop_path(input)
        input = input + self.drop_path(self.mlp(self.norm2(input)))

        return input

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio = 4., qkv_bias = True, drop = 0., attn_drop = 0.,
                 drop_path = 0., norm_layer = nn.LayerNorm, downsample = None, use_checkpoint = False):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2 # M/2向下取整

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim = dim,
                num_heads = num_heads,
                window_size = window_size,
                shift_size = 0 if (i % 2 == 0) else self.shift_size, # 偶数需要
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                drop = drop,
                attn_drop = attn_drop,
                drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer = norm_layer
            )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim = dim, norm_layer = norm_layer)
        else:
            self.downsample = None

    def create_mask(self, input, H, W):
        # SW-MSA的mask
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        # 调整成和feature map一样的形状
        img_mask = torch.zeros((1, Hp, Wp, 1), device = input.device) # [1, Hp, Wp, 1]

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size) # [num_windows, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [num_windows, Mh * Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # [num_windows, 1, Mh * Mw] - [num_windows, Mh * Mw, 1]
        # [num_windows, Mh * Mw, Mh * Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, input, H, W):
        attn_mask = self.create_mask(input, H, W) # [num_windows, Mh * Mw, Mh * Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                input = checkpoint.checkpoint(blk, input, attn_mask)
            else:
                input = blk(input, attn_mask)
        if self.downsample is not None:
            input = self.downsample(input, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return input, H, W

class SwinTransformer(nn.Module):
    def __init__(self, patch_size = 4, in_channel = 3, num_classes = 1000,
                 embed_dim = 96, depths = (2, 2, 6, 2), num_heads = (3, 6, 12, 24),
                 window_size = 7, mlp_ratio = 4., qkv_bias = True,
                 drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.1,
                 norm_layer = nn.LayerNorm, patch_norm = True,
                 use_checkpoint = False, **kwargs):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            patch_size = patch_size, in_channel = in_channel, embed_dim = embed_dim,
            norm_layer = norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim = int(embed_dim * 2 ** i_layer),
                                depth = depths[i_layer],
                                num_heads = num_heads[i_layer],
                                window_size = window_size,
                                mlp_ratio = self.mlp_ratio,
                                qkv_bias = qkv_bias,
                                drop = drop_rate,
                                attn_drop = attn_drop_rate,
                                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer = norm_layer,
                                downsample = PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint = use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input):
        input, H, W = self.patch_embed(input)
        input = self.pos_drop(input) # [B, H * W, C]

        for layer in self.layers:
            input, H, W = layer(input, H, W)

        input = self.norm(input)
        input = self.avgpool(input.transpose(1, 2)) # [B, C, 1]
        input = torch.flatten(input, 1)
        input = self.head(input)
        return input

def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model

# X = torch.rand((1, 3, 224, 224))
# # model = PatchEmbed()
# # input, H, W = model(X)
# # print(input.shape, H, W)
# #
# # model1 = PatchMerging(dim = 96)
# # model1(input, H, W)
# model = swin_tiny_patch4_window7_224()
# print(model(X).shape)