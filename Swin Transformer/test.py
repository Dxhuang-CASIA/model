import torch

coords_h = torch.arange(2)
coords_w = torch.arange(2)
coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing = "ij"))
print(coords)
coords_flatten = torch.flatten(coords, 1)
print(coords_flatten)
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
print(relative_coords.permute(1, 2, 0).sum(-1).view(-1))