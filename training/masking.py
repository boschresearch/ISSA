import numpy as np
import PIL
import PIL.Image as Image
import torch
from torchvision.transforms import ToPILImage
from einops import rearrange
import einops


def patchify(imgs, patch_size=[4, 4]):
    """
    (N, c, H, W) -> (N, L, patch_size**2 *c)
    """
    p_h, p_w = patch_size
    x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p_h, p2=p_w)
    return x


def unpatchify(x, patch_size=[4, 4], img_ratio=2.0):
    """
    (N, L, patch_size**2 *c) -> (N, c, H, W)
    """

    p_h, p_w = patch_size
    h = int((x.shape[1] / img_ratio) ** .5)
    w = int(h * img_ratio)
    #print('h = ', h, 'w = ', w)
    assert h * w == x.shape[1]

    imgs = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p_h, p2=p_w, h=h, w=w)
    return imgs

def random_masking(x, mask_ratio):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def mask_images(img_tensor, patch_size, mask_ratio,
                img_ratio=2, masked_noise_mode='normalized',given_mask=None):
    """
    [N, C, H, W] -> [N, C, H, W]
    """

    p_h, p_w = patch_size
    N,latent_dim,_,_ = img_tensor.shape
    x = patchify(img_tensor, patch_size=patch_size)  # [1, 512, 192]
    if given_mask is None:
        x_masked, mask, ids_restore = random_masking(x, mask_ratio=mask_ratio)
        return_mask = mask
        mask = mask.unsqueeze(-1).repeat(1, 1, p_h * p_w * latent_dim)  # (N, H*W/(p*p), p*p*3)
    else:
        # given_mask: (N, L)
        mask = given_mask.unsqueeze(-1).repeat(1, 1, p_h * p_w * latent_dim)
        return_mask = given_mask
    mask = unpatchify(mask, patch_size=patch_size, img_ratio=img_ratio)  # 1 is removing, 0 is keeping

    if masked_noise_mode == 'normal':
        random_noise = torch.randn_like(mask)
    elif masked_noise_mode == 'zeros':
        random_noise = torch.zeros_like(mask)
    elif masked_noise_mode == 'normalized':
        mean = torch.mean(img_tensor) #(x_masked)
        std  = torch.std(img_tensor) #(x_masked)
        random_noise = torch.randn_like(mask) * std + mean

    im_masked = img_tensor * (1 - mask) + random_noise  * mask

    return im_masked, mask, return_mask