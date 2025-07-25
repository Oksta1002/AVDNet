import numpy as np
import torch
import torch.nn.functional as F
from vd.utils import rgb2ycbcr_pt
from vd.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_vd_psnr(img, img2):
    # normalized_psnr = -10 * np.log10(np.mean(np.power(img - img2, 2)))
    img_np = (img * 255.0).round()
    img2_np = (img2 * 255.0).round()
    mse = np.mean((img_np - img2_np) ** 2)
    if mse == 0:
        return float('inf')
    # if normalized_psnr == 0:
    #     return float('inf')
    # return normalized_psnr
    return 20 * np.log10(255.0 / np.sqrt(mse))

@METRIC_REGISTRY.register()
def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)
    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)
    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))
