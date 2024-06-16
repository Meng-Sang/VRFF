import math

import cv2
import kornia
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def letterbox_image(image, target_size):
    src_height, src_width = image.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / src_width, target_height / src_height)
    new_width = int(src_width * scale)
    new_height = int(src_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_height, target_width, 3), 128, dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return canvas


def to_estimation_tensor(im_cv, unsqueeze=True):
    im_cv = cv2.resize(im_cv, (640, 480))
    im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
    return im_ts.unsqueeze(0) if unsqueeze else im_ts


class MSE(nn.Module):
    """
    Mean squared error loss.
    """

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class LNCC(nn.Module):
    """
        Local (over window) normalized cross correlation.
    """

    def __init__(self):
        super(LNCC, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def forward(self, I, J, win=[17]):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims

        sum_filt = torch.ones([1, 1, *win]).cuda()

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(cc)


class NCC(nn.Module):
    """
        Normalized cross correlation.
    """

    def __init__(self):
        super(NCC, self).__init__()

    def similarity_loss(self, tgt, warped_img):
        sizes = np.prod(list(tgt.shape)[1:])
        flatten1 = torch.reshape(tgt, (-1, sizes))
        flatten2 = torch.reshape(warped_img, (-1, sizes))

        mean1 = torch.reshape(torch.mean(flatten1, dim=-1), (-1, 1))
        mean2 = torch.reshape(torch.mean(flatten2, dim=-1), (-1, 1))
        var1 = torch.mean((flatten1 - mean1) ** 2, dim=-1)
        var2 = torch.mean((flatten2 - mean2) ** 2, dim=-1)
        cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=-1)
        pearson_r = cov12 / torch.sqrt((var1 + 1e-5) * (var2 + 1e-6))
        # ncc_value = torch.sum(1 - pearson_r)
        ncc_value = torch.sum(pearson_r)
        return ncc_value

    def forward(self, y_true, y_pred):
        return self.similarity_loss(y_true, y_pred)