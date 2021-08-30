import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch

from network.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--source', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--output', type=str, default=None, help='input low-quality test image folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    print(f'loading model from {args.model_path}')

    """ initialize model
    """
    model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    model.eval()
    model = model.to(device)

    """ Load source image
    """
    source_img = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    source_img = np.expand_dims(source_img, axis=2)
    
    source_img = np.transpose(source_img if source_img.shape[2] == 1 else source_img[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    source_img = torch.from_numpy(source_img).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    window_size = 8
        # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = source_img.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        source_img = torch.cat([source_img, torch.flip(source_img, [2])], 2)[:, :, :h_old + h_pad, :]
        source_img = torch.cat([source_img, torch.flip(source_img, [3])], 3)[:, :, :, :w_old + w_pad]
        output_img = model(source_img)
        output_img = output_img[..., :h_old * args.scale, :w_old * args.scale]

    # save image
    output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output_img.ndim == 3:
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output_img = (output_img * 255.0).round().astype(np.uint8)  # float32 to uint8
    cv2.imwrite(args.output, output_img)

if __name__ == '__main__':
    main()
