import matplotlib.pyplot as plt
import torch
import os
from dataset.dataset import CTDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model_factory import model_build
from tqdm import tqdm
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.patches as patches
import random
import pandas as pd
import yaml
import argparse

def compute_measure(x, y, pred, data_range):
    original_psnr = compute_PSNR(x, y, data_range)
    original_ssim = compute_SSIM(x, y, data_range)
    original_rmse = compute_RMSE(x, y)
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_ssim = compute_SSIM(pred, y, data_range)
    pred_rmse = compute_RMSE(pred, y)
    return (original_psnr, original_ssim, original_rmse), (pred_psnr, pred_ssim, pred_rmse)


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1,1,shape_ ,shape_ )
        img2 = img2.view(1,1,shape_ ,shape_ )
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def rescale(image, window=[-200, 350]):
    unnormalized = image*4096-1024
    clipped = torch.clip(unnormalized, window[0], window[1])
    rescaled = (clipped - window[0])/(window[1]-window[0])
    return rescaled

def denormalize_(image):
        image = image * (4096) - 1024
        return image


def trunc(mat):
    mat[mat <= -160] = -160
    mat[mat >= 240] = 240
    return mat

def viz_with_data(cfg, exp_name, checkpoint_num=99):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dict_results = {"og_psnr" : [], "pr_psnr" : [], "og_ssim" : [],
                    "pr_ssim" : [], "og_rmse" : [], "pr_rmse" : []}
    indices = []


    os.makedirs(f"result/{exp_name}/", exist_ok=True)
    test_dataset = CTDataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_name = cfg['train']['model']
    model = model_build(model_name=model_name).to(device)
    checkpoint = torch.load(os.path.join(f"checkpoint/{exp_name}/checkpoint_{checkpoint_num}.ckpt"))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    random.seed(2022321249)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):

            width, height = random.randint(50, 120), random.randint(50, 120)
            corner = (random.randint(50, 461-width), random.randint(50, 461-height))

            src, tgt = batch[0].to(device), batch[1].to(device)
            gen = model(src)
            og_measures, pr_measures = compute_measure(src, tgt, gen, 1.0)

            src, tgt, gen = src.detach().cpu().squeeze(0), tgt.detach().cpu().squeeze(0), gen.detach().cpu().squeeze(0)

            src = rescale(src, [-200, 1500])
            tgt = rescale(tgt, [-200, 1500])
            gen = rescale(gen, [-200, 1500])

            plt.figure(figsize=(15, 12))
            ax = plt.subplot(2, 3, 1)
            plt.imshow(src.squeeze(0).numpy(), cmap="gray")
            plt.axis("off")
            plt.title(f"QDCT\n(PSNR : {og_measures[0]:.2f})", fontsize=20)
            ax.add_patch(
                patches.Rectangle(
                    corner,  # (x, y) coordinates of left-bottom corner point
                    width, height,  # width, height
                    edgecolor='red',
                    linewidth=2,
                    facecolor='none'
            ))

            ax = plt.subplot(2, 3, 2)
            plt.imshow(gen.squeeze(0).numpy(), cmap="gray")
            plt.axis("off")
            plt.title(f"Generated\n(PSNR : {pr_measures[0]:.2f})", fontsize=20)
            ax.add_patch(
                patches.Rectangle(
                    corner,  # (x, y) coordinates of left-bottom corner point
                    width, height,  # width, height
                    edgecolor='red',
                    linewidth=2,
                    facecolor='none'
                ))

            ax = plt.subplot(2, 3, 3)
            plt.imshow(tgt.squeeze(0).numpy(), cmap="gray")
            plt.axis("off")
            plt.title(f"NDCT", fontsize=20)
            ax.add_patch(
                patches.Rectangle(
                    corner,  # (x, y) coordinates of left-bottom corner point
                    width, height,  # width, height
                    edgecolor='red',
                    linewidth=2,
                    facecolor='none'
                ))

            plt.subplot(2, 3, 4)
            plt.imshow(src.squeeze(0).numpy()[corner[1]:corner[1]+height, corner[0]:corner[0]+width], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(gen.squeeze(0).numpy()[corner[1]:corner[1] + height, corner[0]:corner[0] + width], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(tgt.squeeze(0).numpy()[corner[1]:corner[1] + height, corner[0]:corner[0] + width], cmap="gray")
            plt.axis("off")

            plt.savefig(f"result/{exp_name}/{str(i).zfill(4)}.png")
            plt.close()

            dict_results["og_psnr"] += [og_measures[0]]
            dict_results["og_ssim"] += [og_measures[1]]
            dict_results["og_rmse"] += [og_measures[2]]

            dict_results["pr_psnr"] += [pr_measures[0]]
            dict_results["pr_ssim"] += [pr_measures[1]]
            dict_results["pr_rmse"] += [pr_measures[2]]

            indices += [str(i)]

    indices += ["AVG", "STDEV"]
    dict_results["og_psnr"] += [np.mean(np.array(dict_results["og_psnr"])), np.std(np.array(dict_results["og_psnr"]))]
    dict_results["og_ssim"] += [np.mean(np.array(dict_results["og_ssim"])), np.std(np.array(dict_results["og_ssim"]))]
    dict_results["og_rmse"] += [np.mean(np.array(dict_results["og_rmse"])), np.std(np.array(dict_results["og_rmse"]))]

    dict_results["pr_psnr"] += [np.mean(np.array(dict_results["pr_psnr"])), np.std(np.array(dict_results["pr_psnr"]))]
    dict_results["pr_ssim"] += [np.mean(np.array(dict_results["pr_ssim"])), np.std(np.array(dict_results["pr_ssim"]))]
    dict_results["pr_rmse"] += [np.mean(np.array(dict_results["pr_rmse"])), np.std(np.array(dict_results["pr_rmse"]))]

    df = pd.DataFrame.from_dict(dict_results)
    df.index = indices
    df.to_csv(f"measure_{exp_name}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline', help='Path for configuration file')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model inference. It can be "cpu" or "cuda" ')
    args = parser.parse_args()

    with open('config/' + args.config + '.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    viz_with_data(cfg, args.config, 30)