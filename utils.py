import torch
import torch.nn as nn
from loss import VGGPerceptualLoss, PSNRLoss, SSIMLoss
import piq

def build_loss_func(loss_dict, device):

    loss_compute_dict = {}

    for key, val in loss_dict.items():
        key = key.lower()
        
        func = None
        # Task : Regression
        if key == 'huber':
            func = nn.HuberLoss(delta=0.1).to(device)

        if key == 'mse':
            func = nn.MSELoss().to(device)

        if key == 'l1':
            func = nn.L1Loss().to(device)

        if key == 'vgg':
            func=VGGPerceptualLoss().to(device)

        if key == 'dists':
            func = piq.DISTS(mean= [0., 0., 0.], std=[1., 1., 1.]).to(device)

        if key == 'lpips':
            func = piq.LPIPS(mean= [0., 0., 0.], std=[1., 1., 1.]).to(device)

        if key == 'psnr':
            func = PSNRLoss().to(device)

        if key == 'ssim':
            func = SSIMLoss().to(device)
    
        if func == None:
            raise NotImplementedError(f"{key} is not implemented yet.")

        weight = val
        loss_compute_dict[key] = {'func': func.to(device), 'weight': weight}

    return loss_compute_dict


def compute_loss(loss_func, src, tgt):
    total_loss = 0

    assert src.get_device() == tgt.get_device(), \
        "Prediction & GT tensor must be in same device"

    for _, loss_dict in loss_func.items():

        loss = loss_dict['func'](src, tgt)
        loss *= loss_dict['weight']
        total_loss += loss
    
    return total_loss

def build_optim(cfg, model):
    optim_name = cfg['train']['optim']
    lr = cfg['train']['lr']
    optim_name = optim_name.lower()

    optim = None
    if optim_name == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    
    if optim_name == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)

    if optim_name == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Add optimizer if you want

    if optim != None:
        return optim
    else:
        raise NotImplementedError(f"{optim_name} is not implemented yet.")


def build_scheduler(cfg, optimizer):
    scheduler_dict = cfg['train']['scheduler']

    sch_name = list(scheduler_dict.keys())[0]
    sch_settings = scheduler_dict[sch_name]
    sch_name = sch_name.lower()

    if sch_name == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sch_settings['milestones'], gamma=sch_settings['gamma']
        )
    
    if sch_name == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=sch_settings['gamma']
        )

    if sch_name == 'cossineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sch_settings['T_max'], eta_min=sch_settings['eta_min']
        )

    # Add optimizer if you want

    if sch_name != None:
        return scheduler
    else:
        raise NotImplementedError(f"{sch_name} is not implemented yet.")
