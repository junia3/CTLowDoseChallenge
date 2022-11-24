import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import pytorch_model_summary
import torchvision.transforms as transforms

from dataset.dataset import CTDataset
from utils import *
from model_factory import model_build
from loss import *
from visualize_test import *

def train(cfg, args, save_results=True):
    patch = args.patch
    device = args.device

    batch_size = cfg['dataset']['batch']
    img_size = cfg['dataset']['image_size']

    model_name = cfg['train']['model']
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation((-180, 180)),
        transforms.RandomAffine((-180, 180), (0, 0.01), scale=(0.9, 1.1), shear=(-2, 2)),
    ])

    train_dataset = CTDataset("train", transform_train, patch=patch)
    val_dataset = CTDataset("validation")
    
    # Check number of each dataset size
    print(f"Training dataset size : {len(train_dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")
    
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = model_build(model_name=model_name)
    print("Model configuration : ")
    print(pytorch_model_summary.summary(model,
                                torch.zeros(batch_size, 1, img_size, img_size),
                                show_input=True))


    loss_func = build_loss_func(cfg['train']['loss'], device=device)

    optimizer = build_optim(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    epochs = cfg['train']['epochs']

    if 'cuda' in device and torch.cuda.is_available():
        model = model.to(device)

    os.makedirs(f'checkpoint/{args.config}/', exist_ok=True)
    for epoch in range(epochs):
        training_loss = 0.0
        model.train()
        loading = tqdm(enumerate(train_dataloader), desc="training...")
        for i, batch in loading:
            
            optimizer.zero_grad()
            src, tgt = batch[0].to(device), batch[1].to(device)
            if args.patch:
                src = src.view(-1, 1, 64, 64)
                tgt = tgt.view(-1, 1, 64, 64)
                
            gen = model(src)

            loss = compute_loss(loss_func, gen, tgt)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

            loading.set_description(f"Loss : {training_loss/(i+1):.4f}")

        print(f"Epoch #{epoch + 1} >>>> Training loss : {training_loss / len(train_dataloader):.6f}")
        scheduler.step()

        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for i, batch in tqdm(enumerate(val_dataloader)):
                src, tgt = batch[0].to(device), batch[1].to(device)
                gen = model(src)
                loss = compute_loss(loss_func, gen, tgt)
                validation_loss += loss.item()

            print(f"Epoch #{epoch + 1} >>>> Validation loss : {validation_loss / len(val_dataloader):.6f}")
        
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch
            }
            , f"checkpoint/{args.config}/checkpoint_{epoch}.ckpt")
        print(f"Epoch #{epoch + 1} >>>> SAVE .ckpt file")

    if save_results:
        viz_with_data(cfg, args.config, epochs-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline', help='Path for configuration file')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model inference. It can be "cpu" or "cuda" ')
    parser.add_argument('--patch', type=bool, default=False, help="Use patch based training or not")
    args = parser.parse_args()

    with open('config/' + args.config + '.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    train(cfg, args)
