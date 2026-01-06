import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from gwel.networks.UNET import UNet
from gwel.networks.unet.data_loader import CocoSegmentationDataset
import numpy as np
import ast 
import pandas as pd

def train(model, dataloader, criterion, optimizer, device, epoch, num_classes):
    model.train()
    epoch_loss = 0.0

    # Track TP, FP, FN per class
    iou_tp = torch.zeros(num_classes, device=device)
    iou_fp = torch.zeros(num_classes, device=device)
    iou_fn = torch.zeros(num_classes, device=device)

    for images, masks in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device).long()   # [N, H, W]

        #masks = masks.squeeze(0).long()
        
        optimizer.zero_grad()
        outputs = model(images)           # [N, C, H, W]

        # Loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Predictions
        preds = torch.argmax(outputs, dim=1)  # [N, H, W]

        # IoU statistics
        for c in range(num_classes):
            pred_c = (preds == c)
            mask_c = (masks == c)

            tp = (pred_c & mask_c).sum()
            fp = (pred_c & ~mask_c).sum()
            fn = (~pred_c & mask_c).sum()

            iou_tp[c] += tp
            iou_fp[c] += fp
            iou_fn[c] += fn

    avg_loss = epoch_loss / len(dataloader)

    # Compute IoU per class
    iou_per_class = []
    for c in range(num_classes):
        denom = (iou_tp[c] + iou_fp[c] + iou_fn[c]).item()
        iou = (iou_tp[c].item() / denom) if denom > 0 else float("nan")
        iou_per_class.append(iou)

    # Prepare metrics row
    metrics = {"epoch": epoch + 1, "loss": avg_loss}
    for c, iou in enumerate(iou_per_class):
        metrics[f"iou_class_{c}"] = float(f"{iou:.3g}")

    return pd.DataFrame([metrics])


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    print(f"Validation loss: {val_loss / len(val_loader):.4f}")


def main(args):

    os.makedirs(args.save_dir, exist_ok=True)

    categories = ast.literal_eval(args.categories)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # UNet with multi-class output
    model = UNet(in_channels=args.in_channels, out_channels=len(categories)+1).to(device)

    # Use CrossEntropyLoss for multi-class segmentation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_dataset = CocoSegmentationDataset(
        image_dir=args.train_images_dir, 
        coco_json=args.train_annotations,
        categories=categories
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    train_data = pd.DataFrame()

    start_epoch = 0

    # Load checkpoint if specified
    if args.resume_checkpoint is not None and os.path.exists(args.resume_checkpoint):
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)  # resume from next epoch
        print(f"Resuming training from epoch {start_epoch + 1}")

        # Load previous metrics CSV if it exists
        if os.path.exists(args.train_metrics_path):
            train_data = pd.read_csv(args.train_metrics_path)

    for epoch in range(start_epoch, args.epochs):
        new_row = train(model, train_loader, criterion, optimizer, device, epoch, len(categories)+1)
        train_data = pd.concat([train_data, new_row], ignore_index=True)
        train_data.to_csv(args.train_metrics_path, index=False)


        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"unet_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                    }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a U-Net model")

    parser.add_argument('-td','--train_images_dir', type=str, required=True, help="Path to directory of training images")
    parser.add_argument('-ta','--train_annotations', type=str, required=True, help="Path to training annotations")
    parser.add_argument('-c','--categories', type=str, required=True, help="Integer tag for annotation categories")
    parser.add_argument('-s','--save_dir', type=str, required=True, help="Directory to save model checkpoints")

    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--in_channels', type=int, default=1, help="Number of input channels (1 for grayscale images)")
    parser.add_argument('--out_channels', type=int, default=1, help="Number of output channels (1 for binary segmentation)")
    parser.add_argument('--save_interval', type=int, default=1, help="How often to save model checkpoints (in epochs)")
    parser.add_argument('--val_interval', type=int, default=1, help="How often to validate the model (in epochs)")
    parser.add_argument('-vd','--val_images_dir', type=str,default=None, help="Path to directory of validation images")
    parser.add_argument('-va','--val_annotations', type=str, default=None, help="Path to validation annotations") 
    parser.add_argument('--train_metrics_path',type=str,default="train-data.csv",help="Path to output train metrics csv")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to a checkpoint .pth to resume training")
    args = parser.parse_args()

    # Check that paths exist
    assert os.path.exists(args.train_images_dir), f"Train images path '{args.train_images_dir}' not found."
    assert os.path.exists(args.train_annotations), f"Train masks path '{args.train_annotations}' not found."
    os.makedirs(args.save_dir, exist_ok=True)

    # Start training
    main(args)

