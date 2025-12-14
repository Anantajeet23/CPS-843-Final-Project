import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


@torch.no_grad()
def dice_coeff(pred_logits, target_bin, eps=1e-6):
    # pred_logits: (B, 2, H, W)
    probs = torch.softmax(pred_logits, dim=1)[:, 1]  # (B, H, W)

    preds = (probs > 0.5).float()                    # (B, H, W)
    target = target_bin.float()                      # (B, H, W)

    # SAFE flattening
    preds = preds.view(preds.size(0), -1)
    target = target.view(target.size(0), -1)

    inter = (preds * target).sum(dim=1)
    denom = preds.sum(dim=1) + target.sum(dim=1) + eps

    return (2.0 * inter / denom).mean().item()


def train_fn(loader, model, optimizer, loss_fn, device, num_classes):
    model.train()
    loop = tqdm(loader, total=len(loader), leave=False)
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loop:
        images = images.to(device)

        if masks.ndim == 4 and masks.shape[1] == 3:
            masks = (masks > 0).any(dim=1).long()
        else:
            masks = masks.long().squeeze(1) if masks.ndim == 4 else masks.long()

        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        dice = dice_coeff(logits, masks) if num_classes == 2 else 0.0

        running_loss += loss.item()
        running_dice += dice
        loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

    n = len(loader)
    return running_loss / n, running_dice / n

def train_fn_resnet(loader, model, optimizer, loss_fn, device, num_classes):
    model.train()
    loop = tqdm(loader, total=len(loader), leave=False)

    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loop:
        images = images.to(device)

        # Convert grayscale → RGB for ResNet models
        if images.shape[1] == 1:                    # (B,1,H,W)
            images = images.repeat(1, 3, 1, 1)      # (B,3,H,W)

    
        if masks.ndim == 4 and masks.shape[1] == 3:
            masks = (masks > 0).any(dim=1).long()   # convert RGB mask → 1 channel
        else:
            masks = masks.long().squeeze(1) if masks.ndim == 4 else masks.long()

        masks = masks.to(device)

        outputs = model(images)

        # FCN / DeepLab structure: {"out": tensor}
        if isinstance(outputs, dict):
            logits = outputs["out"]
        else:
            logits = outputs

        
        loss = loss_fn(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


        dice = dice_coeff(logits, masks) if num_classes == 2 else 0.0

        running_loss += loss.item()
        running_dice += dice

        loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

    n = len(loader)
    return running_loss / n, running_dice / n

@torch.no_grad()
def validate_fn_resnet(loader, model, loss_fn, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(device)

        # grayscale → RGB (B,1,H,W) → (B,3,H,W)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        if masks.ndim == 4 and masks.shape[1] == 3:
            masks = (masks > 0).any(dim=1).long()
        else:
            masks = masks.long().squeeze(1) if masks.ndim == 4 else masks.long()

        masks = masks.to(device)

        outputs = model(images)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs

        loss = loss_fn(logits, masks)
        dice = dice_coeff(logits, masks) if num_classes == 2 else 0.0

        running_loss += loss.item()
        running_dice += dice

    n = len(loader)
    return running_loss / n, running_dice / n


@torch.no_grad()
def validate_fn(loader, model, loss_fn, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(device)

        if masks.ndim == 4 and masks.shape[1] == 3:
            masks = (masks > 0).any(dim=1).long()
        else:
            masks = masks.long().squeeze(1) if masks.ndim == 4 else masks.long()

        masks = masks.to(device)

        #logits = model(images)
        
        output = model(images)
        logits = output[0] if isinstance(output, tuple) else output 
        loss = loss_fn(logits, masks)

        dice = dice_coeff(logits, masks) if num_classes == 2 else 0.0
        running_loss += loss.item()
        running_dice += dice

    n = len(loader)
    return running_loss / n, running_dice / n


@torch.no_grad()
def save_predictions(model, loader, device, folder="outputs"):
    import os
    import numpy as np
    from PIL import Image

    model.eval()
    os.makedirs(folder, exist_ok=True)

    counter = 0  # save every prediction, no limit

    for images, masks in loader:
        images = images.to(device)

        # Forward pass
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1, ...]    # (B,H,W)
        preds = (probs > 0.5).float().cpu()                # (B,H,W)

        images = images.cpu()
        masks = masks.cpu()

        for i in range(images.shape[0]):
            img = images[i].numpy()        # (1,H,W)
            img = img.squeeze(0)           # (H,W)
            img = (img * 255).astype(np.uint8)

            gt = masks[i].numpy()
            gt = (gt * 255).astype(np.uint8)

            pred = preds[i].numpy()
            pred = (pred * 255).astype(np.uint8)

            Image.fromarray(img, mode="L").save(f"{folder}/sample_{counter}_image.png")
            Image.fromarray(gt, mode="L").save(f"{folder}/sample_{counter}_mask.png")
            Image.fromarray(pred, mode="L").save(f"{folder}/sample_{counter}_pred.png")

            counter += 1

    model.train()

@torch.no_grad()
def save_predictions_resnet(model, loader, device, folder="outputs"):
    import os
    import numpy as np
    from PIL import Image

    model.eval()
    os.makedirs(folder, exist_ok=True)

    counter = 0

    for images, masks in loader:
        images = images.to(device)

        if images.shape[1] == 1:               # (B,1,H,W)
            images = images.repeat(1, 3, 1, 1)  # (B,3,H,W)

        outputs = model(images)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.5).float().cpu()

        images = images.cpu()
        masks = masks.cpu()

        for i in range(images.shape[0]):
            img = images[i].numpy()
            img = img[0] if img.shape[0] == 1 else img.mean(axis=0)
            img = (img * 255).astype(np.uint8)

            gt = masks[i].numpy().astype(np.uint8) * 255
            pr = preds[i].numpy().astype(np.uint8) * 255

            Image.fromarray(img, mode="L").save(f"{folder}/sample_{counter}_image.png")
            Image.fromarray(gt, mode="L").save(f"{folder}/sample_{counter}_mask.png")
            Image.fromarray(pr, mode="L").save(f"{folder}/sample_{counter}_pred.png")

            counter += 1

    model.train()
