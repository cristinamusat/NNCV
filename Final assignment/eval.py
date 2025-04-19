import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype
import torchvision.models.segmentation as segmentation

def dice_coefficient_binary(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)

def evaluate(data_dir, model_path, batch_size=16, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transforms (should match your training preprocessing)
    transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])
    
    # Load Cityscapes validation set
    valid_dataset = Cityscapes(
        data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Define the model (using DeepLabV3 as in your training script)
    model = segmentation.deeplabv3_resnet101(pretrained=False)
    model.classifier[4] = nn.Conv2d(256, 19, kernel_size=(1,1))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define group mappings (Cityscapes classes: 0 to 18)
    group_mapping = {
        "flat": [0, 1],
        "construction": [2, 3, 4],
        "object": [5, 6, 7],
        "nature": [8, 9],
        "sky": [10],
        "human": [11, 12],
        "vehicle": [13, 14, 15, 16, 17, 18]
    }
    
    # Initialize accumulators for intersections and unions for each group
    group_intersection = {group: 0.0 for group in group_mapping}
    group_union = {group: 0.0 for group in group_mapping}
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device).long().squeeze(1)  # shape: (B, H, W)
            
            outputs = model(images)['out']
            preds = torch.argmax(outputs.softmax(dim=1), dim=1)  # shape: (B, H, W)
            
            # For each group, compute binary masks and update intersection and union counts
            for group, class_ids in group_mapping.items():
                # Create binary mask for predictions: pixel=1 if predicted class is in the group, else 0
                pred_mask = torch.zeros_like(preds).float()
                label_mask = torch.zeros_like(labels).float()
                for cid in class_ids:
                    pred_mask += (preds == cid).float()
                    label_mask += (labels == cid).float()
                # Convert any non-zero value to 1 (binary)
                pred_mask = (pred_mask > 0).float()
                label_mask = (label_mask > 0).float()
                group_intersection[group] += (pred_mask * label_mask).sum().item()
                group_union[group] += (pred_mask.sum() + label_mask.sum()).item()
    
    # Compute Dice for each group
    dice_scores = {}
    for group in group_mapping:
        dice = (2 * group_intersection[group] + 1e-6) / (group_union[group] + 1e-6)
        dice_scores[f"Dice {group}"] = dice
    
    # Mean Dice is the average of the group Dice scores
    mean_dice = np.mean(list(dice_scores.values()))
    dice_scores["Mean Dice"] = mean_dice
    
    return dice_scores

if __name__ == "__main__":
    data_dir = "./data/cityscapes"  # Adjust if necessary
    model_path = "checkpoints/deeplabv3-training/best_model-epoch=0010-val_loss=0.327847920358181.pth"
    scores = evaluate(data_dir, model_path, batch_size=16, num_workers=4)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
