import os
import logging
import csv
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from Loss import ContrastiveLoss  
from moco_model_encoder_superpixel import MoCoV2Encoder  
import time
import math
import glob
import json
from dataloader_superpixel2 import SuperpixelMoCoDatasetNeighbor, get_moco_v2_augmentations



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JSON_PATH = "/home/valentin/workspaces/histolung/data/interim/tiles_superpixels_with_overlap/superpixel_mapping_train.json"
MODEL_SAVE_DIR = "/mnt/nas7/data/Personal/Darya/saved_models/superpixel_org"
CHECKPOINT_SAVE_DIR = "/mnt/nas7/data/Personal/Darya/Checkpoints/superpixel_org" 
PLOT_SAVE_DIR = "/home/darya/Histo_pipeline/Loss_curve_plot"
CSV_SAVE_PATH = "/home/darya/Histo_pipeline/Superpixel_org.csv"

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)

# Create the CSV file if it doesn't exist
if not os.path.exists(CSV_SAVE_PATH):
    with open(CSV_SAVE_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Batch Size", "Temperature", "Average Training Loss", 
            "Training Time", "Metric Type", "Number of Epochs"
        ])
    logger.info(f"CSV file created at {CSV_SAVE_PATH}")


torch.cuda.empty_cache()

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def update_template_csv(csv_path, batch_size, temperature, avg_loss, train_time, num_epochs):
    """
    Update the CSV file with training details.
    """
    csv_headers = [
        "Batch Size", "Temperature", "Average Training Loss", 
        "Training Time", "Metric Type", "Number of Epochs"
    ]
    row_data = [
        batch_size, temperature, avg_loss, train_time, 
        "ContrastiveLoss", num_epochs
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(csv_headers)
        writer.writerow(row_data)


# Learning Rate Scheduler (Cosine with Warmup) - Manual
def adjust_learning_rate(optimizer, epoch, base_lr, total_epochs):
    """
    MoCo v2: Linear warmup for first 10 epochs, then cosine decay.
    """
    warmup_epochs = 10  
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs  # Linear warmup
    else:
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))  # Cosine decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # Apply LR to optimizer
    
    return lr  # Return LR for logging

def save_checkpoint(epoch, model, optimizer, scaler, base_lr, checkpoint_path, best=False):
    """ Save model, optimizer, scaler, and learning rate state. """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'learning_rate': base_lr  # Save base LR
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    if best:
        best_path = os.path.join(CHECKPOINT_SAVE_DIR, "superpixel_org_best_model.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"New best model saved: {best_path}")

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "superpixel_org_*.pth")), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None  

def load_existing_losses(plot_save_dir):
    """Load previous loss values if they exist."""
    loss_file = os.path.join(plot_save_dir, f"superpixel_org_loss_curve.txt")
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            losses = [float(line.strip()) for line in f.readlines()]
        return losses
    return []

def save_losses(losses, plot_save_dir):
    """Save the loss values to a file."""
    os.makedirs(plot_save_dir, exist_ok=True)  # Ensure the directory exists
    
    loss_file = os.path.join(plot_save_dir, "superpixel_org_loss_curve.txt")
    
    with open(loss_file, "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")    

def train_moco(
    json_path,
    model=None,
    batch_size=128,
    epochs=100,
    alpha=0.5,
    learning_rate=0.003,
    temperature=0.07,
    csv_path=CSV_SAVE_PATH,
    device="cuda",
    resnet_type=None,
    resume_checkpoint=None
):
    logger.info("Starting MoCo training with Superpixel-based DataLoader...")
    start_time = datetime.now()

    
    # DataLoader using MoCo-style augmentations
    logger.info("Initializing DataLoader...")
    train_transform = get_moco_v2_augmentations()
    train_dataset = SuperpixelMoCoDatasetNeighbor(json_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True, prefetch_factor=4)

    logger.info(f"Training DataLoader loaded with {len(train_loader)} batches.")

    # Move model to device
    model.to(device)

    # Optimizer and 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
   
    # Loss function
    criterion = ContrastiveLoss()

    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')

    # Find the latest checkpoint if resume_checkpoint is True
    if resume_checkpoint is True:
        resume_checkpoint = get_latest_checkpoint(MODEL_SAVE_DIR)


    start_epoch = 0
    best_loss = float("inf")

    # Load existing loss values
    epoch_losses = load_existing_losses(PLOT_SAVE_DIR)

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        # Move optimizer states to GPU
        for state in optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)
            elif isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        start_epoch = checkpoint['epoch'] + 1
        learning_rate = checkpoint['learning_rate']
        logger.info(f"Resuming training from epoch {start_epoch}, LR: {learning_rate}")
    else:
        logger.info("No valid checkpoint found, starting training from scratch.")


    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs} started.")
        model.train()
        running_loss = 0.0
        num_batches = 0
        data_loading_times = []
        batch_times = []

        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, learning_rate, epochs)
        logger.info(f"Learning rate adjusted to: {lr:.6f}")

        data_start_time = time.perf_counter()
        epoch_start_time = time.perf_counter()

        for batch_idx, (images_q, images_k1, images_k2) in enumerate(train_loader):
            # Measure data loading time
            data_end_time = time.perf_counter()
            data_loading_time = data_end_time - data_start_time
            data_loading_times.append(data_loading_time)
            batch_start_time = time.perf_counter()
    
            images_q = images_q.to(device, non_blocking=True)
            images_k1 = images_k1.to(device, non_blocking=True)
            images_k2 = images_k2.to(device, non_blocking=True)

            # Forward pass
            with torch.amp.autocast(device_type="cuda"):
                q, k1, k2 = moco_model(images_q, images_k1, images_k2)  
                loss_tile = criterion(q, k1, model.queue)
                loss_neighbor = criterion(q, k2, model.queue)
                loss = alpha*loss_tile + (1 - alpha)*loss_neighbor
                logger.info("Loss Calculated")

            # Backward pass with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update queue
            moco_model.update_queue(k1, k2)

            batch_end_time = time.perf_counter()  # End batch processing time
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)

            running_loss += loss.item()
            num_batches += 1

            logger.info(f"Batch {batch_idx+1}/{len(train_loader)} -> Data Loading Time: {data_loading_time:.6f}s, Batch Processing Time: {batch_time:.6f}s")
            logger.info(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Average time per batch: {(time.perf_counter() - epoch_start_time)/(batch_idx+1):.6f}s")
            logger.info(f"Loss: {loss.item():.4f}, Tile Loss: {loss_tile.item():.4f}, Neighbor Loss: {loss_neighbor.item():.4f}")
            
            # Start timing for the next batch loading
            data_start_time = time.perf_counter()

            #if batch_idx + 1 == 20:
             #  break

        avg_epoch_loss = running_loss / num_batches if num_batches > 0 else float("inf")
        epoch_losses.append(avg_epoch_loss)
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        
       

        # Save checkpoint
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"superpixel_org_{epoch}.pth")
        save_checkpoint(epoch, model, optimizer, scaler, learning_rate, checkpoint_path)

        save_losses(epoch_losses, PLOT_SAVE_DIR)

        # Save every 5 epochs and best model
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_SAVE_DIR, f"superpixel_org_checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(epoch, model, optimizer, scaler, learning_rate, checkpoint_path)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_checkpoint(epoch, model, optimizer, scaler, learning_rate, checkpoint_path, best=True)




    writer.close()


    elapsed_time = datetime.now() - start_time
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    update_template_csv(csv_path, batch_size, temperature, avg_loss, str(elapsed_time), epochs)

    # Plot and save training loss curve
    logger.info("Plotting and saving training loss curve...")
    train_plot_path = os.path.join(PLOT_SAVE_DIR, f"Superpixel_org_loss_curve.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(train_plot_path)
    plt.close()
    logger.info(f"Training loss curve saved at {train_plot_path}")

    

if __name__ == "__main__":
    resnet_type = "resnet50"
    moco_model = MoCoV2Encoder(base_encoder=resnet_type, output_dim=128, queue_size=65536)  # Reduce from 65536 32768
   
   
    train_moco(
        json_path=JSON_PATH,
        model=moco_model,
        batch_size=256,
        epochs=100,
        alpha=0.5,
        learning_rate=0.003,
        device=device,
        resnet_type="resnet50",
        resume_checkpoint=False)