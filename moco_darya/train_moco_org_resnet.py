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
from moco_model_encoder_resnet import MoCoV2Encoder  
import time
import math
import glob
from moco_data_loader import MoCoTileDataset, TwoCropsTransform, get_moco_v2_augmentations



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories for saving models, plots, and CSV results
#MODEL_SAVE_DIR = "/home/darya/Histo_pipeline/Moco_Original_models"
SECOND_SAVE_DIR = "/mnt/nas7/data/Personal/Darya/saved_models/moco_50_256"
CHECKPOINT_SAVE_DIR = "/mnt/nas7/data/Personal/Darya/Checkpoints/moco_50_256" 
PLOT_SAVE_DIR = "/home/darya/Histo_pipeline/Loss_curve_plot"
CSV_SAVE_PATH = "/home/darya/Histo_pipeline/MoCo_org.csv"

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)

# Create the CSV file if it doesn't exist
if not os.path.exists(CSV_SAVE_PATH):
    with open(CSV_SAVE_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Epoch", "Batch Size", "ResNet Type", "Average Training Loss", 
            "Training Time", "Metric Type", "Number of Epochs"
        ])
    logger.info(f"CSV file created at {CSV_SAVE_PATH}")


torch.cuda.empty_cache()

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
        best_path = os.path.join(CHECKPOINT_SAVE_DIR, "moco_best_model_256_101.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"New best model saved: {best_path}")

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "moco_org_*.pth")), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None  

def load_existing_losses(plot_save_dir, batch_size, resnet_type):
    """Load previous loss values if they exist."""
    loss_file = os.path.join(plot_save_dir, f"MOCO_org_loss_curve_{batch_size}_{resnet_type}.txt")
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            losses = [float(line.strip()) for line in f.readlines()]
        return losses
    return []

def save_losses(losses, plot_save_dir, batch_size, resnet_type):
    """Save the loss values to a file."""
    loss_file = os.path.join(plot_save_dir, f"MOCO_org_loss_curve_{batch_size}_{resnet_type}.txt")
    with open(loss_file, "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")       

def train_moco(
    tile_csv_path,
    model=None,
    batch_size=128,
    epochs=100,
    learning_rate=0.003,
    temperature=0.07,
    csv_path=CSV_SAVE_PATH,
    device="cuda",
    resnet_type=None,
    resume_checkpoint=None
):
    logger.info("Starting MoCo training with Mixed Precision")
    start_time = datetime.now()

    
    # DataLoader using MoCo-style augmentations
    logger.info("Initializing DataLoader...")
    train_transform = TwoCropsTransform(get_moco_v2_augmentations())
    train_dataset = MoCoTileDataset(csv_path=tile_csv_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

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
        resume_checkpoint = get_latest_checkpoint(SECOND_SAVE_DIR)


    start_epoch = 0
    best_loss = float("inf")

    # Load existing loss values
    epoch_losses = load_existing_losses(PLOT_SAVE_DIR, batch_size, resnet_type)

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

        for batch_idx, (images_q, images_k) in enumerate(train_loader):
            # Measure data loading time
            data_end_time = time.perf_counter()
            data_loading_time = data_end_time - data_start_time
            data_loading_times.append(data_loading_time)
            batch_start_time = time.perf_counter()
    
            images_q = images_q.to(device, non_blocking=True)
            images_k = images_k.to(device, non_blocking=True)

            # Forward pass
            with torch.amp.autocast(device_type="cuda"):
                q, k = model(images_q, images_k)  
                loss = criterion(q, k, model.queue)
                logger.info("Loss Calculated")

            # Backward pass with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update queue
            model.update_queue(k)

            batch_end_time = time.perf_counter()  # End batch processing time
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)

            running_loss += loss.item()
            num_batches += 1

            logger.info(f"Batch {batch_idx+1}/{len(train_loader)} -> Data Loading Time: {data_loading_time:.6f}s, Batch Processing Time: {batch_time:.6f}s")
            logger.info(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Start timing for the next batch loading
            data_start_time = time.perf_counter()

            #if batch_idx + 1 == 20:
             #   break

        avg_batch_time = sum(batch_time)/len(batch_time)
        avg_epoch_loss = running_loss / num_batches if num_batches > 0 else float("inf")
        epoch_losses.append(avg_epoch_loss)
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        logger.info(f"Epoch {epoch + 1} completed -> Avg Data Loading Time: {avg_data_loading_time:.6f}s, Avg Batch Time: {avg_batch_time:.6f}s, Loss: {avg_epoch_loss:.4f}")
        
       

        # Save checkpoint
        checkpoint_path = os.path.join(SECOND_SAVE_DIR, f"moco_org_{batch_size}_{resnet_type}_{epoch}.pth")
        save_checkpoint(epoch, model, optimizer, scaler, learning_rate, checkpoint_path)

        save_losses(epoch_losses, PLOT_SAVE_DIR, batch_size, resnet_type)

        # Save every 5 epochs and best model
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_SAVE_DIR, f"moco_checkpoint_epoch_{epoch+1}_256_50.pth")
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
    train_plot_path = os.path.join(PLOT_SAVE_DIR, f"MOCO_org_loss_curve_256_50.png")
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
   

    # Path to tile_path.csv (contains all  train image paths)
    tile_csv_path = "/home/darya/Histo/Histo_pipeline_csv/train_path.csv"
   
    train_moco(
        tile_csv_path=tile_csv_path,
        model=moco_model,
        batch_size=256,
        epochs=12,
        learning_rate=0.003,
        temperature=0.07,
        csv_path=CSV_SAVE_PATH,
        device=device,
        resnet_type=resnet_type,
        resume_checkpoint=True
    )