#import os
#import torch
#from torch.utils.data import DataLoader
#from tqdm import tqdm
#import argparse
#
#from src.models.deepwind import DeepWindModel
#from src.models.adapter import apply_finetune_strategy
#from src.data.datasets import DeepWindFinetuneDataset
#
#def main(args):
#    # --------------------------------------------------------------------------
#    # 1. Setup & Device
#    # --------------------------------------------------------------------------
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(f"[{args.site_name}] Starting fine-tuning on {device}...")
#    
#    os.makedirs(args.output_dir, exist_ok=True)
#
#    # --------------------------------------------------------------------------
#    # 2. Load Pre-trained Foundation Model
#    # --------------------------------------------------------------------------
#    print("Loading Foundation Model...")
#    # Initialize config and model structure
#    model = DeepWindModel.from_pretrained(args.pretrained_path)
#    config = model.config
#    # --------------------------------------------------------------------------
#    # 3. Apply Fine-tuning Strategy (PEFT + Head)
#    # --------------------------------------------------------------------------
#    print("Applying PEFT Strategy (Variate-LoRA + Router-Full + Head-Full)...")
#    # This function wraps the model with LoRA and sets requires_grad correctly
#    model = apply_finetune_strategy(model, lora_r=16)
#    model.to(device)
#
#    # --------------------------------------------------------------------------
#    # 4. Prepare Few-shot Data
#    # --------------------------------------------------------------------------
#    print(f"Loading few-shot data from {args.data_path}...")
#    dataset = DeepWindFinetuneDataset(
#        npy_path=args.data_path,
#        metadata_path=args.metadata_path,
#        context_length=config.context_length,
#        stride=args.stride,
#        finetune_rate=args.finetune_rate
#    )
#    print(f"total samples: {len(dataset)}")
#    
#    dataloader = DataLoader(
#        dataset, 
#        batch_size=args.batch_size, 
#        shuffle=True, 
#        num_workers=4,
#        pin_memory=True
#    )
#
#    # --------------------------------------------------------------------------
#    # 5. Optimizer
#    # --------------------------------------------------------------------------
#    # IMPORTANT: Only pass trainable parameters to the optimizer!
#    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
#
#    # Optional: Learning Rate Scheduler (Warmup then Cosine is standard)
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
#
#    # --------------------------------------------------------------------------
#    # 6. Training Loop
#    # --------------------------------------------------------------------------
#    model.train()
#    best_loss = float('inf')
#    
#    for epoch in range(args.epochs):
#        epoch_loss = 0.0
#        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
#        
#        for batch in progress_bar:
#            # Move data to GPU
#            context = batch['context'].to(device)
#            variate_ids = batch['variate_ids'].to(device)
#            channel_mask = batch['channel_mask'].to(device)
#            site_coords = batch['site_coords'].to(device)
#            has_coords = batch['has_coords'].to(device)
#
#            optimizer.zero_grad()
#            
#            # Forward pass
#            # Note: Ensure your model forward returns (loss, output) or calculate loss here
#            outputs = model(
#                context=context,
#                variate_ids=variate_ids,
#                channel_mask=channel_mask,
#                site_coords=site_coords,
#                has_coords=has_coords
#            ) 
#            
#            # Assuming your model definition includes the criterion internally
#            # Or you compute it externally:
#            loss = outputs.loss
#            loss.backward()
#            
#            # Gradient Clipping (Essential for stable fine-tuning)
#            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#            
#            optimizer.step()
#            
#            epoch_loss += loss.item()
#            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
#        
#        scheduler.step()
#        avg_loss = epoch_loss / len(dataloader)
#        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
#
#        # ----------------------------------------------------------------------
#        # 7. Checkpointing
#        # ----------------------------------------------------------------------
#        # Save only if loss improved
#        if avg_loss < best_loss:
#            best_loss = avg_loss
#
#            save_path = os.path.join(args.output_dir, f"best_adapter_{args.site_name}")
#            os.makedirs(save_path, exist_ok=True)
#
#            # Save ONLY the adapter weights (not the whole backbone) to save space
#            model.save_pretrained(save_path) 
#            print(f"[Saved best adapter to] {save_path}")
#
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--site_name", type=str, required=True, help="Name of the target wind farm")
#    parser.add_argument("--data_path", type=str, required=True, help="Path to few-shot csv")
#    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata csv")
#    parser.add_argument("--pretrained_path", type=str, required=True, help="Path to foundation model #checkpoint")
#    parser.add_argument("--output_dir", type=str, default="./checkpoints/finetune")
#    parser.add_argument("--epochs", type=int, default=10, help="Few-shot usually needs fewer epochs (e.g. #10-20)")
#    parser.add_argument("--batch_size", type=int, default=4)
#    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Use smaller LR than pretraining")
#    parser.add_argument("--stride", type=int, default=32)
#    parser.add_argument("--finetune_rate", type=float, default=1.0)
#    
#    args = parser.parse_args()
#    main(args)


import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import argparse

# --- DeepWind Project Imports ---
from src.models.deepwind import DeepWindModel
from src.models.adapter import apply_finetune_strategy
from src.data.datasets import DeepWindFinetuneDataset

def setup_distributed():
    """Initializes the distributed process group and sets the device."""
    if "LOCAL_RANK" not in os.environ:
        # Fallback for non-distributed running
        return -1
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    """Destroys the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def main(args):
    # 1. Setup Distributed Environment
    local_rank = setup_distributed()
    is_distributed = (local_rank != -1)
    
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        is_main_process = (local_rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        is_main_process = True

    if is_main_process:
        print(f"[{args.site_name}] Starting {'DDP ' if is_distributed else ''}fine-tuning...")
        os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Foundation Model
    if is_main_process:
        print(f"Loading Foundation Model from {args.pretrained_path}...")
    model = DeepWindModel.from_pretrained(args.pretrained_path)
    config = model.config

    # 3. Apply Fine-tuning Strategy (PEFT)
    model = apply_finetune_strategy(model, lora_r=16)
    model.to(device)

    # Wrap with DDP if multiple GPUs are used
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 4. Prepare Dataset and Distributed Sampler
    dataset = DeepWindFinetuneDataset(
        npy_path=args.data_path,
        metadata_path=args.metadata_path,
        context_length=config.context_length,
        stride=args.stride,
        finetune_rate=args.finetune_rate
    )
    if is_main_process:
        print(f"[Finetune dataset] total samples: {len(dataset)}")
    
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        shuffle=(sampler is None), # Only shuffle if not using a sampler
        num_workers=4,
        pin_memory=True
    )

    # 5. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 6. Training Loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch) # Important for shuffling
        
        epoch_loss = 0.0
        
        # Setup progress bar only for rank 0
        data_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") if is_main_process else dataloader
        
        for batch in data_iterator:
            context = batch['context'].to(device)
            variate_ids = batch['variate_ids'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            site_coords = batch['site_coords'].to(device)
            has_coords = batch['has_coords'].to(device)

            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(
                context=context,
                variate_ids=variate_ids,
                channel_mask=channel_mask,
                site_coords=site_coords,
                has_coords=has_coords
            ) 
            
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            
            # Update tqdm postfix correctly
            if is_main_process:
                data_iterator.set_postfix({"loss": f"{current_loss:.4f}"})
        
        scheduler.step()
        
        # --- Aggregate Loss Across All Ranks ---
        total_loss_tensor = torch.tensor(epoch_loss).to(device)
        if is_distributed:
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            # Average over total number of batches across all GPUs
            avg_loss = total_loss_tensor.item() / (len(dataloader) * world_size)
        else:
            avg_loss = epoch_loss / len(dataloader)

        # 7. Checkpointing (Rank 0 only)
        if is_main_process:
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(args.output_dir, f"best_adapter_{args.site_name}")
                
                # Unmask DDP model to save via PEFT's save_pretrained
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(save_path) 
                print(f"[Best Model Saved] {save_path}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepWind Distributed Fine-tuning")
    parser.add_argument("--site_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/finetune")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--finetune_rate", type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)
