import argparse
import os
import sys
import torch
import math
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from model.model import MidiLM, MidiLMConfig
from utils import MidiLMDataset, build_prompt_vocab, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset/example_dataset.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--colab", action="store_true")
    return parser.parse_args()

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.001, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    args = parse_args()
    if args.colab: args.checkpoint_dir = "/content/drive/MyDrive/MidiLM/checkpoints"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found. Run prepare_data.py first.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_vocab, total_vocab = build_prompt_vocab(args.dataset)
    config = MidiLMConfig(vocab_size=total_vocab)
    model = MidiLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = GradScaler('cuda')
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        global_step, _, _, _ = load_checkpoint(args.resume, model, optimizer)
    dataset = MidiLMDataset(args.dataset, prompt_vocab)
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check your MIDI source folder and run prepare_data.py.")
        return
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)
    model.train()
    best_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        for i, (input_ids, targets) in enumerate(dataloader):
            input_ids, targets = input_ids.to(device), targets.to(device)
            with autocast('cuda'):
                _, loss, _ = model(input_ids, targets)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            epoch_loss += loss.item() * args.grad_accum
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} loss: {avg_loss}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, global_step, config, prompt_vocab, total_vocab, os.path.join(args.checkpoint_dir, "best.pt"))
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(model, optimizer, global_step, config, prompt_vocab, total_vocab, os.path.join(args.checkpoint_dir, "latest.pt"))

if __name__ == "__main__":
    main()
