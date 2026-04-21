import json
import os
import torch
from torch.utils.data import Dataset
import sys
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer.midi_tokenizer import (
    TOKEN_TO_ID,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    VOCAB_SIZE,
    tokens_from_string,
    encode,
)

def load_tokenizer(path="tokenizer.json"):
    return Tokenizer.from_file(path)

class MidiLMDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_seq_len=1024):
        with open(dataset_path, "r") as f:
            self.raw_data = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        self._prepare()

    def _prepare(self):
        bos_id = self.tokenizer.token_to_id("[BOS]")
        eos_id = self.tokenizer.token_to_id("[EOS]")
        pad_id = self.tokenizer.token_to_id("[PAD]")
        sep_id = self.tokenizer.token_to_id("SEP")
        for item in self.raw_data:
            prompt_encoded = self.tokenizer.encode(item["prompt"]).ids
            music_encoded = self.tokenizer.encode(item["tokens"]).ids
            seq = [bos_id] + prompt_encoded + [sep_id] + music_encoded + [eos_id]
            seq = seq[:self.max_seq_len]
            while len(seq) < self.max_seq_len:
                seq.append(pad_id)
            self.samples.append(seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

def save_checkpoint(model, optimizer, step, config, tokenizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "vocab_size": tokenizer.get_vocab_size(),
    }, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("step", 0), checkpoint.get("config", {}), checkpoint.get("vocab_size", 8192)
