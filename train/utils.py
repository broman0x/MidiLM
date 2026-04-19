import json
import os
import torch
from torch.utils.data import Dataset
import sys

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

PROMPT_SEP = "SEP"

def build_prompt_vocab(dataset_path):
    global VOCAB_SIZE, ID_TO_TOKEN
    with open(dataset_path, "r") as f:
        data = json.load(f)
    words = set()
    lyric_tokens = set()
    for sample in data:
        for word in sample["prompt"].split():
            words.add(word)
        for tok in sample["tokens"].split():
            if tok.startswith("LYRIC_"):
                if tok not in TOKEN_TO_ID:
                    lyric_tokens.add(tok)
    
    for tok in sorted(lyric_tokens):
        if tok not in TOKEN_TO_ID:
            TOKEN_TO_ID[tok] = VOCAB_SIZE
            ID_TO_TOKEN[VOCAB_SIZE] = tok
            VOCAB_SIZE += 1
            
    start_id = VOCAB_SIZE
    vocab = {PROMPT_SEP: start_id}
    start_id += 1
    for word in sorted(words):
        vocab[word] = start_id
        start_id += 1
    return vocab, start_id

def tokenize_prompt(prompt_text, prompt_vocab):
    ids = []
    for word in prompt_text.split():
        if word in prompt_vocab:
            ids.append(prompt_vocab[word])
    return ids

class MidiLMDataset(Dataset):
    def __init__(self, dataset_path, prompt_vocab, max_seq_len=1024):
        with open(dataset_path, "r") as f:
            self.raw_data = json.load(f)
        self.prompt_vocab = prompt_vocab
        self.max_seq_len = max_seq_len
        self.samples = []
        self._prepare()

    def _prepare(self):
        sep_id = self.prompt_vocab[PROMPT_SEP]
        for item in self.raw_data:
            prompt_ids = tokenize_prompt(item["prompt"], self.prompt_vocab)
            music_tokens = tokens_from_string(item["tokens"])
            music_ids = [TOKEN_TO_ID[t] for t in music_tokens if t in TOKEN_TO_ID]
            seq = [BOS_ID] + prompt_ids + [sep_id] + music_ids + [EOS_ID]
            seq = seq[: self.max_seq_len]
            while len(seq) < self.max_seq_len:
                seq.append(PAD_ID)
            self.samples.append(seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

def save_checkpoint(model, optimizer, step, config, prompt_vocab, total_vocab, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "prompt_vocab": prompt_vocab,
        "total_vocab": total_vocab,
    }, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("step", 0), checkpoint.get("config", {}), checkpoint.get("prompt_vocab", {}), checkpoint.get("total_vocab", VOCAB_SIZE)
