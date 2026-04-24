import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from tokenizer.midi_tokenizer import ALL_TOKENS

def train_tokenizer(dataset_path, vocab_size=16384):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    texts = []
    for item in data:
        texts.append(item["prompt"])
        if not item.get("tokens", "").startswith("TEMPO_"):
            texts.append(item["tokens"])

    temp_file = "tokenizer_train_data.txt"
    with open(temp_file, "w") as f:
        for t in texts:
            f.write(t + "\n")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "SEP"] + ALL_TOKENS
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True
    )

    tokenizer.train(files=[temp_file], trainer=trainer)
    tokenizer.save("tokenizer.json")
    if os.path.exists(temp_file): os.remove(temp_file)

if __name__ == "__main__":
    train_tokenizer("dataset/example_dataset.json")
