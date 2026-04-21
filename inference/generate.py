import argparse
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import MidiLM, MidiLMConfig
from train.utils import load_tokenizer
from converter.tokens_to_midi import tokens_to_midi

def generate(model, tokenizer, prompt, max_len=512, temp=1.0, top_p=0.9):
    device = next(model.parameters()).device
    model.eval()
    
    bos_id = tokenizer.token_to_id("[BOS]")
    sep_id = tokenizer.token_to_id("SEP")
    
    prompt_encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([bos_id] + prompt_encoded + [sep_id], device=device).unsqueeze(0)
    
    generated = []
    with torch.no_grad():
        for _ in range(max_len):
            logits, _, _ = model(input_ids)
            logits = logits[:, -1, :] / temp
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            token_str = tokenizer.id_to_token(next_token.item())
            
            if token_str == "[EOS]":
                break
            generated.append(token_str)
            
    return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.mid")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer("tokenizer.json")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = MidiLMConfig.from_dict(checkpoint["config"])
    model = MidiLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"Generating for prompt: {args.prompt}")
    tokens = generate(model, tokenizer, args.prompt, max_len=args.max_len, temp=args.temp)
    print(f"Generated {len(tokens)} tokens.")
    
    if any(t.startswith("TEMPO_") or t.startswith("NOTE_") for t in tokens):
        tokens_to_midi(tokens, args.output)
        print(f"MIDI saved to {args.output}")
    else:
        print("Response: " + " ".join(tokens))

if __name__ == "__main__":
    main()
