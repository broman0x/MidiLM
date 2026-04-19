import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
sys.path.insert(0, os.path.dirname(__file__))

from model.model import MidiLM, MidiLMConfig
from tokenizer.midi_tokenizer import BOS_ID, EOS_ID, PAD_ID, decode, tokens_to_string, TOKEN_TO_ID, ID_TO_TOKEN, VOCAB_SIZE
from utils import load_checkpoint, tokenize_prompt
from prompt_parser import parse_prompt
from converter.tokens_to_midi import tokens_to_midi
from converter.midi_to_audio import midi_to_mp3

def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    return logits

def generate_tokens(model, prompt_ids, max_new_tokens=512, temperature=1.0, top_p=0.9, repetition_penalty=1.2, device="cpu"):
    model.eval()
    input_ids = torch.tensor([[BOS_ID] + prompt_ids], dtype=torch.long, device=device)
    generated = []
    past_key_values = None
    with torch.no_grad():
        for i in range(max_new_tokens):
            if past_key_values is None:
                logits, _, past_key_values = model(input_ids, use_cache=True)
            else:
                logits, _, past_key_values = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            next_logits = logits[0, -1, :] / temperature
            for token_id in set(generated):
                next_logits[token_id] /= repetition_penalty
            if top_p < 1.0:
                next_logits = top_p_sampling(next_logits, p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == EOS_ID: break
            if next_token == PAD_ID: continue
            input_ids = torch.cat((input_ids, torch.tensor([[next_token]], device=device)), dim=1)
            generated.append(next_token)
    return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--no_audio", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsed = parse_prompt(args.prompt)
    if not parsed["valid"]: sys.exit(1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "total_vocab" in ckpt:
        target_vocab = ckpt["total_vocab"]
        if target_vocab > VOCAB_SIZE:
            for tid in range(VOCAB_SIZE, target_vocab):
                tok = ckpt["id_to_token"].get(tid) if "id_to_token" in ckpt else f"UNKNOWN_{tid}"
                TOKEN_TO_ID[tok] = tid
                ID_TO_TOKEN[tid] = tok
    config = MidiLMConfig.from_dict(ckpt["config"])
    prompt_vocab = ckpt.get("prompt_vocab", {})
    model = MidiLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    prompt_ids = tokenize_prompt(args.prompt, prompt_vocab)
    sep_id = prompt_vocab.get("SEP")
    if sep_id is not None: prompt_ids.append(sep_id)
    ids = generate_tokens(model, prompt_ids, max_new_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.repetition_penalty, device=device)
    toks = decode(ids)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "generated_tokens.txt"), "w") as f: f.write(tokens_to_string(toks))
    midi_path = tokens_to_midi(toks, os.path.join(args.output_dir, "output.mid"))
    if not args.no_audio:
        try: midi_to_mp3(midi_path, os.path.join(args.output_dir, "output.mp3"))
        except: pass

if __name__ == "__main__":
    main()
