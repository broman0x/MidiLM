import os
import torch
import numpy as np
from gguf import GGUFWriter
from pathlib import Path
import sys
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model.model import MidiLM, MidiLMConfig

def export_gguf(checkpoint_path, output_path, tokenizer_path="tokenizer.json"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    config_dict = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]
    n_embd = config_dict.get("hidden_size", 768)
    n_layer = config_dict.get("num_hidden_layers", 12)
    n_head = config_dict.get("num_attention_heads", 12)
    n_inner = config_dict.get("intermediate_size", 3072)
    n_ctx = 2048 
    writer = GGUFWriter(output_path, "llama")   
    writer.add_name("MidiLM-Chat")
    writer.add_context_length(n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_inner)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_bos_token_id(tokenizer.token_to_id("[BOS]"))
    writer.add_eos_token_id(tokenizer.token_to_id("[EOS]"))
    writer.add_pad_token_id(tokenizer.token_to_id("[PAD]"))
    tokens = []
    scores = []
    toktypes = []
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    for token, tid in sorted_vocab:
        tokens.append(token.encode("utf-8"))
        scores.append(0.0)
        if any(x in token for x in ["NOTE_", "TIME_", "VELOCITY_", "INST_", "TEMPO_"]):
            toktypes.append(4)
        else:
            toktypes.append(1)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    writer.add_tensor("token_embd.weight", state_dict["model.embed_tokens.weight"].numpy())
    writer.add_tensor("output_norm.weight", state_dict["model.norm.weight"].numpy())
    writer.add_tensor("output.weight", state_dict["lm_head.weight"].numpy())
    for i in range(n_layer):
        prefix = f"model.layers.{i}."
        gguf_prefix = f"blk.{i}."
        writer.add_tensor(f"{gguf_prefix}attn_q.weight", state_dict[f"{prefix}self_attn.q_proj.weight"].numpy())
        writer.add_tensor(f"{gguf_prefix}attn_k.weight", state_dict[f"{prefix}self_attn.k_proj.weight"].numpy())
        writer.add_tensor(f"{gguf_prefix}attn_v.weight", state_dict[f"{prefix}self_attn.v_proj.weight"].numpy())
        writer.add_tensor(f"{gguf_prefix}attn_output.weight", state_dict[f"{prefix}self_attn.o_proj.weight"].numpy())
        writer.add_tensor(f"{gguf_prefix}attn_norm.weight", state_dict[f"{prefix}input_layernorm.weight"].numpy())
        writer.add_tensor(f"{gguf_prefix}ffn_norm.weight", state_dict[f"{prefix}post_attention_layernorm.weight"].numpy())
        writer.add_tensor(f"{gguf_prefix}ffn_gate.weight", state_dict[f"{prefix}mlp.gate_proj.weight"].numpy())
        writer.add_tensor(f"{gguf_prefix}ffn_up.weight", state_dict[f"{prefix}mlp.up_proj.weight"].numpy())
        writer.add_tensor(f"{gguf_prefix}ffn_down.weight", state_dict[f"{prefix}mlp.down_proj.weight"].numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

if __name__ == "__main__":
    export_gguf("checkpoints/best.pt", "checkpoints/midilm-f16.gguf")
