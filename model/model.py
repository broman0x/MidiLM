import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MidiLMConfig:
    def __init__(
        self,
        vocab_size=2500,
        max_seq_len=1024,
        n_layers=8,
        n_heads=12,
        d_model=768,
        d_ff=3072,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, offset=0):
        return x + self.pe[:, offset : offset + x.size(1), :]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0))

    def forward(self, x, layer_past=None, use_cache=False):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if layer_past is not None:
            k = torch.cat((layer_past[0], k), dim=-2)
            v = torch.cat((layer_past[1], v), dim=-2)
        present = (k, v) if use_cache else None
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if layer_past is None:
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = (self.attn_drop(attn) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y)), present

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, layer_past=None, use_cache=False):
        a, p = self.attn(self.ln1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, p

class MidiLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_len)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout, config.max_seq_len) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if hasattr(module, "bias") and module.bias is not None: nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None, past_key_values=None, use_cache=False):
        B, T = input_ids.size()
        offset = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        x = self.drop(self.pos_enc(self.token_emb(input_ids), offset=offset))
        presents = []
        for i, block in enumerate(self.blocks):
            x, p = block(x, layer_past=past_key_values[i] if past_key_values is not None else None, use_cache=use_cache)
            if use_cache: presents.append(p)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.pad_token_id) if targets is not None else None
        return logits, loss, presents

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
