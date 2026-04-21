import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MidiLMConfig:
    def __init__(self, vocab_size=16384, n_embd=768, n_layer=12, n_head=12, n_inner=None, 
                 resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-5, 
                 initializer_range=0.02, max_position_embeddings=2048, rope_theta=10000.0):
        self.vocab_size = vocab_size
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.intermediate_size = n_inner if n_inner is not None else 4 * n_embd
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.rms_norm_eps = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

    @classmethod
    def from_dict(cls, d):
        config = cls()
        for k, v in d.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif k == "num_hidden_layers": config.num_hidden_layers = v
            elif k == "num_attention_heads": config.num_attention_heads = v
            elif k == "hidden_size": config.hidden_size = v
            elif k == "intermediate_size": config.intermediate_size = v
        return config

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"]
        }

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask[:, :, :seqlen, :seqlen]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, freqs_cis, mask=None):
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class MidiLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_position_embeddings, config.rope_theta)

    def forward(self, input_ids, targets=None):
        bsz, seqlen = input_ids.shape
        h = self.model.embed_tokens(input_ids)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.view(1, 1, seqlen, seqlen)
        for layer in self.model.layers:
            h = layer(h, freqs_cis, mask)
        h = self.model.norm(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, None

if __name__ == "__main__":
    config = MidiLMConfig(vocab_size=1000, n_layer=2, n_head=4, n_embd=128)
    model = MidiLM(config)
    x = torch.randint(0, 1000, (1, 10))
    logits, loss, _ = model(x, targets=x)
    print(f"DONE: {logits.shape}")
