import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import gens

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=1
        ).bool()

        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        d_model=256,
        n_heads=4,
        n_layers=4,
        ff_dim=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_emb.weight

        self.seq_len = seq_len

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.seq_len

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

def train_step(model, optimizer, batch, device):
    model.train()
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.seq_len:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

    return idx

def get_vocabulary(by: str="download") -> tuple[typing.Any, int]:
    if by == "download":
        model = api.load('glove-twitter-100', return_path=False)
        return model, model.vector_size
    elif by == "preload":
        model_path = api.load('glove-twitter-100', return_path=True)

        model = KeyedVectors.load_word2vec_format(
            model_path,
            binary=False,
            # no_header=True
        )
        return model, model.vector_size
    else:
        return None, None

vocab_size = 10000
seq_len = 128

model = GPT(
    vocab_size=vocab_size,
    seq_len=seq_len,
    d_model=256,
    n_heads=4,
    n_layers=4,
)

device = torch.device("cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)