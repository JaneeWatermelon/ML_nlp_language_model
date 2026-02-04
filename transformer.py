import math
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from vocab import Vocab
from nltk import WordPunctTokenizer
import vars
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

from core.helpers import char_noise, load_quora_dataset, load_arxiv_dataset, load_dungeon_dataset, load_ssau_dataset

class AttentionModule(nn.Module):
    def __init__(self, d_model, n_heads, dropout_p: float=0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"Размерность токена ({d_model}) не кратна кол-ву голов ({n_heads})"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # dim = (B, T, 3*C)
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        # dim = (B, T, C)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # dim = (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # dim = (B, n_heads, T, T)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        attention_mask = torch.triu(
            torch.ones((T, T), device=x.device),
            diagonal=1
        ).bool()

        att = att.masked_fill(attention_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # dim = (B, n_heads, T, head_dim)
        out = att @ v
        # dim = (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout_p, ff_dim):
        super().__init__()

        self.ln_1 = nn.LayerNorm(d_model)
        self.att_layer = AttentionModule(d_model, n_heads, dropout_p)

        self.processing_layer = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout_p),
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.att_layer(self.ln_1(x))
        x = x + self.processing_layer(self.ln_2(x))
        return x
    
class LM(nn.Module):
    def __init__(self, seq_len, n_layers, voc_size, d_model, n_heads, dropout_p, ff_dim):
        super().__init__()

        self.vocab_emb = nn.Embedding(voc_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout_p, ff_dim)
            for _ in range(n_layers)
        ])

        self.l_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, voc_size, bias=False)

        self.lm_head.weight = self.vocab_emb.weight

        self.seq_len = seq_len

    def forward(self, idx: torch.Tensor, targets: torch.Tensor=None):
        B, T = idx.shape

        assert T <= self.seq_len, f"Длина контекста ({T}) превышает максимум ({self.seq_len})"

        pos = torch.arange(T, device=idx.device)

        x = self.vocab_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.l_norm(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
        
class LMDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        # сколько возможных окон
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

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

def sample_top_p(probs: torch.Tensor, p: float = 0.8) -> torch.Tensor:
    """
    probs: (vocab_size,) — вероятности (после softmax)
    возвращает: scalar tensor — индекс токена
    """
    assert 0 < p <= 1.0

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=0)

    # минимальное множество с суммой >= p
    cutoff = torch.searchsorted(cum_probs, p, right=True)
    cutoff = max(cutoff.item(), 1)

    top_idx = sorted_idx[:cutoff]
    top_probs = sorted_probs[:cutoff]
    top_probs = top_probs / top_probs.sum()

    sampled = torch.multinomial(top_probs, num_samples=1)
    return top_idx[sampled]

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.seq_len:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)
        next_token = sample_top_p(probs[0], p=0.8)
        next_token = next_token.unsqueeze(0)

        idx = torch.cat([idx, next_token], dim=1)

    return idx

@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, max_batches=None):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for i, (x, y) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        # loss — средний по (B*T)
        B, T = y.shape
        total_loss += loss.item()
        total_tokens += B * T

    avg_loss = total_loss * B * T / total_tokens
    ppl = math.exp(avg_loss)

    return ppl, avg_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    # train_text, val_text = load_arxiv_dataset(vars.DATASETS_ROOT)
    # full_text = " ".join(train_text)
    # full_text = load_dungeon_dataset(vars.DATASETS_ROOT)
    full_text = load_ssau_dataset()

    tokenizer = WordPunctTokenizer()

    def pretokenize(text: str) -> str:
        return " ".join(tokenizer.tokenize(text.lower()))

    full_text = pretokenize(full_text)

    augmented = []
    for msg in full_text.split():
        if random.random() < 0.5:
            augmented.append(char_noise(msg, p=0.02))
        else:
            augmented.append(msg)

    full_text = " ".join(augmented)

    # --------------------------------------------------
    # 2. Learn + apply BPE
    # --------------------------------------------------
    text_for_bpe_path = os.path.join(vars.DATASETS_ROOT, "train.txt")
    bpe_rules_path = os.path.join(vars.DATASETS_ROOT, "bpe_rules")
    if not os.path.exists(bpe_rules_path):
        with open(text_for_bpe_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        with open(text_for_bpe_path, "r", encoding="utf-8") as f_in, \
            open(bpe_rules_path, "w", encoding="utf-8") as f_out:
            learn_bpe(f_in, f_out, num_symbols=16000)

    bpe = BPE(open(bpe_rules_path, encoding="utf-8"))
    bpe_text = bpe.process_line(full_text)
    bpe_tokens = bpe_text.split()

    # --------------------------------------------------
    # 3. Build vocab
    # --------------------------------------------------
    inp_voc = Vocab(bpe_tokens)
    vocab_size = len(inp_voc)

    print("vocab size:", vocab_size)

    eos_id = inp_voc.word2index(vars.EOS)

    # --------------------------------------------------
    # 4. Encode full corpus
    # --------------------------------------------------
    data = torch.tensor(
        [inp_voc.word2index(token) for token in bpe_tokens],
        # dtype=torch.int
    )
    print(data[-1])

    # --------------------------------------------------
    # 5. Dataset / DataLoader
    # --------------------------------------------------
    seq_len = 128
    batch_size = 16
    d_model = 128

    dataset = LMDataset(data, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # --------------------------------------------------
    # 6. Model
    # --------------------------------------------------
    model = LM(
        seq_len=seq_len,
        n_layers=4,
        voc_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        dropout_p=0.1,
        ff_dim=1024,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    if not os.path.exists(os.path.join(vars.MODELS_ROOT, "ssau_model.pth")):
        # --------------------------------------------------
        # 7. Training
        # --------------------------------------------------
        print(f"Кол-во примеров длиной {seq_len} = {len(loader)}")
        for step, (x, y) in enumerate(loader):
            if step > 2000:
                break

            loss = train_step(model, optimizer, (x, y), device)

            if step % 200 == 0:
                print(f"step {step}, loss {loss:.4f}")

        torch.save(model.state_dict(), os.path.join(vars.MODELS_ROOT, "ssau_model.pth"))
    else:
        state_dict = torch.load(os.path.join(vars.MODELS_ROOT, "ssau_model.pth"), weights_only=True)
        model.load_state_dict(state_dict)

    ppl, avg_loss = evaluate_perplexity(model, loader, device, max_batches=200)
    print(f"Perplexity: {ppl:.2f}, avg loss: {avg_loss:.4f}")


    # --------------------------------------------------
    # 8. Generation
    # --------------------------------------------------
    def decode_bpe(ids: torch.Tensor) -> str:
        tokens = [inp_voc.index2word(id) for id in ids.tolist()]
        text = " ".join(tokens)
        return text.replace("@@ ", "").replace("@@", "")

    prompt = "Награждение"
    prompt = pretokenize(prompt)
    prompt = bpe.process_line(prompt).split()

    start_ids = torch.tensor(
        [inp_voc.word2index(token) for token in prompt],
        # dtype=torch.long
    ).unsqueeze(0).to(device)

    out = generate(
        model,
        start_ids,
        max_new_tokens=1000,
        temperature=1.0,
    )

    print("\n=== GENERATED ===")
    print(decode_bpe(out[0]))
