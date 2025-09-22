# serve_api.py
import torch, json
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- must match training ----
BLOCK_SIZE = 128
class ByteTokenizer:
    def encode(self, s: str): return list(s.encode("utf-8"))
    def decode(self, ids): return bytes(ids).decode("utf-8", errors="ignore")
tokenizer = ByteTokenizer()

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None,:,:]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, temperature=0.9, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
    def forward(self, x):
        B, T, C = x.size()
        nh = self.n_head; hs = C // nh
        k = self.key(x).view(B,T,nh,hs).transpose(1,2)
        q = self.query(x).view(B,T,nh,hs).transpose(1,2)
        v = self.value(x).view(B,T,nh,hs).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0 / (hs ** 0.5))
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# load checkpoint
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # portfolio/

ckpt_path = os.path.join(PROJECT_ROOT, "backend/out", "tiny_gpt.pt")
ckpt = torch.load(ckpt_path, map_location=DEVICE)
cfg = ckpt["config"]
model = TinyGPT(cfg["vocab_size"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["block_size"], cfg["dropout"]).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

app = FastAPI()

class ChatIn(BaseModel):
    message: str
    temperature: float = 0.9
    max_new_tokens: int = 128

@app.post("/chat")
def chat(inp: ChatIn):
    prompt = f"<|user|> {inp.message}\n<|bot|>"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    out = model.generate(ids, max_new_tokens=inp.max_new_tokens, temperature=inp.temperature, top_k=50)
    text = tokenizer.decode(out[0].tolist())
    # cut at <|end|> if present
    if "<|end|>" in text:
        text = text.split("<|end|>")[0]
    # return only bot span
    reply = text.split("<|bot|>")[-1].strip()
    return {"reply": reply}
