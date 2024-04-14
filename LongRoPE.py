
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast  
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import math
# hyperparameters
batch_size = 2
block_size = 256
max_iters = 3000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Load your data here
with open('Poetry.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, context_length=block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    total_loss = 0
    for _ in range(eval_iters):
        X, Y = get_batch('val')
        with autocast():
            logits, loss = model(X, Y)
        total_loss += loss.item()
    model.train()
    return total_loss / eval_iters
class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(DynamicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('position', torch.arange(max_len).unsqueeze(1))

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.register_parameter('div_term', nn.Parameter(div_term, requires_grad=True))

    def forward(self, x):
        d_model = x.size(-1)
        pe = torch.zeros(x.size(0), 1, d_model, device=x.device)
        pe[:, 0, 0::2] = torch.sin(self.position[:x.size(0)] * self.div_term)
        pe[:, 0, 1::2] = torch.cos(self.position[:x.size(0)] * self.div_term)

        x = x + pe
        return self.dropout(x)
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head)
        self.feed_forward = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.dropout(self.feed_forward(self.ln2(x)))
        return x
class NonUniformPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(NonUniformPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        self.register_buffer('position', position)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.register_parameter('div_term', nn.Parameter(div_term, requires_grad=True))


        self.scale = nn.Parameter(torch.ones(max_len, 1))

    def forward(self, x):
        d_model = x.size(-1)
        pe = torch.zeros(x.size(0), 1, d_model, device=x.device)


        scaled_position = self.position[:x.size(0)] * self.scale[:x.size(0)]


        pe[:, 0, 0::2] = torch.sin(scaled_position * self.div_term[0: d_model // 2])
        pe[:, 0, 1::2] = torch.cos(scaled_position * self.div_term[0: d_model // 2])

        x = x + pe
        return self.dropout(x)
class LongRoPETransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = NonUniformPositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        self.position_embedding = DynamicPositionalEncoding(n_embd, dropout=0.1, max_len=block_size)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        b, t = idx.size()
        token_embeddings = self.token_embedding(idx)  # Shape: (batch, block_size, n_embd)
        position_embeddings = self.position_embedding(token_embeddings)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            return logits, loss
        else:
            return logits
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Instantiate model, optimizer, loss scaler for mixed precision, and scheduler
model = GPTLanguageModel().to(device)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler()
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# Training loop
context_lengths = [128, 256, 512, 1024]  
for context_length in context_lengths:
    print(f"Training with context length: {context_length}")
    for iter in range(max_iters):
        xb, yb = get_batch('train', context_length)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            output = model(xb)
            loss = F.cross_entropy(output.view(-1, vocab_size), yb.view(-1))

      
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if iter % eval_interval == 0 or iter == max_iters - 1:
            val_loss = estimate_loss(model)
            print(f"step {iter}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")
            scheduler.step(val_loss)
# Example of generating text
context = torch.tensor([encode('Hello')], dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
