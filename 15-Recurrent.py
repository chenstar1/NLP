'''Integration of token and position embedding and cyclic memory and converter layers,Efficiently handles long sequences'''
import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)

# Hyperparameters
batch_size = 16
block_size = 256  
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
memory_size =16  


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


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, eval_iters, batch_size, block_size):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size)
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
        losses.append(loss.item())
    model.train()
    return torch.tensor(losses).mean().item()
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

class RecurrentMemory(nn.Module):
    def __init__(self, n_embd, memory_size):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(memory_size, n_embd), requires_grad=True)

    def forward(self, x, previous_memory):
        combined = torch.cat([previous_memory, x], dim=1)
        new_memory = self.update_memory(combined)
        return new_memory

    def update_memory(self, x):
        return x.mean(dim=1)
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class SegmentProcessor(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, vocab_size, memory_size):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head) for _ in range(n_layer)])
        self.memory_module = RecurrentMemory(n_embd, memory_size)
        self.final_layer = nn.Linear(n_embd, vocab_size)

    def forward(self, x, memory):
        x = torch.cat([memory, x], dim=1)  # Concatenate memory at the input
        for layer in self.layers:
            x = layer(x)
        memory = self.memory_module(x, memory)
        logits = self.final_layer(x)
        return logits, memory

class RMTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, memory_size, device):
        super().__init__()
        self.segment_processor = SegmentProcessor(n_embd, n_head, n_layer, vocab_size, memory_size)
        self.memory_size = memory_size
        self.n_embd = n_embd
        self.device = device
        self.initial_memory = torch.zeros(1, memory_size, n_embd, device=device)  # Initial memory state

    def forward(self, segments):
        memory = self.initial_memory
        all_logits = []
        for segment in segments:
            logits, memory = self.segment_processor(segment.to(self.device), memory)
            all_logits.append(logits)
        return torch.cat(all_logits, dim=0), memory
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = self.position_embedding[:, :tokens.size(1), :]
        x = tokens + positions
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        logits = self.head(x)
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



model = GPTLanguageModel().to(device)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for iter in range(max_iters):
    X, Y = get_batch(train_data, block_size, batch_size)
    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad()
    logits = model(X)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        train_loss = estimate_loss(model, train_data, eval_iters, batch_size, block_size)
        val_loss = estimate_loss(model, val_data, eval_iters, batch_size, block_size)
        print(f"Step {iter}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")



# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
