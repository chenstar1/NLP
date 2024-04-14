'''Introduction of PCA downscaling and orthogonal transformations'''
import torch
import torch.nn as nn
from torch.nn import functional as F



# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
pca_dim_reduction=0.8
weight_slice_interval= 100
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('Poetry.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
def apply_weight_slice(module, keep_ratio=0.8):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() == 2:
                u, _, v = torch.svd(param.data, some=True)
                k = int(param.size(1) * keep_ratio)
                reduced_weight = torch.mm(u[:, :k], v[:, :k].t())
                param.data = reduced_weight
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale
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
def prune_weights(param, compression_rate=0.1):

    with torch.no_grad():
        weight_abs = torch.abs(param.data)
        threshold = torch.quantile(weight_abs, compression_rate)
        mask = weight_abs.ge(threshold).float()
        param.data.mul_(mask)
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

class OrthogonalTransform(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(n_embd))

    def forward(self, x):
        # Apply SVD to ensure the weight matrix is orthogonal
        u, _, v = torch.svd(self.weight.data)
        self.weight.data = torch.mm(u, v.t())
        return x @ self.weight

class ModifiedPCAFeedForward(nn.Module):
    def __init__(self, n_embd, pca_dim,compression_rate=0.1, pca_dim_reduction=0.8):
        super().__init__()
        self.compression_rate = compression_rate
        pca_dim = int(n_embd * pca_dim_reduction)
        self.linear1 = nn.Linear(n_embd, pca_dim)
        self.linear2 = nn.Linear(pca_dim, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    def prune_weights(self):
        prune_weights(self.linear1.weight, self.compression_rate)
        prune_weights(self.linear2.weight, self.compression_rate)




class ModifiedTransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, pca_dim):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, n_embd // n_head)
        self.ln2 = RMSNorm(n_embd)
        self.ff = ModifiedPCAFeedForward(n_embd, pca_dim)
        self.orthogonal_transform_input = OrthogonalTransform(n_embd)
        self.orthogonal_transform_output = OrthogonalTransform(n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = self.attn(x) + x
        x = self.ln2(x)
      
        x = self.orthogonal_transform_input(x)
        x = self.ff(x)
        x = self.orthogonal_transform_output(x) + x
        return x

class ModifiedGPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = len(chars)  
        pca_dim = int(n_embd * pca_dim_reduction)
        self.blocks = nn.Sequential(*[ModifiedTransformerBlock(n_embd, n_head, pca_dim) for _ in range(n_layer)])

        self.token_embedding_table = nn.Embedding(self.vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.vocab_size)

        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

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

chars = sorted(list(set(text)))
model = ModifiedGPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % weight_slice_interval == 0:
    
        for block in model.blocks:
            apply_weight_slice(block.ff)

        losses = estimate_loss()  
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))



