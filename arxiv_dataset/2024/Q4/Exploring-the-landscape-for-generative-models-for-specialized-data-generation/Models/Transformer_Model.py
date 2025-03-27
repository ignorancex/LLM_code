
import torch
import torch.nn as nn
from torch.nn import functional as F
import random

batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 79 # what is the maximum context length for predictions?
max_iters = 6000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)
#comments 
with open('/home/mohammad/Desktop/mapped_words_modified.txt', 'r', encoding='utf-8') as f:
    text = f.read()  
    #at the beg of each line add . to make it a token 
    
    

# here are all the unique characters that occur in this text
chars = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', 'a1', 'b1', 'c1', 'd1',
    'e1', 'f1', 'g1', 'h1', 'i1', 'j1', 'k1', 'l1', 'm1',
    'n1', 'o1', 'p1', 'q1', 'r1', 's1', 't1', 'u1', 'v1', 'w1'
] 
# create a mapping of character -> index and index -> character 
# this is not necessary but it is good practice 
# as it will allow you to know the index of the prediction 
stoi = {s:i+1 for i,s in enumerate(chars)} 
# add a padding token at index 0   
stoi["."] = 0  
# inverse mapping
itos = {i:s for s,i in stoi.items()} 
# add a padding token at index 0 

# write decoder and encoder functions  
"""encode a string s into a tensor of indices"""
def encode(s):
    l = s.split() 
    
    unknown = random.choice([i for i in range(len(chars))])
    out = [stoi.get(i, unknown) for i in l]
    return out
"""decode a tensor of indices into a string"""
def decode(l):
    return "".join([itos[i] for i in l])

vocab_size = len(stoi)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n] # first 90% 
val_data = data[n:]# last 10%

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # choose the split
    ix = torch.randint(len(data) - block_size, (batch_size,))# choose the starting index of each sequence
    x = torch.stack([data[i:i+block_size] for i in ix])# stack the sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])# the targets are the same as the inputs but one step ahead
    x, y = x.to(device), y.to(device)# move the data to the proper device
    return x, y# return the data

@torch.no_grad() # let's make sure to turn off the gradients for the evaluation
def estimate_loss():# function to estimate the loss on the train and val splits
    out = {}# store the losses here
    model.eval()# put the model in evaluation mode
    for split in ['train', 'val']:# iterate over both splits
        losses = torch.zeros(eval_iters)# store the losses here
        for k in range(eval_iters):# iterate over the number of iterations
            X, Y = get_batch(split)# get a batch
            logits, loss = model(X, Y)# forward the data in the model
            losses[k] = loss.item()# store the loss
        out[split] = losses.mean()# store the mean loss for the split
    model.train()# put the model back in training mode
    return out# return the losses

class Head(nn.Module):
    """ single head of self-attention """ 
    """This is the core of the transformer model. It computes the attention scores between each token in the sequence and aggregates the values accordingly."""
    """key query and value are linear transformations of the input embeddings. The attention scores are computed as the dot product between the queries and keys, scaled by the square root of the dimensionality of the keys. The values are then aggregated according to the attention scores. The output of the head is the weighted sum of the values."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ a multi-head attention module """ 
    """The multi-head attention module is a collection of heads that operate in parallel. The outputs of the heads are concatenated and projected back to the original dimensionality.""" 
    """This allows the model to attend to different parts of the input sequence in parallel, which is useful for capturing different types of dependencies."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ simple position-wise feed-forward module"""  
    """The feed-forward module is a simple two-layer neural network that is applied to each position in the sequence independently. It is used to capture complex dependencies between tokens that are not captured by the self-attention mechanism.""" 

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
    """ a transformer block """ 
    """The transformer block is the core building block of the transformer model. It consists of a multi-head attention module followed by a position-wise feed-forward module. The output of the block is the sum of the input and the output of the feed-forward module, followed by layer normalization.""" 
    """The block is applied multiple times in the model to capture dependencies at different scales."""

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

# super simple bigram model
class BigramLanguageModel(nn.Module):  
    """ a simple bigram language model""" 
    """The bigram language model is a simple model that predicts the next token in the sequence based on the previous token. It consists of an embedding layer, a position embedding layer, a stack of transformer blocks, a final layer normalization, and a linear layer that predicts the logits for the next token.""" 
    """The model is trained to minimize the cross-entropy loss between the predicted logits and the true targets.""" 
    

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

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

    def generate(self, idx, num_sequences, tokens_per_sequence):
        sequences = []
        for _ in range(num_sequences):
            out = idx
            while len(out[0]) < tokens_per_sequence:
                idx_cond = out[:, -block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                out = torch.cat((out, idx_next), dim=1)
            sequences.append(out[0, -tokens_per_sequence:].tolist())
        return sequences

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
sequences = model.generate(context, num_sequences=100, tokens_per_sequence=79)

# Print the generated sequences in the desired format
for seq in sequences:
    print(decode(seq))