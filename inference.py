import torch
import torch.nn as nn

# Hyperparameters
batch_size = 64
max_len = 256 # maximum length of data to be fed at a time for context
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_iters = 5000
eval_interval = 380
lr = 3e-4
eval_iters = 200
embed_size = 384
num_heads = 6
block_layers = 6
dropout = 0.2

path = "Dataset/shakespeare/input.txt"
# read and inspect the shakespeare file

with open(path, 'r', encoding='utf-8') as f:
  text = f.read()

# let's print all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoding and decoding of characters into a sequence of integers correspondign to out vocabulary(character level vocab)
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Let's write the model
class Head(nn.Module):

  def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(embed_size, head_size)
      self.query = nn.Linear(embed_size, head_size)
      self.value = nn.Linear(embed_size, head_size)
      self.dropout = nn.Dropout(dropout)
      self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))

  def forward(self, X):

      # X is of shape (B, T, C)
      B, T, C = X.shape
      k = self.key(X)
      q = self.query(X)

      # compute attention scores
      tril = torch.tril(torch.ones((T, T), device=device))
      wei = q @ k.transpose(-2, -1) / (C ** 0.5)
      wei = torch.masked_fill(wei, tril==0, float('-inf'))
      wei = torch.softmax(wei, dim=-1)
      wei = self.dropout(wei)

      # compute the output
      output = wei @ self.value(X)
      return output

class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

  def forward(self, X):
        # X is of shape (B, T, C)
        # output is of shape (B, T, num_heads * head_size)
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, embed_dim):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(embed_dim, embed_dim),
         nn.ReLU(),
         nn.Linear(embed_dim, embed_dim),
         nn.Dropout(dropout)
      )

    def forward(self, X):
       return self.net(X)

class Block(nn.Module):

  def __init__(self, embed_dim, num_heads):
      super().__init__()
      self.attention = MultiHeadAttention(num_heads, embed_dim//num_heads)
      self.ffwd = FeedForward(embed_dim)
      self.ln1 = nn.LayerNorm(embed_dim)
      self.ln2 = nn.LayerNorm(embed_dim)

  def forward(self, X):
      X = X + self.attention(self.ln1(X))
      X = X + self.ffwd(self.ln2(X))
      return X


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding_table = nn.Embedding(max_len, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, num_heads) for _ in range(block_layers)])
        self.ln = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, X, targets=None):

        B, T = X.shape

        tok_embeds = self.token_embedding_table(X)
        pos_embeds = self.pos_embedding_table(torch.arange(T, device=device))
        X = tok_embeds + pos_embeds
        X = self.blocks(X)
        X = self.ln(X)
        logits = self.linear(X)

        if targets is None:
          loss = None

        else:
          B, T, C = logits.shape
          logits = logits.view(B*T, C)
          targets = targets.view(B*T)
          loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a list of integers
        # max_new_tokens is the maximum number of tokens to generate
        for _ in range(max_new_tokens):

            # get the last max_len tokens
            idx_cond = idx[:, -max_len:]
            # get the predictions
            logits, loss = self(idx_cond)
            # get the last time step
            logits = logits[:, -1, :]
            # apply softmax
            logits = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(logits, num_samples=1)
            # append to the sequence
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx

m = BigramLanguageModel()
m = m.to(device)
m.eval()

# Now let us load up the model
m.load_state_dict(torch.load('shakespeare_model.pth', map_location=device))

# Now we need to define a function to generate text
def generate_text(model, seed, max_len):
    model.eval()
    seed = encode(seed)
    seed = torch.tensor(seed, dtype=torch.long, device=device).unsqueeze(0)
    generated_text = model.generate(seed, max_len)
    generated_text = decode(generated_text[0].tolist())
    return generated_text

print(generate_text(m, "The king said", 100))


