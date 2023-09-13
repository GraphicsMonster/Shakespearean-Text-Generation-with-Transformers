import torch
import torch.nn as nn

# Hyperparameters
batch_size = 32
max_len = 8 # maximum length of data to be fed at a time for context
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_iters = 3800
eval_interval = 380
lr = 1e-3
eval_iters = 100

path = "Dataset/shakespeare/input.txt"
# read and inspect the shakespeare file
with open(path, 'r', encoding='utf-8') as f:
  text = f.read()

# print the number of characters in the file
print("length of the entire thing: ", len(text))

# let's print all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("here are the unique characters in the entire dataset: ", chars)
print("the vocab size: ", vocab_size)

# encoding and decoding of characters into a sequence of integers correspondign to out vocabulary(character level vocab)
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('yo bro what is up'))
print(decode(encode('yo bro what is up')))

import torch

# Let's encode all the text in the tinyshakespeare dataset
data = torch.tensor(encode(text), dtype=torch.long)

# train-test split
n = int(0.9 * len(data))
train_set = data[:n]
val_set = data[n:]


# define a system to pull out a sequence of max length at random from the entire dataset
batch_size = 4
max_len = 8 # maximum length of data to be fed at a time for context

def get_batch(split):
  data = train_set if (split=='train') else val_set
  ix = torch.randint(len(data) - max_len, (batch_size, ))
  x = torch.stack([data[i:i+max_len] for i in ix])
  y = torch.stack([data[i+1:i+max_len+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
   out = {}
   m.eval()
   for split in ['train', 'val']:
      losses = torch.zeros(eval_iters, device=device)
      for k in range(eval_iters):
         X, Y = get_batch(split)
         logits, loss = m(X, Y)
         losses[k] = loss.item()
      out[split] = losses.mean().item()
    
   m.train()
   return out

# Now that we have the data ready let's build the model

class BigramLanguageModel(nn.Module):
  
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):

        loss=None

        logits = self.token_embedding_table(x)

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
            # get the predictions
            logits, loss = self(idx)
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

# optmizer
optimizer = torch.optim.Adam(m.parameters(), lr=lr)

# training loop

for iter in range(num_iters):
   
   # every once in a while evaluate the loss on train and val sets
  if iter%eval_interval == 0:
      losses = estimate_loss()
      print(f"iter {iter} train loss: {losses['train']:.2f} val loss: {losses['val']:.2f}")
    
    # get a batch of data
  xb, yb = get_batch('train')
  
  # evaluate the loss

  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# generate some text
# generate some text
generated_text_indices = m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=200)
generated_text = decode(generated_text_indices[0].tolist())
print(generated_text)
