import torch
import torch.nn as nn

B, T, C = 4, 8, 10
X = torch.random.randn(B, T, C)

head_size = 16

key = nn.Linear(C, head_size)
query = nn.Linear(C, head_size)

value = nn.Linear(C, head_size)

K = key(X)
Q = query(X)

wei = Q @ K.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
wei = torch.masked_fill(wei, tril==0, float('-inf'))
wei = torch.softmax(wei, dim=-1)

V = value(X)
output = V @ wei

print(K.shape, Q.shape, V.shape)

