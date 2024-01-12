import torch
import mmap
import random 

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(device)

chars = ""
with open("shakespeare.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
        
vocab_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [ string_to_int[c] for c in s ]
decode = lambda l: ''.join([int_to_string[i] for i in l])

import torch.nn.functional as F
import torch.nn as nn

lr = 0.002

block_size = 64
batch_size = 128
max_iters = 1000
n_embd = 384
voacb_size = len(chars)
n_layer = 4
n_head=4
dropout = 0.2
eval_iters = 100

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
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embd_size)
        self.position_embedding_table = nn.Embedding(block_size, embd_size)

        self.blocks = nn.Sequential(*[Block(embd_size, n_head=n_head) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(embd_size)
        self.lm_head = nn.Linear(embd_size, voacb_size)

        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if(module.bias is not None):
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

    def forward(self, index, targets=None):

        B, T = index.shape
        
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x=tok_emb+pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # print(logits.shape)

        if(targets is None):
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            
            targets = targets.view(B*T)

            # print(logits.shape, targets.shape)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):

            # print(index.shape)
            tp = index
            if(tp.shape[1]>block_size):
                tp = tp[:, -64:]
            logits, loss = self.forward(tp)
            # print(logits.shape)
            # print()
            logits = logits[:, -1, :]


            probs = F.softmax(logits, dim=-1)

            index_next = torch.multinomial(probs, num_samples=1)

            index = torch.cat((index, index_next), dim=1)

            # print(index.shape)

        return index
    

# model = GPTModel(voacb_size, n_embd)
model = torch.load("./pretrained/shakes_1_20.pt")
model.to(device)

# model.load_state_dict(torch.load('model_weights_best.pth'))

input1 = ""
while input1!="exit":
    print("\nEnter the prompt ('exit': to exit chat): ")
    input1 = input()

    tp1 = encode(input1)
    context = torch.LongTensor([tp1]).to(device)
    generated_chars = decode(model.generate(context, max_new_tokens=250)[0].tolist())

    print("\nOutput: \n", generated_chars,"\n")