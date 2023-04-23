#!/usr/bin/env python
# coding: utf-8

# In[62]:


import torch.nn.functional as F
import torch

# Nano Name Generator - Bigram - Statistical Model
# Author: Arun Shankar Manoharan
# Date: 4/21/23

# Read the input text document
# Each line has a name and there are 32000 names
# Below are the first 5 names
# emma
# olivia
# ava
# isabella
# sophia

# Read input file
words = open('C:\\Users\\Arun Manoharan\\makemore\\names.txt', 'r').read().splitlines()
print('Nano Name Generator - NNG - Neural Network')
print('Overview:')
print('----------')
print('NNG is a generative AI project that learns of next character of the sequence given the previous character.')
print('This project uses neural network to achieve a comparable loss wrt to statistical method.')
print('----------')
print('Input = Text file with child names')
print('Output = Generated names using neural nets')
print('Output = Loss function or Maximum likelihood estimation')
print('----------')

print("Number of names in the input file =", len(words))
print('First 5 names in the input file', words[:5])


# In[52]:


#### STAGE 2
#### Now we are going to use and train neural nets to predict next character
####
#### Create dataset

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
        
xs = torch.tensor(xs)
ys = torch.tensor(ys)  
num = xs.nelement()
print('Number of bigram examples', num)

# Randomly initialize weights of 27 neurons. Each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)


# In[25]:





# In[94]:


# Gradient descent and optimize the log loss to same level as statistical method
print('---------------------------------------------------------------')
print(' Start gradient descent and update neural network weights')
print('---------------------------------------------------------------')

for k in range(1):

    # Forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W # log counts
    ## softmax - Normalization
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    ## softmax
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    print('--------------------------------------------------------------------------')
    print('Loss function - Maximum likelihood estimation = ', "{:.4f}".format(loss.item()))
    print('--------------------------------------------------------------------------')
    # Backward pass
    W.grad = None
    loss.backward()

    #update weights 
    W.data += -50 * W.grad


# In[104]:


# Name generator
g = torch.Generator().manual_seed(2147483647)
print('OUTPUT')
print('------')
print("Create 25 new names based on bigram probabilites")
print('')

for i in range(25):
    out = []
    ix = 0
    
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)    #Probability of next character

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break
    
    print(''.join(out))
    


# In[ ]:




