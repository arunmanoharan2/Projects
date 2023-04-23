#!/usr/bin/env python
# coding: utf-8

# In[36]:


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


words = open('C:\\Users\\Arun Manoharan\\makemore\\names.txt', 'r').read().splitlines()
print('Nano Name Generator - NNG')
print('Overview:')
print('----------')
print('NNG is a generative AI project that learns the statistical probabilities ')
print('of next character given the first character. ')
print('NNG learns statistical probabilities from a input files with child names. ')
print('NNG uses multinomial sampling to generate first letter of new name and ')
print('generates next characters based on probability it learnt from input child names')
print('----------')
print('Input = Text file with child names')
print('Output = Generated names through bigram prediction')
print('Output = Loss function or Maximum likelihood estimation')
print('----------')
print("Number of names in the input file =", len(words))
print('First 5 names in the input file', words[:5])


# In[3]:


# Add Start and End delimiters
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs,chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1



# In[4]:


N = torch.zeros((27,27), dtype=torch.int32)


# In[5]:


# Sort and convert to integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


# In[6]:


# Create Tuples for bigrams.
# In bigram table, Rows are input character and colums are output character 
# and values of the array is the likelihood of column character is the next characters given row character as input
# This bigram tensor stores the statistical distribution/probability of next char given the 
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 


# In[41]:


print('---------------------------------------------------------------')
print('Bigram table based on learning from input child names')
print('---------------------------------------------------------------')
print('Row = Input character')
print('Column = Output character')
print('Output value of array = Likelihood count - Not Normalized')
print('---------------------------------------------------------------')


# In[7]:


# Plot for visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');


# In[27]:


# convert entire tensor to float and normalize
p = N.float()
p = p/p.sum()


# In[23]:


# Smoothing function
# P is 27, 27
# P.sum() is 27, 1
P = (N+1).float()
P /= P.sum(1, keepdim=True)


# In[37]:


# Create New names by sampling
g = torch.Generator().manual_seed(2147483647)
print('OUTPUT')
print('------')
print("Create 20 new names based on bigram probabilites")
print('')
for i in range(20):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() 
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))
        


# In[38]:


# Maximum likelihood estimation
log_likelihood = 0.0
n = 0
#for w in words[:3]:
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2}: {prob:.4f} {logprob:1.4f}' )

#print(f'{log_likelihood=}')
nll = -log_likelihood
#print(f'{nll=}')
# nll/n
output = (nll/n).item()
print('--------------------------------------------------------------------------')
print('Loss function - Maximum likelihood estimation = ', "{:.4f}".format(output))
print('--------------------------------------------------------------------------')

