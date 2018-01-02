import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers,hidden_size,emd_martix):
        super(RNNLM, self).__init__()
        #self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.emd_martix=emd_martix
     
    def init_weights(self):
        #self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
   
        
    def forward(self, inp, hidden):
        # Embed word ids to vectors
        #inp = self.embed(inp) 
        inp=F.embedding(inp,self.emd_martix)
        # Forward propagate RNN  
        out, hidden = self.lstm(inp, hidden)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        # Decode hidden states of all time step
        # out = self.linear(out)  
        return out, hidden
    
    
    
    