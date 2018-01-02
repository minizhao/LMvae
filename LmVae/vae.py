import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable
import sys
import torch.nn.functional as F

 # Encoder
class Encoder(nn.Module):
    def __init__(self, batch_size,embedding_dim, hidden_dim,emb_martix, dropout=0.2):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.emb_martix=emb_martix
        self.embedding_dim = embedding_dim
        #self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.EncoderNet = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=3, bidirectional=True, \
            dropout=dropout)
        self.hidden_to_z=nn.Linear(2*3*self.hidden_dim, self.hidden_dim)
        
        # self.Encoder2hidden = nn.Linear(2*3*self.hidden_dim, self.hidden_dim)
        # self.dropout_linear = nn.Dropout(p=dropout)
        # self.Encoder2out = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, inp, hidden,num_samples=0):
        
        emb=F.embedding(inp,self.emb_martix)
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _,h_out = self.EncoderNet(emb, hidden)
        h_out=h_out.permute(1,0,2).contiguous()  
        if  num_samples==0:
            h_out=h_out.view(self.batch_size,-1)
        else:
            h_out=h_out.view(num_samples,-1)
        h_z=self.hidden_to_z(h_out)

        return h_z            #6 x batch_size x z_dim

    def init_hidden(self, batch_size=1):
        h1 = autograd.Variable(torch.randn(6, batch_size, self.hidden_dim))
        return (h1.cuda())

class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=None):
        """
        Attention mechanism
        :param enc_dim: Dimension of hidden states of the encoder h_j
        :param dec_dim: Dimension of the hidden states of the decoder s_{i-1}
        :param dec_dim: Dimension of the internal dimension (default: same as decoder).
        """
        super(Attention, self).__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = self.dec_dim if attn_dim is None else attn_dim

        # W_h h_j
        self.encoder_in = nn.Linear(self.enc_dim, self.attn_dim, bias=False)
        self.decoder_in = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.att_linear = nn.Linear(self.attn_dim, 1, bias=False)

    def forward(self, dec_state, context, mask=None):
        """
        :param dec_state:  batch x dec_dim
        :param context: batch x T x enc_dim
        :return: Weighted context, batch x enc_dim
                 Alpha weights (viz), batch x T
        """
        batch, source_l, enc_dim = context.size()

        assert enc_dim == self.enc_dim

        # W*s over the entire batch (batch, attn_dim)
        dec_contrib = self.decoder_in(dec_state)
        
        # W*h over the entire length & batch (batch, source_l, attn_dim)
        enc_contribs = self.encoder_in(
            context.view(-1, self.enc_dim)).view(batch, source_l, self.attn_dim)


        # tanh( Wh*hj + Ws s_{i-1} )     (batch, source_l, dim)
        pre_attn = F.tanh(enc_contribs + dec_contrib.unsqueeze(1).expand_as(enc_contribs))

        # v^T*pre_attn for all batches/lengths (batch, source_l)
        energy = self.att_linear(pre_attn.view(-1, self.attn_dim)).view(batch, source_l)

        alpha = F.softmax(energy)
      
        weighted_context = torch.bmm(alpha.unsqueeze(1), context).squeeze(1)  # (batch, dim)

        return weighted_context, alpha


# Decoder解码器
class Decoder(nn.Module):
    def __init__(self,batch_size, embedding_dim, hidden_dim, vocab_size,emb_martix,\
                    start_letter,rnnlm,dropout=0.2):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.emb_martix=emb_martix
        self.start_letter=start_letter
        self.rnnlm=rnnlm
        
        #gen context vector
        self.EnccontextNet=nn.GRU(self.embedding_dim*2, self.hidden_dim, num_layers=3, bidirectional=True, \
            dropout=dropout)

        self.attn = Attention(self.hidden_dim*3, self.hidden_dim,attn_dim=256)

        self.DecoderNet = nn.GRU(self.embedding_dim*4, self.hidden_dim, num_layers=1, bidirectional=False, \
            dropout=dropout)
        self.Decoder2out = nn.Linear(self.hidden_dim, self.vocab_size)

        # self.attn = Attention(self.encoder_hidden_dim, self.hidden_dim)

    def forward(self,inp,hidden,w_context):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        inp:batch X 1
        w_context:batch X hidden_dim
        """
        # input dim                            # batch_size
        
        emb=F.embedding(inp,self.emb_martix) #batch X emb_sizes
        gru_inp=torch.cat((emb,w_context),1)
        gru_inp=gru_inp.unsqueeze(0)
        out, hidden = self.DecoderNet(gru_inp, hidden)  
        out = self.Decoder2out(out.view(-1, self.hidden_dim))  # batch_size x vocab_size
        out = F.log_softmax(out)
        return out, hidden

    def enc_context(self,inp,hidden,h_z,max_len,batch_size=64):

        states = (Variable(torch.zeros(1, batch_size, 128)).cuda(),
                  Variable(torch.zeros(1, batch_size, 128)).cuda())

        lmout,_=self.rnnlm(inp,states)


        #hz:batch X emb_size
        emb=F.embedding(inp,self.emb_martix)
        emb = emb.permute(1, 0, 2)
        z_expad=h_z.expand(max_len,batch_size,self.hidden_dim)
        emb=torch.cat((emb,z_expad),2).contiguous()   #add the hideen z 
        hidden=hidden.contiguous()
        out,h_out = self.EnccontextNet(emb, hidden)
        
        out = out.permute(1, 0, 2).contiguous()
        lmout=lmout.view(batch_size,max_len,-1)

        out=torch.cat((out,lmout),2).contiguous()

        return out
        
        # h_out=h_out.permute(1,0,2).contiguous()   
        # h_out=h_out.view(self.batch_size,-1)
        # h_z=self.hidden_to_z(h_out)

    def sample(self, num_samples, input_x,max_len,encoder):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """
        input_x=input_x[:num_samples]
        samples = torch.zeros(num_samples, max_len).type(torch.LongTensor)
        samples = samples.cuda()
        input_x = input_x.cuda()

        hidden= autograd.Variable(torch.randn(num_samples, self.hidden_dim)).cuda()
        
        h_re = encoder.init_hidden(input_x.size()[0])
        h_z  = encoder.forward(input_x, h_re,num_samples)

        # h_z=autograd.Variable(torch.randn(num_samples, self.hidden_dim)).cuda()
        inp = autograd.Variable(torch.LongTensor([self.start_letter]*num_samples)).cuda()
       
        
        
        h_z_expand=hidden.expand(3*2,num_samples,self.hidden_dim)
        context=self.enc_context(input_x,h_z_expand,h_z,max_len,num_samples)
        h= autograd.Variable(torch.randn(1,num_samples, self.hidden_dim)).cuda()
        for i in range(max_len):
            w_context, _ = self.attn(h.squeeze(0), context) #c_t :batch_size x hidden_size 
            out, h = self.forward(inp,h,w_context)
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.data
            inp = out.view(-1)
        return input_x,samples
        
    def batchNLLLoss(self,decoder_inp,target,batch_size,hidden,max_len):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """
        loss_fn = nn.NLLLoss()
        target = target.permute(1, 0)     # seq_len x batch_size
        hz=hidden
        h_z_expand=hidden.expand(3*2,self.batch_size,self.hidden_dim)
        loss = 0
        #get the context
        context=self.enc_context(decoder_inp,h_z_expand,hz,max_len)
        inp = autograd.Variable(torch.LongTensor([self.start_letter]*batch_size)).cuda() #batch X 1
        h= autograd.Variable(torch.randn(1,self.batch_size, self.hidden_dim)).cuda()
        for i in range(max_len):
            w_context, _ = self.attn(h.squeeze(0), context) #c_t :batch_size x hidden_size 
            out, h = self.forward(inp,h,w_context)
            loss += loss_fn(out, target[i])
            out = torch.multinomial(torch.exp(out), 1)
            inp = out.view(-1)

        return loss/max_len     # per batch

class Discriminator(nn.Module):
    def __init__(self, z_dim,h_dim, gpu=False,dropout=0.2):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.z_to_hidden = nn.Linear(z_dim, h_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.ReLU=nn.ReLU()
        self.z_to_out = nn.Linear(h_dim, 1)
        self.Sigmoid=nn.Sigmoid()

    def clf(self,inp):
        hidden=self.z_to_hidden(inp)
        hidden = self.dropout_linear(hidden)
        logist=self.z_to_out(hidden)
        out = self.Sigmoid(logist)
        return out