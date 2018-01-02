import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable
from data_helper import Corpus
import pickle as pkl
import sys
from lm import RNNLM
import os
import vae

cuda = True
vocab_size = 26066  #词典长度
seq_length = 20
batch_size = 64
embedding_dim = 128
hidden_dim = 128

######################pre trian lmmode##########
# Truncated Backpropagation 
def detach(states):
    return [state.detach() for state in states] 

def pre_train_lm(corpus,seq_length,batch_size,num_epochs):
    #pre_train language model
    vocab_size=len(corpus.w2idx)
    ids = corpus.get_data(batch_size)
    embed_size=embedding_dim
    hidden_size=hidden_dim
    num_layers=1
    learning_rate = 0.002
    num_batches = ids.size(1) // seq_length
    save_path = './saves/language_model.trc'
    
    model=RNNLM(vocab_size,embed_size,num_layers,hidden_size,(corpus.word_matrix).cuda())
    model.cuda()
    if os.path.exists(save_path):
        model.load_state_dict(torch.load('./saves/language_model.trc'))     
        return model
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
        # Initial hidden and memory states
            states = (Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda(),
                      Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda())

            for i in range(0, ids.size(1) - seq_length, seq_length):
                # Get batch inputs and targets
                inputs = Variable(ids[:, i:i+seq_length]).cuda()
                targets = Variable(ids[:, (i+1):(i+1)+seq_length].contiguous()).cuda()

                # Forward + Backward + Optimize
                model.zero_grad()
                states = detach(states)
                outputs, states = model(inputs, states) 
                loss = criterion(outputs, targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
                optimizer.step()

                step = (i+1) // seq_length
                if step % 100 == 0:
                    print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                           (epoch+1, num_epochs, step, num_batches, loss.data[0], np.exp(loss.data[0])))
        print('pre_train language model done')
        torch.save(model.state_dict(), save_path)
        return model
    
#####################vae model loss##############

def reconstruction_loss(inp,target,encoder,decoder,max_len):
    # input dim    
    h = encoder.init_hidden(inp.size()[0])
    h_z  = encoder.forward(inp, h)
    loss=decoder.batchNLLLoss(decoder_inp=inp,target=target,batch_size=batch_size,hidden=h_z,max_len=max_len)
    return loss

def regularization_dis_loss(inp,encoder,dis,max_len):
    z_real = Variable(torch.randn(batch_size, hidden_dim)).cuda()
    
    hidden= autograd.Variable(torch.randn(6, batch_size, hidden_dim)).cuda()
    z_fake = encoder.forward(inp,hidden)
    # z_fake= torch.cat((z_fake[0].view(-1, HIDDEN_DIM),z_fake[1].view(-1, HIDDEN_DIM)),0)
    D_real = dis.clf(z_real)
    D_fake = dis.clf(z_fake)
    D_loss = -torch.mean((D_real) +(1 - D_fake))
    return D_loss

def regularization_gen_loss(inp,encoder,dis,max_len):
    hidden= autograd.Variable(torch.randn(6, batch_size, hidden_dim)).cuda()
    z_fake = encoder.forward(inp,hidden)
    D_fake = dis.clf(z_fake)
    G_loss = -torch.mean((D_fake))
    return G_loss

def reset_grad(net):
    net.zero_grad()

############gen data class############

class gen_batch():
    """docstring for ClassName"""
    def __init__(self,w2idx,idx2w,batch_size,file='data/sents_17w.csv'):
        super(gen_batch, self).__init__()
        self.i = 0
        self.batch_size = batch_size
        self.file=file
        self.w2idx=w2idx
        self.idx2w=idx2w
        self.pad=self.w2idx['<pad>']
        self.sents=[]
        self.nums_batch=0
    
    def initsnets(self):
        with open(self.file,'r') as f:
            for line in f:
                words = line.strip('\n').split()+['<eos>']
                self.sents.append([self.w2idx[w] for w in words])
        self.nums_batch=len(self.sents)//self.batch_size
        
    def next_batch(self):
        batch=self.sents[self.i:self.i+self.batch_size]
        lens=list(map(len,batch))
        max_len=max(lens)
        for ba,le in zip(batch,lens):
            ba.extend([self.pad]*(max_len-le))
        batch=torch.from_numpy(np.array(batch)).cuda()
        self.i=(self.i+self.batch_size)%(self.nums_batch*self.batch_size)
        return self.i,batch,max_len

    def randnsent(self,n):
        rand_idx=np.random.permutation(len(self.sents))[:n]
        rand_sents=[self.sents[x] for x in rand_idx]
        lens=list(map(len,rand_sents))
        max_len=max(lens)

        for ba,le in zip(rand_sents,lens):
            ba.extend([self.pad]*(max_len-le))
        rand_sents=torch.from_numpy(np.array(rand_sents)).cuda()

        return rand_sents,max_len

    def transcode(self,input_x,samples):
        for sent_x,sent_g in zip(input_x,samples):
            tra_sent_x=[self.idx2w[int(x)] for x in sent_x if int(x) in self.idx2w.keys() and int(x)!=self.pad]
            tra_sent_g=[self.idx2w[x] for x in sent_g if x in self.idx2w.keys() and int(x)!=self.pad]
            print (tra_sent_x)
            print (tra_sent_g)
            print('==============')


def sample_sents_lm(num_samples,lm_model,corpus):
    state = (Variable(torch.zeros(1, 1, 128)).cuda(),
             Variable(torch.zeros(1, 1, 128)).cuda())

    prob = torch.ones(len(corpus.w2idx))
    input = Variable(torch.multinomial(prob, num_samples=1).unsqueeze(1),
                     volatile=True).cuda()
    print(input.size())

    for i in range(num_samples):
        # Forward propagate rnn 
        output, state = lm_model(input, state)
        
        # Sample a word id
        prob = output.squeeze().data.exp().cpu()
        word_id = torch.multinomial(prob, 1)[0]
        
        # Feed sampled word id to next time step
        input.data.fill_(word_id)
        
        # File write
        word = corpus.idx2w[word_id]

        word + ' '
        print(word,end=' ')



############strat train model##############
if __name__ == '__main__':
    
    ########load rnnlm model ######
    print('load language model')
    corpus = Corpus(path='./data',train_file="sents_17w_eospad.csv")
    
    lm_model=pre_train_lm(corpus=corpus,seq_length=30,batch_size=20,num_epochs=40)
    
    # sample_sents_lm(100,lm_model,corpus)
    # sys.exit(0)


    vocab_size=len(corpus.w2idx)
    lr = 1e-3
    word_matrix=corpus.word_matrix.cuda()
    encoder = vae.Encoder(batch_size,embedding_dim,hidden_dim,word_matrix)
    decoder = vae.Decoder(batch_size,embedding_dim,hidden_dim,vocab_size,word_matrix,\
        corpus.w2idx['<start>'],lm_model)
    
    dis = vae.Discriminator(hidden_dim,hidden_dim*2)
    
    if cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        dis = dis.cuda()

    encoder_solver = optim.RMSprop(encoder.parameters(), lr=lr)
    decoder_solver = optim.RMSprop(decoder.parameters(), lr=lr)
    dis_solver = optim.RMSprop(dis.parameters(), lr=lr)
    
    gen_sents=gen_batch(corpus.w2idx,corpus.idx2w,batch_size)
    gen_sents.initsnets()
    nums_batch=gen_sents.nums_batch
   
    for epoch in range(50):
        for _ in range(nums_batch):
            i,batch,max_len=gen_sents.next_batch()
            inputs = Variable(batch).cuda()
            target = Variable(batch).cuda()
            """ Reconstruction phase """
            recon_loss=reconstruction_loss(inputs,target,encoder,decoder,max_len)
            recon_loss.backward()
            encoder_solver.step()
            decoder_solver.step()
            reset_grad(encoder)
            reset_grad(decoder)
            
            """ Regularization phase """
            # Discriminator
            dis_loss=regularization_dis_loss(inputs,encoder,dis,max_len)
            dis_loss.backward()
            dis_solver.step()
            reset_grad(dis)
            # Generator
            gen_loss = regularization_gen_loss(inputs,encoder,dis,max_len)
            gen_loss.backward()
            encoder_solver.step()
            reset_grad(encoder)
            if (i/64)%100==0:
                print('epoch-{}; Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
                          .format(epoch,i, dis_loss.data[0], gen_loss.data[0], recon_loss.data[0]))
                input_x,max_len=gen_sents.randnsent(10)
                inp_x,samples=decoder.sample(num_samples=10,input_x=input_x,\
                    max_len=max_len,encoder=encoder)
                gen_sents.transcode(inp_x,samples)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
