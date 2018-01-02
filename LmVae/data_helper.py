import torch
import os
import load_wordvec
import numpy as np
class Corpus(object):
    def __init__(self, path='./data',train_file='train.txt'):
        self.w2idx,self.word_matrix = load_wordvec.load_vectorsmodel()
        self.idx2w=dict(zip(self.w2idx.values(),self.w2idx.keys()))
        
        self.file_path = os.path.join(path,train_file)
        if '<start>' not in self.w2idx.keys():
            self.w2idx['<start>']=len(self.w2idx)
            self.word_matrix= torch.cat((self.word_matrix,torch.FloatTensor(np.array([0]*128)).view(-1,128)), 0)

    def get_data(self, batch_size=20):
        # Add '<eos>' to the dictionary
        
        with open(self.file_path, 'r') as f:
            tokens=0
            for line in f:
                words = line.split() + ['<eos>']+['<pad>']
                tokens += len(words)
              
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(self.file_path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.w2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]
        return ids.view(batch_size, -1)


    