import shutil
import gensim
import os
import numpy as np
import torch 
import torch.nn.functional as F
def load_vectorsmodel(path='/home/lab713/data1/ipython/glove/'):
    
    #加载词向量文件
    
    gensim_file='saves/gensim_file.txt'
    
    if os.path.exists(gensim_file):
        model= gensim.models.KeyedVectors.load_word2vec_format(gensim_file) #GloVe Model
    else:
        gensim.scripts.glove2word2vec.glove2word2vec(path+'vectors.txt',gensim_file)
        model= gensim.models.KeyedVectors.load_word2vec_format(gensim_file) #GloVe Model
    
    #转化出词向量矩阵和vocab
    w2idx=dict()
    vec_matrix=[]
    for word in model.vocab:
        w2idx[word]=len(w2idx)
        vec_matrix.append(model.wv[word])
        
    vec_matrix=np.array(vec_matrix)
    return w2idx,torch.FloatTensor(vec_matrix)
    


