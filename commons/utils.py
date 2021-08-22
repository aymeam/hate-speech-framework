# coding: utf-8
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
# from tqdm import tqdm, trange
import pandas as pd
import io
# from transformers import *
from laserembeddings import Laser
laser = Laser()

import gc
import numpy as np
import matplotlib.pyplot as plt
# from google.colab import drive
import os
import pickle
# from smart_open import open
# from gensim.models.wrappers import FastText
from collections import Counter
import sys
# import syssdds3


sys.path.append('/gdrive/My Drive/Workspace/Code/Deep Learning Hate Speech/models')
sys.path.append('./ContextualEmbeddings')
sys.path.append('../transformers')
sys.path.append('/gdrive/My Drive/Workspace/Code/Deep Learning Hate Speech/ContextualEmbeddings/')
sys.path.append('/gdrive/My Drive/')

# import Pretrained
# from Pretrained import *
# import FNN_model
# from FNN_model import FNN
# from Models import HateBiLSTM,HateCNN,HateLSTM,HateFNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
from itertools import chain, repeat, islice


import json
import gensim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from nltk import tokenize as tokenize_nltk
from string import punctuation
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from keras.preprocessing.sequence import pad_sequences
# !pip install fasttext
# import fasttext
TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
# from gensim.models import FastText as ft
def mult(lista, entero):
    for i in range(len(lista)):
        lista[i] *= entero
    return lista
   
def pad(iterable, size, padding=None):
       return islice(pad_infinite(iterable, padding), size)


def contextualizer2(X, current_combination, context_embedding_setup, device, embedding_type):
  batch_size = 100

  n_batches_t = int(len(X)/batch_size)
  if n_batches_t * batch_size < len(X):
      n_batches_t += 1
  tokenizer, model_type, weights = context_embedding_setup
  embedder = Contextual_Embedder(tokenizer, model_type, weights, device, embedding_type)  
 
  #print(n_batches_t)
  last_batch = False
  first = True
  for i in range(n_batches_t):
    #   print(i)
      inputs = X[i*batch_size:(i+1)*batch_size]
      if len(inputs) <  batch_size:
        inputs = X[i*batch_size:len(X)]  

      if first:
        emb = embedder.get_embeddings(inputs, int(current_combination["max_len"]))
        first = False
      else:
        emb_batch = embedder.get_embeddings(inputs, int(current_combination["max_len"]))
        #print(emb_batch.shape)
        emb = torch.cat((emb,emb_batch),dim = 0)
        #print(emb.shape)
        del emb_batch
        gc.collect()
  return emb 
  
def load_embeddings_from_dic(vec, X, experiment_name,a):
        model_1,model_2,model_3,word2id_1,word2id_2,word2id_3 = a
        vocab = gen_vocab(X)
        # print("Loading embeddings ...")
        word2idx = {word: ii for ii, word in enumerate(vocab, 1)}
        #print(model_1)
        embedding_dim = len(model_1[0])
        embeddings = np.zeros((len(word2idx) + 1, embedding_dim))
        c = 0 
        for k, v in vocab.items():
            if experiment_name.find('cross') >= 0:
                try:
                    index_in_model_1 = word2id_1[k]
                    # print(len(model_1[index_in_model_1]))
                    embeddings[v] = model_1[index_in_model_1]
                    #print('embeddings[v]',embeddings[v])
                except:
                    try:
                        index_in_model_2 = word2id_2[k]
                        embeddings[v] = model_2[index_in_model_2]
                        # embeddings[v] /= 2
                        # print(embeddings[v])
                    except:
                        try:
                            index_in_model_3 = word2id_3[k]
                            embeddings[v] = model_3[index_in_model_3]
                            # print(embeddings)
                        except:
                            pass
            elif experiment_name.find('mono') >= 0:
                try:
                    index_in_model = word2id[k]
                    embeddings[v] = model[index_in_model]
                except:
                    pass
        # print(embeddings)    
        return torch.from_numpy(embeddings).float(), embedding_dim
def most_frequent(List): 
    counter=Counter(List)
    max_freq= max(set(List), key = List.count)
    return max_freq,counter[max_freq]


def CheckForGreater(list1, val):  
    return(all(x > val for x in list1))  

def pad_infinite(iterable, padding=None):
       return chain(iterable, repeat(padding))

def encode_data_(vector,id2word,word2id, data, labels):
  tuple_new_data=[]

  for row in range(len(data)):
    list_token_id=[]
    
    words=data[row].split(' ')
    for word in words:
        try:
            index=word2id[word]
        except KeyError:
            index=len(list(word2id.keys()))
        list_token_id.append(index)
    with_padding_text=list(pad(list_token_id, 128, len(list(word2id.keys()))+1))
    if labels == None:
      tuple_new_data.append((with_padding_text, 0, data[row]))
    else:
      tuple_new_data.append((with_padding_text, labels[row], data[row]))
    
  return tuple_new_data
   
def load_embeddings_static(vec, experiment_name, device):
    model_1,model_2,model_3,word2id_1,word2id_2,word2id_3  = {},{},{},{},{},{}
    if vec.find('cca') >= 0:
       # print(vec)
        if vec.find('clean') >= 0:
            basedir = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/REFINED/'
        else:
            basedir = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/'
    
    if vec.find('muse_aligment') >= 0:
            
        basedir = '/gdrive/My Drive/HateVectors/aligned_vectors/MUSE/'

    if vec.find('muse_aligment') >= 0 or vec.find('cca') >= 0:
        if experiment_name == 'logs_cross_esp_ita':
            path1 = basedir + "SPANISH.txt"
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = basedir +  "ITALIAN.txt"
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 100
            
        if experiment_name == 'logs_cross_ita_esp':
            path1 = basedir +  "ITALIAN.txt"
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = basedir + "SPANISH.txt"
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 100  
    
        elif experiment_name == 'logs_cross_eng_ita':
            path1 = basedir + 'IT_EN_English.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = basedir + 'IT_EN_Italian.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 50  
            
        elif experiment_name == 'logs_cross_ita_eng':
            path1 = basedir + 'IT_EN_Italian.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = basedir + 'IT_EN_English.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 50  
            
        elif experiment_name == 'logs_cross_eng_ita':
            path1 = basedir + 'ENGLISH.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = basedir + 'ITALIAN.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 100  
            
        elif experiment_name == 'logs_cross_ita_eng':
            path1 = basedir + 'ITALIAN.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = basedir + 'ENGLISH.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 100  
            
        elif experiment_name == 'logs_cross_ita_eng__esp':
            path1 = basedir + 'ITALIAN.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = basedir + 'ENGLISH.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            path3 = basedir + 'SPANISH.txt'
            model_3, id2word_3, word2id_3 = load_vec(path3)
            embedding_dim = 100  
                    
        elif experiment_name == 'logs_cross_ita_esp__eng':
            path1 = basedir + 'ITALIAN.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = basedir + 'SPANISH.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            path3 = basedir + 'ENGLISH.txt'
            model_3, id2word_3, word2id_3 = load_vec(path3)
            embedding_dim = 100  
        
        elif experiment_name == 'logs_cross_eng_esp__ita':
            path1 = basedir + 'ENGLISH.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = basedir + 'SPANISH.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            path3 = basedir + 'ITALIAN.txt'
            model_3, id2word_3, word2id_3 = load_vec(path3)
            embedding_dim = 100  
            
        
        elif experiment_name == 'logs_cross_esp_eng_esp':
            path1 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/en-es-spanish-cca.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/en-es-english-cca.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 50  
        
        elif experiment_name == 'logs_cross_esp_ita_esp':
            path1 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/es-ita-italian.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/es-ita-spanish.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 50  
            
        elif experiment_name == 'logs_cross_ita_esp_ita':
            path1 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/es-ita-italian.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/es-ita-spanish.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 50  
                
        elif experiment_name == 'logs_cross_esp_ita_ita':
            path1 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/es-ita-italian.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/es-ita-spanish.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            embedding_dim = 50  
            
        elif experiment_name == 'logs_cross_ita_eng_esp_ita':
            path1 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/es-ita-italian.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/es-ita-spanish.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            path3 = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/en-ita-english.txt'
            model_3, id2word_3, word2id_3 = load_vec(path3)
            embedding_dim = 50               
    
        elif experiment_name == 'logs_mono_eng':
            path = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/ENGLISH.txt'
            model_1, id2word_1, word2id_1 = load_vec(path)
            print('model_1[0]',model_1[0])
            embedding_dim = 100  
    
        elif experiment_name == 'logs_mono_ita':
            path = '/gdrive/My Drive/HateVectors/italian/w2v_ita_3.txt'
            model, id2word, word2id = load_vec(path)
            embedding_dim = 100 
            
        elif experiment_name == 'logs_mono_esp':
            path = '/gdrive/My Drive/HateVectors/aligned_vectors/CCA/ES_EN_Spanish.txt'
            model, id2word, word2id = load_vec(path)
            embedding_dim = 100
           
    if vec == 'muse':
        dire = '/gdrive/My Drive/Workspace'
        if experiment_name == 'logs_cross_ita_eng__esp':
            path1  = dire + '/vectors/wiki.multi.it.vec.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2  = dire + '/vectors/wiki.multi.en.vec.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            path3  = dire + '/vectors/wiki.multi.es.vec.txt'
            model_3, id2word_3, word2id_3 = load_vec(path3)
            # model = np.concatenate((model_en,model_es), axis = 0)
            embedding_dim = 300
        
        if experiment_name == 'logs_cross_ita_esp__eng':
            path1  = dire + '/vectors/wiki.multi.it.vec.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2  = dire + '/vectors/wiki.multi.es.vec.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            path3  = dire + '/vectors/wiki.multi.en.vec.txt'
            model_3, id2word_3, word2id_3 = load_vec(path3)
            embedding_dim = 300
        
        if experiment_name == 'logs_cross_eng_esp__ita':
            path1  = dire + '/vectors/wiki.multi.en.vec.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2  = dire + '/vectors/wiki.multi.es.vec.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            path3  = dire + '/vectors/wiki.multi.it.vec.txt'
            model_3, id2word_3, word2id_3 = load_vec(path3)
            embedding_dim = 300
        
        
        if experiment_name == 'logs_cross_esp_ita':
            path1  = dire + '/vectors/wiki.multi.es.vec.txt'
            path2  = dire + '/vectors/wiki.multi.it.vec.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            model_2, id2word_2, word2id_2 = load_vec(path2)
            # model = np.concatenate((model_en,model_es), axis = 0)
            embedding_dim = 300
        
        if experiment_name == 'logs_cross_ita_esp':
            path1  = dire + '/vectors/wiki.multi.es.vec.txt'
            path2  = dire + '/vectors/wiki.multi.it.vec.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            model_2, id2word_2, word2id_2 = load_vec(path2)
            # model = np.concatenate((model_en,model_es), axis = 0)
            embedding_dim = 300
        
        if experiment_name == 'logs_cross_eng_esp':
            path1  = dire + '/vectors/wiki.multi.en.vec.txt'
            path2  = dire + '/vectors/wiki.multi.es.vec.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            model_2, id2word_2, word2id_2 = load_vec(path2)
            # model = np.concatenate((model_en,model_es), axis = 0)
            embedding_dim = 300

        if experiment_name == 'logs_cross_esp_eng':
            path1  = dire + '/vectors/wiki.multi.es.vec.txt'
            path2  = dire + '/vectors/wiki.multi.en.vec.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            model_2, id2word_2, word2id_2 = load_vec(path2)
            # model = np.concatenate((model_en,model_es), axis = 0)
            embedding_dim = 300               
            
        if experiment_name == 'logs_cross_ita_eng':
            path1  = dire + '/vectors/wiki.multi.it.vec.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2  = dire + '/vectors/wiki.multi.en.vec.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            # model = np.concatenate((model_en,model_es), axis = 0)
            embedding_dim = 300
        
        if experiment_name == 'logs_cross_eng_ita':
            path1  = dire + '/vectors/wiki.multi.en.vec.txt'
            model_1, id2word_1, word2id_1 = load_vec(path1)
            path2  = dire + '/vectors/wiki.multi.it.vec.txt'
            model_2, id2word_2, word2id_2 = load_vec(path2)
            # model = np.concatenate((model_en,model_es), axis = 0)
            embedding_dim = 300
        
        elif experiment_name == 'logs_mono_eng':
            path = dire + '/vectors/wiki.multi.en.vec.txt'
            model, id2word, word2id = load_vec(path)
            embedding_dim = 300

        elif experiment_name == 'logs_mono_ita':
            path = dire + '/vectors/wiki.multi.it.vec.txt'
            model, id2word, word2id = load_vec(path)
            embedding_dim = 300
       
        elif experiment_name == 'logs_mono_esp':
            path = dire + '/vectors/wiki.multi.es.vec.txt'
            model, id2word, word2id = load_vec(path)
            embedding_dim = 300

    return model_1,model_2,model_3,word2id_1,word2id_2,word2id_3
        
 
def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id
def load_embeddings(vec, X, experiment_name, device, current_combination):
    if experiment_name.find('cross') >= 0: #seteando para cada contextual model
        model_type = BertModel
        tokenizer = BertTokenizer
        weights = 'bert-base-multilingual-uncased'  
    
    elif experiment_name == 'logs_mono_esp':
        tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        weights = "dccuchile/bert-base-spanish-wwm-uncased"
        model_type = BertModel
        #model_type = AutoModel

    elif experiment_name == 'logs_mono_eng':
        model_type = BertModel
        tokenizer = BertTokenizer
        weights = 'bert-base-uncased'  
    
    elif experiment_name == 'logs_mono_ita':
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
        weights = "dbmdz/bert-base-italian-uncased"
        model_type = BertModel
        
    else:
        print('no model available')
        
    context_embedding_setup = (tokenizer, model_type, weights) 
    return context_embedding_setup
    # print(weights)


def gen_vocab(tweets):
    vocab, reverse_vocab = {}, {}
    vocab_index = 1
    for tweet in tweets:
        text = TOKENIZER(tweet.lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'
    return vocab


def gen_sequence(tweets,vocab):
    X = []
    for tweet in tweets:
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet.lower())
        # text = ' '.join([c for c in text if c not in punctuation])
        # words = text.split()
        # words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in text:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
    return X  


def plot_training_performance(loss, acc, val_loss, val_acc):
    plt.figure(figsize=(5,5))
    plt.title("loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['train', 'validation'], loc='upper left')

    plt.figure(figsize=(5,5))
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def get_config(model_name, var, json_file, envio):
    #read parameters form config file
    configurations = 1
    if envio == "local":
        base_dir = "./"
    else:
        base_dir = "/gdrive/My Drive/Workspace/Code/Deep Learning Hate Speech/"    
    if var == 'config':
        json_file = open(base_dir + 'config/' + model_name + '.json', 'r')
        params = json.load(json_file)
        for value in params.values():
            configurations *= len(value)
        print("The posible configurations are: ", configurations)
        return(params)
    if var == 'best':
        best = open(json_file, 'r')
        params = json.load(best)
        return(params)

def data_loaders(X, y_train, batch_size):
 
  y_train = np.array(binarize(y_train))  
  train_data = TensorDataset(X, torch.tensor(y_train))
  train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
  return train_loader

from sklearn.model_selection import train_test_split

def max_length(X):
  post_length = max(np.array([len(x.split(" ")) for x in X]))
  return post_length 

def save_object(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def load_object(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def load_data(dataset):
    data = load_object(dataset)
    x_text = []
    labels = []
    for i in range(len(data)):
        x_text.append(data[i]['text'])
        labels.append(data[i]['label'])
    return x_text, labels

  
def save_object(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def load_object(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def load_data_csv(dataset):
    data = pd.read_csv(dataset)
    x_text = []
    labels = []
    
    for i in data.values:
        x_text.append(i[0])
        labels.append(i[1])
#         print(dataset)          
    print(Counter(labels))
    return x_text, labels

def load_data(dataset):
    data = load_object(dataset)
    x_text = []
    labels = []
    for i in range(len(data)):
        try:
          x_text.append(data[i]['text'])
          labels.append(data[i]['label'])
        except:
          print(dataset)          

    print(Counter(labels))
    return x_text, labels

#MAX LENGTH FUNCTION
#FUNCTION TO CALCULATE THE MAX LENGTH OF THE DATA TO BE USE FOR PADDING LATER
def max_length(X):
  post_length = max(np.array([len(x.split(" ")) for x in X]))
  return post_length 


#FUNCTION TO MAP INTO BINARY CLASSES
#THE ORGINAL CLASSES ARE TANSFORMED INTO BINARY 
def binarize(y_):
    y_map = {
            'none': 0,
            'normal': 0,
            'neither':0,
             'both':1,
            'racism': 1,
            'sexism': 1,
            'hate':1,
            'hateful':1,
            'abusive': 1,
            'ofenssive':1,
            'NOT': 0,
            'HS': 0,

            'OFF': 0, 
            'N':0,
            'H':0,
            '1': 1,
            '0':0,
            1: 1,
            0:0
    }
    y = []
    for i in y_:
        y.append(y_map[i])
    return y  

#EVALUATION METRICS FUNCTIONS
# from tunning import hyperparameter_tunning
def contextualizer(X, X_val, current_combination, context_embedding_setup, device, embedding_type):
  batch_size = 100#current_combination['batch_size']
  n_batches_t = int(len(X)/batch_size)
  # print('TTTTurning into embeddings all data', current_combination["embedding_type"])
 
  tokenizer, model_type, weights = context_embedding_setup
  embedder = Contextual_Embedder(tokenizer, model_type, weights, device, embedding_type)  
 
  print(n_batches_t)
  last_batch = False
  first = True
  for i in range(n_batches_t):
      print(i)
      inputs = X[i*batch_size:(i+1)*batch_size]
      if len(inputs) < batch_size:
        break
      if first:
        emb = embedder.get_embeddings(inputs, int(current_combination["max_len"]))
        first = False
      else:
        emb_batch = embedder.get_embeddings(inputs, int(current_combination["max_len"]))
        emb = torch.cat((emb,emb_batch),dim = 0)
        if i % 2 == 0:
          save_object(emb,'/gdrive/My Drive/temp/Workspace/Code/Deep Learning Hate Speech/pkg' + str(i) + '.pkl')
          print('object_saved')
        #   print(emb.shape)

          emb = 0
          del emb
          first = True
        del inputs
          
        gc.collect()

  # X = emb
  emb = 0
  print('val')
  batch_size = 100#current_combination['batch_size']

  n_batches = int(len(X_val)/batch_size)
  print(n_batches)
  last_batch = False
  first = True
  for i in range(n_batches):
      print(i)
      inputs = X_val[i*batch_size:(i+1)*batch_size]
      if len(inputs) < batch_size:
        break
      if first:
        emb = embedder.get_embeddings(inputs, int(current_combination["max_len"]))
        first = False
      else:
        emb_batch = embedder.get_embeddings(inputs, int(current_combination["max_len"]))
        emb = torch.cat((emb,emb_batch),dim = 0)
        del emb_batch
      del inputs
      gc.collect()
  
#   print('emb.shape',emb.shape)
  X_val = emb
  del emb
  del embedder
  gc.collect()

  first =True
  for i in range(n_batches_t):
    if i % 2 == 0 and i != 0:
        if first:
          a = torch.tensor(load_object('/gdrive/My Drive/temp/Workspace/Code/Deep Learning Hate Speech/pkg' + str(i) + '.pkl'))
        #   print(a.shape)
          first = False
        else:
          a = torch.cat((a, load_object('/gdrive/My Drive/temp/Workspace/Code/Deep Learning Hate Speech/pkg' + str(i) + '.pkl')) ,dim = 0)
#   print(a[0])
  X = torch.tensor(a)
  del a
  gc.collect()

#   print('X.shape',X.shape)
  return X, X_val
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def evaluate(predictions, test_labels):
    precision, recall, fscore, support = score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print(sum(precision)/2)
    print('recall: {}'.format(recall))
    print(sum(recall)/2)
    print('fscore: {}'.format(fscore))
    print(sum(fscore)/2)
    print('support: {}'.format(support))

def plot_training_performance(loss, acc, val_loss, val_acc):
    plt.figure(figsize=(5,5))
    plt.title("loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['train', 'validation'], loc='upper left')

    plt.figure(figsize=(5,5))
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def get_embedding_weights(vectors_file):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(vectors_file)
    embedding = torch.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.iteritems():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    
    return embedding

def gen_vocab(tweets):
    vocab, reverse_vocab = {}, {}
    vocab_index = 1
    for tweet in tweets:
        text = TOKENIZER(tweet.lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'
    return vocab

def finalizar_trainning(losses,accs):
    plot_training_performance(losses,accs,'losses')
    plot_training_performance(losses,accs,'accs')

def gen_sequence(tweets,vocab):
    X = []
    for tweet in tweets:
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet.lower())
        # text = ' '.join([c for c in text if c not in punctuation])
        # words = text.split()
        # words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in text:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
    return X  
import gensim

#PREPARING DATA FUNCTION
#THE FUNCTIONS ADD THE SPECIAL TOKENS CLS AND SEP
#THE SENTENCES ARE TOKENIZE USING THE BERT TOKENIZER
def preparing_data(X, tokenizer):
  sentences = ["[CLS] " + query + " [SEP]" for query in X]
#   print(sentences[0])

  # Tokenize with BERT tokenizer
  tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
  print ("Tokenize the first sentence:")
  print (tokenized_texts[1])
  return sentences, tokenized_texts

# Pad our input tokens
def ids(t_texts,MAX_LEN, tokenizer):
  input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in t_texts], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

  # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in t_texts]
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  return input_ids

def attention_m(input_ids):
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

def save_preds(preds, filee):
    with open(filee, 'w+') as out:
      for p in preds:
        out.write(str(int(p)))
        out.write('\n')


def gen_data_laser(tweets_list, langu):

  embeddings = laser.embed_sentences(tweets_list, lang = langu) 
  embeddings = np.array(embeddings)
  print(embeddings.shape)
  return embeddings