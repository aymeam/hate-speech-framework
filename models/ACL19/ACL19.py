import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import argparse
import sys
import os
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from utils import *
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURR_DIR)
sys.path.append(CURR_DIR + "/source/source code/cross-lingual/Joint Learning/")
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
np.random.seed(1)
import codecs
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional, Flatten, concatenate
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tensorflow as tf
from keras_self_attention import SeqSelfAttention

from keras import backend as K
# tf.set_random_seed(1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess) 
import codecs

def parse_training(fp):

    y = []
    corpus = []
    with codecs.open(fp, encoding="ANSI") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = int(line.split("\t")[0])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def parse_testing(fp):

    y = []
    corpus = []
    with codecs.open(fp, encoding="ANSI") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = int(line.split("\t")[2])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def buildEmbedding1 (X, emb):
    embeddings_index = {}
    tokenizer = Tokenizer(num_words = 15000)
    tokenizer.fit_on_texts(X)
    with codecs.open(emb, encoding='utf-8') as f:
        print("read embedding...")
        for line in f:
            values = line.split(" ")
            word = values[0]
            #print(word)
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    print("vovabulary size = "+str(len(tokenizer.word_index)))
    print("embedding size = "+str(len(embeddings_index)))
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
    unk_dict = {}
    vocab = len(tokenizer.word_index)
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        elif word in unk_dict:
            embedding_matrix[i] = unk_dict[word]
        else:
            # random init, see https://github.com/bwallace/CNN-for-text-classification/blob/master/CNN_text.py
            unk_embed = np.random.random(300) * -2 + 1
            unk_dict[word] = unk_embed
            embedding_matrix[i] = unk_dict[word]
    return vocab, embedding_matrix

def buildEmbedding2 (X,emb):
    embeddings_index = {}
    tokenizer = Tokenizer(num_words = 15000)
    tokenizer.fit_on_texts(X)
    with codecs.open(emb, encoding='utf-8') as f:
        print("read embedding...")
        for line in f:
            values = line.split(" ")
            word = values[0]
            #print(word)
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    print("vovabulary size = "+str(len(tokenizer.word_index)))
    print("embedding size = "+str(len(embeddings_index)))
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
    unk_dict = {}
    vocab = len(tokenizer.word_index)
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        elif word in unk_dict:
            embedding_matrix[i] = unk_dict[word]
        else:
            # random init, see https://github.com/bwallace/CNN-for-text-classification/blob/master/CNN_text.py
            unk_embed = np.random.random(300) * -2 + 1
            unk_dict[word] = unk_embed
            embedding_matrix[i] = unk_dict[word]
    return vocab, embedding_matrix
    
def RNN(X,X2,emb1,emb2):
    vocab1, embedding_matrix1 = buildEmbedding1(X,emb1)
    vocab2, embedding_matrix2 = buildEmbedding2(X2,emb2)
    en_inputs = Input(name='inputs',shape=(None,))
    es_inputs = Input(name='inputs2',shape=(None,))
    en_embedding = Embedding(vocab1+1, 300, input_length=100, weights=[embedding_matrix1])(en_inputs)
    es_embedding = Embedding(vocab2+1, 300, input_length=100, weights=[embedding_matrix2])(es_inputs)
    lstm1 = LSTM(16)(en_embedding)
    lstm2 = LSTM(16)(es_embedding)
    dense1 = Dense(4,name='FC1')(lstm1)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(0.4)(dense1)
    dense2 = Dense(4,name='FC2')(lstm2)
    dense2 = Activation('relu')(dense2)
    dense2 = Dropout(0.4)(dense2)  
    concat = concatenate([dense1, dense2], axis=-1)
    layer = Dense(2,name='out_layer')(concat)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=[en_inputs,es_inputs],outputs=layer)
    return model
class ACL19():
    def __init__(self, setup):
      super(ACL19, self).__init__()
      self.name = "ACL_Joint_Learning"
      self.paper = "ACL_Joint_Learning"   
      self.paper = "ACL_Joint_Learning"   
      self.original_source_code = "https://github.com/dadangewp/ACL19-SRW"
      self.max_words = 15000
      self.max_len = 100
      self.setup = setup

    def train(self, dataTrain, dataLabel, dataTrain2, emb1, emb2):

      Y_train = pd.get_dummies(dataLabel)
      #Y_test = pd.get_dummies(labelTest)

      tok1 = Tokenizer(num_words=self.max_words)
      tok1.fit_on_texts(dataTrain)

      vocab1 = len(tok1.word_index)

      tok2 = Tokenizer(num_words=self.max_words)
      tok2.fit_on_texts(dataTrain2)
      vocab2 = len(tok2.word_index)

      sequences = tok1.texts_to_sequences(dataTrain)
      sequences_matrix = sequence.pad_sequences(sequences,maxlen=self.max_len)

      sequences2 = tok2.texts_to_sequences(dataTrain2)
      sequences_matrix2 = sequence.pad_sequences(sequences2,maxlen=self.max_len)

      allxtrain = [sequences_matrix,sequences_matrix2]

      model = RNN(dataTrain, dataTrain2, emb1, emb2)
      model.summary()
      model.compile(loss='mse',optimizer=RMSprop())
      model.fit(allxtrain,Y_train,batch_size=16,epochs=4,
              validation_split=0.2)#,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
      return model
    
    def test(self, model, dataTest):
            
      tok1 = Tokenizer(num_words=self.max_words)
      tok1.fit_on_texts(dataTest)

      test_sequences = tok1.texts_to_sequences(dataTest)
      test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=self.max_len)
      
      # test_sequences2 = tok2.texts_to_sequences(dataTest2)
      # test_sequences_matrix2 = sequence.pad_sequences(test_sequences2,maxlen=max_len)
      
      y_prob = model.predict([test_sequences_matrix, test_sequences_matrix])
      y_pred = np.argmax(y_prob, axis=-1)

      file_out = CURR_DIR +"/results_" +  self.setup + '_'+ self.name+".txt"
      save_preds(y_pred, file_out)