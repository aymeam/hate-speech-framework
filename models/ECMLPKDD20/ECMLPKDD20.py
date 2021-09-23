from utils import *
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
print("CURR_DIR")
print(CURR_DIR)
from sklearn.linear_model import LogisticRegression
sys.path.append(CURR_DIR + "/source/CNN_GRU/Models")
sys.path.append(CURR_DIR + "/source/CNN_GRU")
sys.path.append(CURR_DIR + "/source/BERT Classifier")
print(CURR_DIR + "/source/BERT Classifier")

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig

#sys.path.append(CURR_DIR + "source/BERT Classifier/bert_codes")
from transformers import get_linear_schedule_with_warmup
import time

from bert_codes.feature_generation import combine_features,return_dataloader,return_cnngru_dataloader

from bert_codes import utils as bert_codes_utils
print(sys.path)
# import BERT_inference
# import BERT_training_inference
import glob
import numpy as np
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification
from utils import *

import glob
import numpy as np
from Models.modelUtils import *
from Models.torchDataGen import *
from bert_codes.utils import *

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
import time
from Models.model import *
from collections import Counter
# from BERT_training_inference import *

import collections
import io
import numpy as np
from bert_codes.feature_generation import combine_features,return_dataloader,return_cnngru_dataloader
from transformers import AutoTokenizer, AutoModel

from bert_codes.data_extractor import data_collector
from bert_codes import *
from torchDataGen import *
import argparse

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig


def select_model(args,vector=None):
    text=args["path_files"]

    if(text=="birnn"):
        model=BiRNN(args)
    if(text == "birnnatt"):
        model=BiAtt_RNN(args,return_att=False)
    if(text == "birnnscrat"):
        model=BiAtt_RNN(args,return_att=True)
    if(text == "cnngru"):
        model=CNN_GRU(args,vector)
    if(text == "lstm_bad"):
        model=LSTM_bad(args)
    return model

class ECMLPKDD20():
    def __init__(self, setup):
      super(ECMLPKDD20, self).__init__()
      self.name = "ECMLPKDD20"
      self.paper = "ECMLPKDD20"   
      self.original_source_code = "https://github.com/hate-alert/DE-LIMIT"
      self.c = 10
      self.seed_val = 2018
      self.tokenizer =  tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased", do_lower_case=False)
      self.setup = setup
    def train(self, train, dataLabel, model_type, emb1 = None):
      print("model_type",model_type)
      lang_train = self.setup.split('-')[0]
      lang_test = self.setup.split('-')[1] 
      if model_type == 'LASER+L':
        print('LASER+LRRRRRR')
        model = LogisticRegression(C=self.c,solver='lbfgs',class_weight='balanced',random_state=self.seed_val)
        dataTrain = gen_data_laser(train, lang_train)
        #train_model(params)
        model.fit(dataTrain, dataLabel)
        return model
    
      elif model_type == 'MBERT' or model_type == 'Translation + BERT':
        print('MBERT')
        params={'is_train':True,	'is_model':True,	'learning_rate':2e-5,	'files':'../Dataset',	'csv_file':'*_full.csv',	'samp_strategy':'stratified',
        'epsilon':1e-8,	'path_files':'multilingual_bert',	'take_ratio':False,	'sample_ratio':16,	'how_train':'baseline',	'epochs':5,	'batch_size':16,
        'to_save':True,	'weights':[1.0,1.0],	'what_bert':'normal',	'save_only_bert':False,	'max_length':128,	'random_seed':42}
        
        ######        
        #Load the bert tokenizer
        print('Loading BERT tokenizer...')
        sentences_train = train#.text.values

        labels_train = dataLabel
        #labels_val = df_val.label.values

        label_counts=Counter(dataLabel)#df_train['label'].value_counts()
        print("label_counts")
        print(label_counts)
        label_weights = [ (len(train))/label_counts[0],len(train)/label_counts[1] ]

        # Select the required bert model. Refer below for explanation of the parameter values.
        model= BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased")#select_model(params['what_bert'],params['path_files'],params['weights'])
        # Tell pytorch to run this model on the GPU.
        model.cuda()

        # Do the required encoding using the bert tokenizer
        input_train_ids,att_masks_train=combine_features(sentences_train,self.tokenizer,params['max_length'])

        # Create dataloaders for both the train and validation datasets.
        train_dataloader = return_dataloader(input_train_ids,labels_train,att_masks_train,batch_size=params['batch_size'],is_train=params['is_train'])

        # Initialize AdamW optimizer.
        optimizer = AdamW(model.parameters(),
                lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
              )

        # Number of training epochs (authors recommend between 2 and 4)
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * params['epochs']

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                              num_warmup_steps = int(total_steps/10), # Default value in run_glue.py
                              num_training_steps = total_steps)

        # Set the seed value all over the place to make this reproducible.
        bert_codes_utils.fix_the_random(seed_val = params['random_seed'])
        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        # Create a new experiment in neptune for this run. 
        bert_model = params['path_files']
        # For each epoch...
        for epoch_i in range(0, params['epochs']):
          print("ffffff")
          print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
          print('Training...')

          # Measure how long the training epoch takes.
          t0 = time.time()

          # Reset the total loss for this epoch.
          total_loss = 0
          model.train()

          # For each batch of training data...
          for step, batch in tqdm(enumerate(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
              # Calculate elapsed time in minutes.
              elapsed = format_time(time.time() - t0)
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to('cuda')
            b_input_mask = batch[1].to('cuda')
            b_labels = batch[2].to('cuda')
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Get the model outputs for this batch.
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,  labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        return model

      elif model_type == 'CNN-GRU':
        params={'logging':'local','language':'German','is_train':True,'is_model':True,
        'learning_rate':1e-4,'files':'../Dataset','csv_file':'*_full.csv','samp_strategy':'stratified',
        'epsilon':1e-8,'path_files':'cnngru','take_ratio':True,'sample_ratio':100,'how_train':'baseline',
        'epochs':20,'batch_size':16,'to_save':False,'weights':[1.0,1.0],'what_bert':'normal',
        'save_only_bert':False,'max_length':128,'columns_to_consider':['directness','target','group'],
        'random_seed':42,'embed_size':300,'train_embed':True,'take_target':False,'pair':False} 

        params['learning_rate']=2e-4
        params['random_seed']=2018
        ###########
        try:
          vector,id2word,word2id=load_vec(emb1)

        except:
          sys.out("provide the embeddings path")

        train_data=encode_data_(vector,id2word,word2id,train,dataLabel)
        
        pad_vec=np.random.randn(1,300) 
        unk_vec=np.random.randn(1,300)
        merged_vec=np.append(vector, unk_vec, axis=0)
        merged_vec=np.append(merged_vec, pad_vec, axis=0)
        params['vocab_size']=merged_vec.shape[0]

        # Generate the dataloaders
        train_dataloader = return_cnngru_dataloader(train_data,batch_size=params['batch_size'],is_train=True)

        model=select_model(params,merged_vec)
        # Tell pytorch to run this model on the GPU.
        model.cuda()
        optimizer = AdamW(model.parameters(),
                      lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
                    )

        fix_the_random(seed_val = params['random_seed'])
        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        bert_model = params['path_files']
        language  = params['language']
        name_one=bert_model+"_"+language
        if(params['take_target']):
            name_one += '_target'
    
        for epoch_i in range(0, params['epochs']):
          print("fffff")
          print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
          print('Training...')
          # Measure how long the training epoch takes.
          t0 = time.time()
          # Reset the total loss for this epoch.
          total_loss = 0
          model.train()
          # For each batch of training data...
          for step, batch in tqdm(enumerate(train_dataloader)):
              # Progress update every 40 batches.
              if step % 40 == 0 and not step == 0:
                  # Calculate elapsed time in minutes.
                  elapsed = format_time(time.time() - t0)
              # `batch` contains three pytorch tensors:
              #   [0]: input ids 
              #   [2]: labels 
              b_input_ids = batch[0].to('cuda')
              b_labels = batch[1].to('cuda')
              # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
              model.zero_grad()        

              outputs = model(b_input_ids,b_labels)
              # The call to `model` always returns a tuple, so we need to pull the 
              # loss value out of the tuple.
              loss = outputs[0]

              if step % 40 == 0 and not step == 0:
                  print('batch_loss',loss)
              total_loss += loss.item()

              # Perform a backward pass to calculate the gradients.
              loss.backward()

              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              optimizer.step()
              print("step1")
        return model
        ###########
    
    def test(self, model, test, setup, model_type, emb2 = None):
      lang_test = setup.split('-')[1] 

      if model_type == 'LASER+L':
        print('test')
        dataTest = gen_data_laser(test, lang_test)
        #train_model(params)
        y_pred = model.predict(dataTest)
        del model
        torch.cuda.empty_cache()
        file_out = CURR_DIR +"/results_" +  setup + '_'+ self.name+".txt"
        save_preds(y_pred, file_out)
      
      elif model_type == 'MBERT' or model_type == 'Translation + BERT':
        print('MBERT')
        params={'is_train':True,	'is_model':True,	'learning_rate':2e-5,	'files':'../Dataset',	'csv_file':'*_full.csv',	'samp_strategy':'stratified',
        'epsilon':1e-8,	'path_files':'multilingual_bert',	'take_ratio':False,	'sample_ratio':16,	'how_train':'baseline',	'epochs':5,	'batch_size':16,
        'to_save':True,	'weights':[1.0,1.0],	'what_bert':'normal',	'save_only_bert':False,	'max_length':128,	'random_seed':42}
        test=encode_data_(vector,id2word,word2id,test,dataLabel)

        test_dataloader = DataLoader(test, batch_size=params['batch_size'])

        for step, batch in tqdm(enumerate(test_dataloader)):
          # Progress update every 40 batches.
          t0 = time.time()

          if step % 40 == 0 and not step == 0:
              # Calculate elapsed time in minutes.
              elapsed = format_time(time.time() - t0)
          #   [0]: input ids 
          #   [2]: labels 
          b_input_ids = batch.to('cuda')
          out = model(b_input_ids, labels=None)

          model.zero_grad()  
          if step == 0:      
            y_pred=torch.argmax(out.logits, axis=1)
          else:
            y_pred = torch.cat((y_pred, torch.argmax(out.logits, axis=1)), 0)
        del model
        torch.cuda.empty_cache()
        file_out = CURR_DIR + "/results_" +  setup + '_'+ self.name+".txt"
        save_preds(y_pred, file_out)        

      elif model_type == 'CNN-GRU':
        params={'logging':'local','language':'German','is_train':True,'is_model':True,
        'learning_rate':1e-4,'files':'../Dataset','csv_file':'*_full.csv','samp_strategy':'stratified',
        'epsilon':1e-8,'path_files':'cnngru','take_ratio':True,'sample_ratio':100,'how_train':'baseline',
        'epochs':20,'batch_size':16,'to_save':True,'weights':[1.0,1.0],'what_bert':'normal',
        'save_only_bert':False,'max_length':128,'columns_to_consider':['directness','target','group'],
        'random_seed':42,'embed_size':300,'train_embed':True,'take_target':False,'pair':False} 
        model.cuda()
        vector,id2word,word2id=load_vec(emb2)

        test_data=encode_data_(vector,id2word,word2id,test,None)

        test_dataloader=return_cnngru_dataloader(test_data,batch_size=params['batch_size'],is_train=False)

        model.eval()
        for step, batch in tqdm(enumerate(test_dataloader)):
            t0 = time.time()

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            #   [0]: input ids 
            #   [2]: labels 
            b_input_ids = batch[0].to('cuda')
            model.zero_grad()  
            out = torch.argmax(model(b_input_ids),axis=1)

            if step == 0:      
              outputs = out
            else:
              outputs = torch.cat((outputs, out), 0)
        del model
        torch.cuda.empty_cache()
        file_out = CURR_DIR +"/results_" +  setup + '_'+ self.name+".txt"
        save_preds(outputs, file_out)



