import os
import shutil
import time
import csv
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split
import sys
import argparse
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURR_DIR)
sys.path.append(CURR_DIR + "/examples")
sys.path.append(CURR_DIR + "/source")
sys.path.append(CURR_DIR + "/commons")

from utils import *
from deepoffense.classification import ClassificationModel
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel
from turkish.turkish_deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, language_modeling_args, args, SEED, RESULT_FILE
    
# TEMP_DIRECTORY = '/gdrive/MyDrive/Workspace/Code/EMNLP 2020/temp/'

# if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
# if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
#     os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

# if GOOGLE_DRIVE:
#     download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURR_DIR)


class EMNLP20():
    def __init__(self, setup):
      super(EMNLP20, self).__init__()
      self.name = "EMNLP"
      self.paper = "EMNLP"   
      self.paper = "EMNLP"  
      self.setup = setup 
      self.lang_train = setup.split('-')[0]
      self.lang_test = setup.split('-')[1] 
      self.original_source_code = "https://github.com/dadangewp/ACL19-SRW"

    def train(self, train, dataLabel):
      #dataLabel = encode(dataLabel)
      #test_sentences = test
      from turkish.turkish_deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, args, SEED, RESULT_FILE
      
      train_list = train
      dicti = {'text':train, 'labels':dataLabel}
      train_df = pd.DataFrame(dicti)
      model = LanguageModelingModel('auto', MODEL_NAME, args=args)
      print('os.path.join(TEMP_DIRECTORY, "lm_train.txt")')
      TEMP_DIRECTORY = CURR_DIR
      print(model)
      model.train_model(train_list)
      MODEL_NAME = language_modeling_args["best_model_dir"]
      MODEL_NAME = language_modeling_args["best_model_dir"]
      print('language_modeling_args["best_model_dir"]')

      print(language_modeling_args["best_model_dir"])

      # Train the model
      print("Started Training")

      #train['labels'] = encode(train["label"])

      #test_sentences = test#['text'].tolist()
      # args["save_steps"] = False
      # args["save_model_every_epoch"] = False
      # args["save_best_model"] = False
      # args["save_eval_checkpoints"] = False
      # args["save_recent_only"] = False
      # args["best_model_dir"] = CURR_DIR
      # args["cache_dir"] = CURR_DIR
      # args["output_dir"] = CURR_DIR
    
      print(args)
      model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,  use_cuda=torch.cuda.is_available())
      model.train_model(train_df)
      return model

    def test(self,model, test_sentences):
        y_pred, raw_outputs = model.predict(test_sentences)
        print(len(y_pred))
        print(raw_outputs)

        file_out = str(args.preds_path) + str(setup) + '/'+ args.model +".txt"
        save_preds(y_pred, args.preds_path, file_out)