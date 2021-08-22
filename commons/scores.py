from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y_pred', '--y_pred', required=True)
    parser.add_argument('-y_test', '--y_test', required=True)

    args = parser.parse_args()
    pred_path = args.y_pred
    test_path = args.y_test

    pred = []

    test_data = pd.read_csv(test_path)
    test = test_data['label']

    with open(pred_path, 'r') as p:
      lines_pred = p.readlines()
      print('len(lines_pred)','len(test)')
      print(len(lines_pred),len(test))
      assert  len(lines_pred) == len(test), "preds and real values different dimensions"

      for i in lines_pred:
        try:
          pred.append(int(i.split('(')[1].split(',')[0]))

        except:
          pred.append(int(i))

    pred = np.array(pred)
    test = np.array(test)
    print("pred, test")
    print(pred, test)
    print(len(pred), len(test))

    # print(pred)
    # print(test)
    # precision     
    precision_micro = precision_score(test, pred, average= 'micro')
    precision_average = precision_score(test, pred, average= 'macro')
    precision_weigthed = precision_score(test, pred, average= 'weighted')
    
    # recall     
    recall_micro = recall_score(test, pred, average= 'micro')
    recall_average = recall_score(test, pred, average= 'macro')
    recall_weigthed = recall_score(test, pred, average= 'weighted')

    # fscore     
    f1_score_micro = f1_score(test, pred, average= 'micro')
    f1_score_average = f1_score(test, pred, average= 'macro')
    f1_score_weigthed = f1_score(test, pred, average= 'weighted')
    
    #MSError
    mse = np.square(np.subtract(test, pred)).mean()
    
    #AUC 
    fpr, tpr, thresholds = roc_curve(test, pred)
    auc = auc(fpr, tpr)
    print(auc, mse, precision_micro, precision_average, precision_weigthed, recall_micro, recall_average,recall_weigthed,f1_score_micro, f1_score_average, f1_score_weigthed)
