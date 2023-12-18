import math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

from elasticsearch import Elasticsearch 
import libs.text_processor as tp


def get_categories():
    return ["AGRICULTURE","AIRLINE","ASSOCIATIONS","AUTOMOBILE","CAR RENTAL","DIGITAL",
        "DIRECT SALES","DRUGSTORE","DUTY FREE","ELETRONICS","ENTERTAINMENT","FEES & FINES",
        "FOOD","FUEL","GENERAL","HEALTH","HOME","HOTEL","JEWELRY","MONEY","PARKING","RENTAL",
        "RETAIL","SCHOOL","SERVICES","SUPERMARKET","SUPPLIES","TRANSPORT","TRAVEL","UTILITIES",
        "WHOLESALE", "NONE"]


def split_and_balance(train_table, label_dict, label, split_size=0.8):
    
    train_dataframe = None
    valid_dataframe = None
    
    for code, category in label_dict.items():
        shard = train_table[(train_table[label] == category)].copy()
        cut_index =  int(len(shard)*split_size)
        
        if train_dataframe is None:
            train_dataframe = shard[:cut_index]
        else:
            train_dataframe = pd.concat([train_dataframe, shard[:cut_index]], axis=0)
        
        if valid_dataframe is None:
            valid_dataframe = shard[cut_index:]
        else:
            valid_dataframe = pd.concat([valid_dataframe, shard[cut_index:]], axis=0)
        
        
    return train_dataframe, valid_dataframe





def gen_random_batches(X, Y, batch_size = 64, seed = 0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * batch_size : k * batch_size + batch_size]
        mini_batch_Y = shuffled_Y[k * batch_size : k * batch_size + batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * batch_size : m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

class CustomMetrics:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def feed(self, y_true, y_pred):

        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.concatenate((self.y_true, y_true), axis=None)

        if self.y_pred is None:
            self.y_pred = y_pred
        else:        
            self.y_pred = np.concatenate((self.y_pred, y_pred), axis=None)


    def results(self):

        f1_result = f1_score(self.y_true, self.y_pred, average='weighted')
        r_result = recall_score(self.y_true, self.y_pred, average='weighted')
        p_result = precision_score(self.y_true, self.y_pred, average='weighted')

        return {"F1": f1_result, "Recall": r_result, "Precision": p_result}

    def reset_states(self):
        self.y_true = []
        self.y_pred = []       



def calcule_confusion_matrix(model, valid_batches):
    y_true = None
    y_pred = None

    for batch in valid_batches:
        (batch_x, batch_y) = batch
        predictions = model(batch_x)
        predictions = tf.argmax(tf.nn.softmax(predictions), axis=1)
        
        if y_true is None:
            y_true = batch_y
        else:
            y_true =  np.concatenate((y_true, batch_y), axis=None)
            
        if y_pred is None:
            y_pred = predictions
        else:
            y_pred =  np.concatenate((y_pred, predictions), axis=None)

    return confusion_matrix(y_true, y_pred)


def gen_symn_augmentation(list_descriptor, list_class):
    es_client = Elasticsearch([{'host':'localhost','port':9200}])
    filters=["lowercase", #convert to lowercase
         "stop_words", #remove stopwords in current language
         "synonyms"] #generate synonyms

    tp.create_text_analyzer(es_client, filters, language="_portuguese_")

    aug_x = []
    aug_y = []

    for index, descriptor in enumerate(list_descriptor):
        #Gen text by synonyms
        list_gen = tp.analyze_text_gen_synonym_aug(es_client, text=descriptor)
        if len(list_gen) <= 1:
            continue

        for gen_str in list_gen[1:]:
            aug_x.append(gen_str)
            aug_y.append(list_class[index])
            print("FROM:", descriptor,"->",gen_str, "class", list_class[index] )

    return aug_x, aug_y