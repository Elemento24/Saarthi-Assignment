### IMPORTING PACKAGES ###
import os
import sys
import yaml
import joblib
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython

from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
import tensorflow.keras as keras
from tensorboard import program
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Softmax, Concatenate


### GETTING THE DATA ###
if len(sys.argv) < 3 or len(sys.argv) > 5:
  raise Exception("Invalid Arguments Passed!")

# Perform evaluation for a given test file
if sys.argv[1] == '--file': 
    is_file = True
  
# Output predictions for a given string
elif sys.argv[1] == '--text': 
    is_file = False
    sample = sys.argv[3]

config_path = sys.argv[2]

with open(f"{config_path}/config.yaml") as file:
  yaml_data = yaml.safe_load(file)


### DEFINING THE INDICES SEPARATING THE ACTION, OBJECT, LOCATION ###
act_ind = yaml_data['indices']['act_ind']
obj_ind = yaml_data['indices']['obj_ind']
loc_ind = yaml_data['indices']['loc_ind']


### DEFINING CUSTOM LOSS CLASS ###
class MyLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y_true, y_pred):
        act_true = y_true[ : , :act_ind]
        obj_true = y_true[ : , act_ind:obj_ind]
        loc_true = y_true[ : , obj_ind:loc_ind]
        
        act_pred = y_pred[ : , :act_ind]
        obj_pred = y_pred[ : , act_ind:obj_ind]
        loc_pred = y_pred[ : , obj_ind:loc_ind]

        cce = CategoricalCrossentropy()
        loss = cce(act_true, act_pred) + cce(obj_true, obj_pred) + cce(loc_true, loc_pred)
        return loss


### DEFINING CUSTOM ACCURACY CLASS ###
class MyAcc(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.acc = None
        
    def update_state(self, y_true, y_pred, sample_weight = None):
        # Initialization
        act_ind = 6
        obj_ind = 20
        loc_ind = 24
        
        act_true = y_true[ : , :act_ind]
        obj_true = y_true[ : , act_ind:obj_ind]
        loc_true = y_true[ : , obj_ind:loc_ind]

        act_pred = y_pred[ : , :act_ind]
        obj_pred = y_pred[ : , act_ind:obj_ind]
        loc_pred = y_pred[ : , obj_ind:loc_ind]
        
        act_true_labels = tf.math.argmax(act_true, axis = 1)
        act_pred_labels = tf.math.argmax(act_pred, axis = 1)
        
        obj_true_labels = tf.math.argmax(obj_true, axis = 1)
        obj_pred_labels = tf.math.argmax(obj_pred, axis = 1)
        
        loc_true_labels = tf.math.argmax(loc_true, axis = 1)
        loc_pred_labels = tf.math.argmax(loc_pred, axis = 1)
        
        act_correct = tf.cast(act_true_labels == act_pred_labels, "int32")
        obj_correct = tf.cast(obj_true_labels == obj_pred_labels, "int32")
        loc_correct = tf.cast(loc_true_labels == loc_pred_labels, "int32")
        
        sum_acc = tf.math.reduce_sum(act_correct * obj_correct * loc_correct) / len(y_true)
        self.acc = sum_acc
      
    def result(self):
        return self.acc


### DEFINING CUSTOM F1-SCORE FUNCTION ###
def MyF1Score(y_true, y_pred):
    act_true = y_true[ : , :act_ind]
    obj_true = y_true[ : , act_ind:obj_ind]
    loc_true = y_true[ : , obj_ind:loc_ind]

    act_pred = y_pred[ : , :act_ind]
    obj_pred = y_pred[ : , act_ind:obj_ind]
    loc_pred = y_pred[ : , obj_ind:loc_ind]
    
    act_true_labels = np.argmax(act_true, axis = 1)
    act_pred_labels = np.argmax(act_pred, axis = 1)
    
    obj_true_labels = np.argmax(obj_true, axis = 1)
    obj_pred_labels = np.argmax(obj_pred, axis = 1)
    
    loc_true_labels = np.argmax(loc_true, axis = 1)
    loc_pred_labels = np.argmax(loc_pred, axis = 1)
    
    act_f1_score = f1_score(act_true_labels, act_pred_labels, average = 'weighted')
    obj_f1_score = f1_score(obj_true_labels, obj_pred_labels, average = 'weighted')
    loc_f1_score = f1_score(loc_true_labels, loc_pred_labels, average = 'weighted')
    
    sum_f1_score = np.mean([act_f1_score, obj_f1_score, loc_f1_score])
    return [act_f1_score, obj_f1_score, loc_f1_score, sum_f1_score]


### DEFINING A FUNCTIONAL MODEL ###
def create_model():
    # Input Layer
    model_input = keras.Input(shape=(92), name="input")

    # Common Architecture
    x = Dense(units = 64, activation = "relu")(model_input)
    x = Dense(units = 64, activation = "relu")(x)
    x = Dense(units = 32, activation = "relu")(x)
    x = Dense(units = 32, activation = "relu")(x)

    # Different Branches, i.e., action, object and location
    p1 = Dense(units = 32, activation = "relu")(x)
    out1 = Dense(units = 6, activation = "softmax")(p1)

    p2 = Dense(units = 32, activation = "relu")(x)
    out2 = Dense(units = 14, activation = "softmax")(p2)

    p3 = Dense(units = 32, activation = "relu")(x)
    out3 = Dense(units = 4, activation = "softmax")(p3)

    # Concatenate the Outputs
    out = Concatenate(axis = 1)([out1, out2, out3])

    model = tf.keras.Model(model_input, out, name = 'label_model')
    return model

label_model = create_model()

### LOADING THE SAVED ENCODER, VECTORIZER AND MODEL ###
enc = joblib.load(yaml_data['save_path'] + '/one_hot_encoder.joblib')
vectorizer = joblib.load(yaml_data['save_path'] + '/vectorizer.joblib')
label_model.load_weights(yaml_data['save_path'] + '/model.h5')


### PERFORM EVALUATION FOR A GIVEN TEST FILE ###
if is_file:
    # Loading the Data
    if not yaml_data['test_path'].endswith("csv"): 
        raise Exception("The file must be a csv file!")
    df_test = pd.read_csv(yaml_data['test_path'])
    print(f"Test Dataset: {df_test.shape}")
    
    # Pre-Processing the Test Data
    # We don't need the audio files, since we already have the text transcripts
    pre_test = df_test.drop("path", axis = 1)

    # One-Hot Encoding the outputs, i.e., action, object and location
    pre_test_to_encode = pre_test[["action", "object", "location"]]
    Y_test = enc.transform(pre_test_to_encode)
    Y_test = Y_test.toarray()

    # Vectorizing the Transcripts using Binary Bag of Words (BoW)
    X_test = vectorizer.transform(pre_test["transcription"])
    X_test = X_test.toarray()
    
    # Predictions for Test Dataset
    preds_test = label_model.predict(X_test)

    test_f1 = MyF1Score(Y_test, preds_test)
    print(f"/nTest F1 Score: Action | Object | Location | Sum")
    print(f"                 : {test_f1[0]} | {test_f1[1]} | {test_f1[2]} | {test_f1[3]}")
    
    
### OUTPUT PREDICTIONS FOR A GIEVN STRING ###
else: 
    # Loading the Data
    df_test = [sample]
    
    # Vectorizing the Transcript using Binary Bag of Words (BoW)
    X_test = vectorizer.transform(df_test)
    X_test = X_test.toarray()
    
    # Predictions for String
    y_pred = label_model.predict(X_test)
    act_pred = y_pred[ : , :act_ind]
    obj_pred = y_pred[ : , act_ind:obj_ind]
    loc_pred = y_pred[ : , obj_ind:loc_ind]
    
    act_pred_label = np.argmax(act_pred, axis = 1)
    obj_pred_label = np.argmax(obj_pred, axis = 1)
    loc_pred_label = np.argmax(loc_pred, axis = 1)
    
    print(f"/nAction: {enc.categories_[0][act_pred_label][0]}")
    print(f"Object: {enc.categories_[1][obj_pred_label][0]}")
    print(f"Location: {enc.categories_[2][loc_pred_label][0]}")