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


### GETTING THE CONFIG FILE PATH ###
if len(sys.argv) < 3 or len(sys.argv) > 3:
  raise Exception("Invalid Arguments Passed!")
elif sys.argv[1] != '--config':
  raise Exception("Please provide the config path!")

config_path = sys.argv[2]

with open(f"{config_path}/config.yaml") as file:
  yaml_data = yaml.safe_load(file)
  
  
### LOAD THE TENSORBOARD NOTEBOOK EXTENSION ###
tracking_address = yaml_data['log_path']
if not os.path.isdir(tracking_address):
  os.mkdir(tracking_address)

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tracking_address])
url = tb.launch()
print(f"Tensorflow listening on {url}")


### LOADING THE DATA ###
data_path = yaml_data['dataset_path']
df_train = pd.read_csv(f"{data_path}/train_data.csv")
df_valid = pd.read_csv(f"{data_path}/valid_data.csv")
print(f"Training Dataset: {df_train.shape}")
print(f"Validation Dataset: {df_valid.shape}")


### PREPROCESSING THE TRAINING DATA ###
# We don't need the audio files, since we already have the text transcripts
pre_train = df_train.drop("path", axis = 1)

# One-Hot Encoding the outputs, i.e., action, object and location
pre_train_to_encode = pre_train[["action", "object", "location"]]
enc = OneHotEncoder(handle_unknown='ignore')
Y_train = enc.fit_transform(pre_train_to_encode)
Y_train = Y_train.toarray()

# Saving the One Hot Encoder
if not os.path.isdir(yaml_data['save_path']):
  os.mkdir(yaml_data['save_path'])
joblib.dump(enc, yaml_data['save_path'] + "/one_hot_encoder.joblib")

# Vectorizing the Transcripts using Binary Bag of Words (BoW)
vectorizer = CountVectorizer(binary = True)
X_train = vectorizer.fit_transform(pre_train["transcription"])
X_train = X_train.toarray()

# Saving the Count Vectorizer
joblib.dump(vectorizer, yaml_data['save_path'] + "/vectorizer.joblib")


### PREPROCESSING THE VALIDATION DATA ###
# We don't need the audio files, since we already have the text transcripts
pre_val = df_valid.drop("path", axis = 1)

# One-Hot Encoding the outputs, i.e., action, object and location
pre_val_to_encode = pre_val[["action", "object", "location"]]
Y_val = enc.transform(pre_val_to_encode)
Y_val = Y_val.toarray()

# Vectorizing the Transcripts using Binary Bag of Words (BoW)
X_val = vectorizer.transform(pre_val["transcription"])
X_val = X_val.toarray()


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


### COMPILING THE MODEL ##
label_model.compile(
    loss=MyLoss(),
    optimizer='adam',
    metrics=MyAcc(),
)

log_dir = yaml_data['log_path'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

### TRAINING THE MODEL ###
label_model.fit(
  X_train, Y_train, batch_size = yaml_data['batch_size'], epochs = yaml_data['epochs'], 
  callbacks = [tensorboard_callback]
)

# Saving the Model
label_model.save_weights(yaml_data['save_path'] + "/model.h5")
# label_model.save(yaml_data['save_path'] + "/model.h5")

### PREDICTIONS FOR TRAIN AND VALIDATION DATASETS ###
preds_train = label_model.predict(X_train)
preds_val = label_model.predict(X_val)

train_f1 = MyF1Score(Y_train, preds_train)
print(f"\nTraining F1 Score: Action | Object | Location | Sum")
print(f"                 : {train_f1[0]} | {train_f1[1]} | {train_f1[2]} | {train_f1[3]}")

val_f1 = MyF1Score(Y_val, preds_val)
print(f"\nValidation F1 Score: Action | Object | Location | Sum")
print(f"                   : {val_f1[0]} | {val_f1[1]} | {val_f1[2]} | {val_f1[3]}")