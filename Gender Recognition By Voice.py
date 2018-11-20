__machine_teacher__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split

source_url = ("https://raw.githubusercontent.com/SerkanKorkusuz/Gender-Recognition-By-Voice/master/voice.csv")

myData = pd.read_csv(source_url, header = 0)
#print(myData.info())
#print(myData)

myData.label = [1 if each == "female" else 0 for each in myData.label]
#print(myData)
#print(myData.info())

label = myData.label.values
att = myData.drop(["label"], axis = 1)

#normalization
att = (att - np.min(att)) / (np.max(att) - np.min(att))
#print(att)

#train test split
att_train, att_test, label_train, label_test = train_test_split(att, label, test_size = 0.3, random_state = 42)
att_train = att_train.T
att_test = att_test.T
label_train = label_train.T
label_test = label_test.T
#print(att_test, "\n", label_test)
#print("att_train: ", att_train.shape)
#print("att_test: ", att_test.shape)
#print("label_train: ", label_train.shape)
#print("label_test: ", label_test.shape)

#initializing weights and bias
def initialize(dimension):
    w = np.full((dimension, 1), 0.01)
    bias = 0.0
    return w, bias

#sigmoid function
def sigmoid(z):
    label_sigmoid = 1.0 / (1 + np.exp(-z))
    return label_sigmoid

#... to be continued
