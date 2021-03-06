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

#forward and backward propagation
def forward_backward_pro(w, bias, att_train, label_train):
    z = np.dot(w.T, att_train) + bias
    label_sigmoid = sigmoid(z)
    loss = -label_train * np.log(label_sigmoid) - (1 - label_train) * np.log(1 - label_sigmoid)
    cost = (np.sum(loss)) / att_train.shape[1]

    derivative_weight = (np.dot(att_train,((label_sigmoid - label_train).T))) / att_train.shape[1] 
    derivative_bias = np.sum(label_sigmoid - label_train) / att_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients

def recovery(w, bias, att_train, label_train, learning_rate, iteration):
    cost_list = []
    cost_list_interval = []
    index = []
    for i in range(iteration):
        cost, gradients = forward_backward_pro(w, bias, att_train, label_train)
        cost_list.append(cost)
        w = w - learning_rate * gradients["derivative_weight"]
        bias = bias - learning_rate * gradients["derivative_bias"]
        if i % 400 == 0:
            cost_list_interval.append(cost)
            index.append(i)
            print("Cost after {}. iteration is {}.".format(i, cost))
        model_parameters = {"weight" : w, "bias" : bias}
    plot.plot(index, cost_list_interval)
    plot.xticks(index, rotation = "vertical")
    plot.xlabel("Number of Iterations")
    plot.ylabel("Cost")
    plot.show()
    return model_parameters, gradients, cost_list

def prediction(w, bias, att_test):
    label_sigmoid = sigmoid(np.dot(w.T, att_test) + bias)
    label_predicted = np.zeros((1, att_test.shape[1]))
    for j in range(label_sigmoid.shape[1]):
        if label_sigmoid[0, j] <= 0.5:
            label_predicted[0, j] = 0
        else:
            label_predicted[0, j] = 1
    return label_predicted

def logistic_regression(att_train, label_train, att_test, label_test, learning_rate, iteration):
    dimension = att_train.shape[0]
    w, bias = initialize(dimension)
    model_parameters, gradients, cost_list = recovery(w, bias, att_train, label_train, learning_rate, iteration)
    label_predicted = prediction(model_parameters["weight"], model_parameters["bias"], att_test)
    print("Test accuracy: {}".format(100 - np.mean(np.abs(label_predicted - label_test) * 100)))

logistic_regression(att_train, label_train, att_test, label_test, learning_rate = 0.1, iteration = 2000)


#... to be continued
