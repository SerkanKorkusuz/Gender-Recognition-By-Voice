__machine_teacher__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

source_url = ("https://raw.githubusercontent.com/SerkanKorkusuz/Gender-Recognition-By-Voice/master/voice.csv")

myData = pd.read_csv(source_url, header = 0)

myData.label = [1 if each == "female" else 0 for each in myData.label]

label = myData.label.values
att = myData.drop(["label"], axis = 1)
att = (att - np.min(att)) / (np.max(att) - np.min(att))

att_train, att_test, label_train, label_test = train_test_split(att, label, test_size = 0.3, random_state = 42)

my_model = LogisticRegression()
my_model.fit(att_train, label_train)
print("Test accuracy (with the help of skicit-learn class): {}".format(my_model.score(att_test.T, label_test.T) * 100))
