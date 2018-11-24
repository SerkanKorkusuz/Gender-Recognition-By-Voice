__machine_teacher__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split

source_url = ("https://raw.githubusercontent.com/SerkanKorkusuz/Gender-Recognition-By-Voice/master/voice.csv")

myData = pd.read_csv(source_url, header = 0)

myData.label = [1 if each == "female" else 0 for each in myData.label]

label = myData.label.values
att = myData.drop(["label"], axis = 1)
att = (att - np.min(att)) / (np.max(att) - np.min(att))
