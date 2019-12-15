import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from neural_network.neural_network import NeuralNet
from neural_network.activations import *

df = pd.read_csv("data/diabetes.csv")


X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

stdscl = StandardScaler()
X_scaled = stdscl.fit_transform(X)

y_scaled = np.array([[0, 1] if i==1 else [1, 0] for i in y])


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)


np.random.seed(0)
for i in [0.01, 0.05, 0.1, 0.5]:
    nn = NeuralNet()
    nn.add_layer(10, sigmoid, X.shape)
    nn.add_layer(2, softmax)

    history = nn.train(X_train.T, y_train.T, epochs=100, lr=i)

    plt.plot(history, label=str(i))
plt.legend()
plt.show()

accuracy_score(nn.predict(X_test.T), y_test)