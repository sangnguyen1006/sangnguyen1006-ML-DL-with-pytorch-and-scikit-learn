import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from perceptron import Perceptron

# Get iris dataset
link_2data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(link_2data, header=None, encoding='utf-8')
# Select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
# Extract sepal lenght and petal lenght
X = df.iloc[0:100, [0, 2]].values
# Plot data
# plt.scatter(X[:50, 0], X[:50, 1],
#             color='red', marker='o', label='Setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1],
#             color='blue', marker='s', label='Versicolor')
# plt.xlabel('Sepal lenght [cm]')
# plt.ylabel('Petal lenght [cm]')
# plt.legend(loc='upper left')
# plt.show()

# Training
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1),
#          ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of update')
# plt.show()