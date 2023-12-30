import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# ensemble bagging of Iris dataset

dataFrame = pd.read_csv("./iris.csv")
# change this path according to the path of the file
dataFrame.head(5)
dataFrame.info()
dataFrame.describe()

# id, sepalLengthCm sepalWidthCm petalLengthCm petalWidthCm species

# Converting continuous y('Species') values to categorical

encoder = preprocessing.LabelEncoder()
dataFrame['Species'] = encoder.fit_transform(dataFrame['Species'])

dataFrame.sample(3)
# dataFrame.shape  # Return a tuple representing the dimension of the dataframe
dataFrameTrain = dataFrame.sample(80)
dataFrameValue = dataFrame.sample(30)
dataFrameTest = dataFrame.sample(40)

dataFrameTrain.sample(3)
dataFrameValue.sample(3)
dataFrameTest.sample(3)

xTest = dataFrameValue.iloc[:, [0, 1, 2, 3]].values
yTest = dataFrameValue.iloc[: 4].values

# yTest


def evaluate(clf, x, y):
    clf.fit(x, y)
    plot_tree(clf)
    plt.show()

    plot_decision_regions(x.values, y.values, clf == clf)

    y_prediction = clf.predict(xTest)
    print(accuracy_score(y_prediction, yTest))


# case-1: Bagging


# data for tree 1:

dataFrameBagForCase1 = dataFrameTrain.sample(50, replace=True)

xBagForCase1 = dataFrameBagForCase1.iloc[:, [0, 1, 2, 3]]
yBagForCase1 = dataFrameBagForCase1.iloc[:, 4]

dataFrameBagForCase1.sample(3)

dataBag1 = DecisionTreeClassifier()

evaluate(dataBag1, xBagForCase1, yBagForCase1)


# data for tree 2:

dataFrameBagForCase2 = dataFrameTrain.sample(50, replace=True)

xBagForCase2 = dataFrameBagForCase2.iloc[:, [0, 1, 2, 3]]
yBagForCase2 = dataFrameBagForCase2.iloc[:, [4]]

dataFrameBagForCase2.sample(3)

dataBag2 = DecisionTreeClassifier()

evaluate(dataBag2, xBagForCase2, yBagForCase2)

# data for tree 3:

dataFrameBagForCase3 = dataFrameTrain.sample(50, replace=True)

xBagForCase3 = dataFrameBagForCase3.iloc[:, [0, 1, 2, 3]]
yBagForCase3 = dataFrameBagForCase3.iloc[:, [4]]

dataFrameBagForCase3.sample(3)

dataBag3 = DecisionTreeClassifier()

evaluate(dataBag3, xBagForCase3, yBagForCase3)

print("Predictor For Case 1", dataBag1.predict(np.array([6.1, 3.0, 4.6, 1.4]).reshape(1, 4)))
print("Predictor For Case 2", dataBag2.predict(np.array([6.1, 3.0, 4.6, 1.4]).reshape(1, 4)))
print("Predictor For Case 3", dataBag3.predict(np.array([6.1, 3.0, 4.6, 1.4]).reshape(1, 4)))


# Types of Bagging

# Pasting

# Row sampling without  replacement

dataFrame.sample(5)

# Random Subspaces
dataFrame.sample(2, replace=True, axis=1)

# Random Patches

dataFrame.sample(5, replace=True).sample(2, replace=True, axis=1)

