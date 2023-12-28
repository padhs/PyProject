# importing all dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')                                                                                                     

dataFrame = pd.read_csv("./iris.csv")
dataFrame.head(5)
dataFrame.info()
dataFrame.describe()

# They describe what type of data is there in the first rows i.e id, sepalLengthCm sepalWidthCm petalLengthCm petalWidthCm species and their respective datatypes
# Run the project to check the description of the data (iris.csv)
# Run all these commands one by one to see them separately

encoder = LabelEncoder()
#dataFrame['species'] = encoder.fit_transform(dataFrame['species'])
dataFrame.sample(3)


