import inline
import matplotlib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import plotly.express as plotEx
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./iris.csv')
df
df.head()
df.describe()
df.describe(include='all')
df.info()
df['Species'].value_counts()

df.shape
# the type of matrix of data (150, 5) ---> rows x columns

df.columns
# columns

df.isna().sum()
# displays the counts of missing fields

df['Species'].unique()
df['SepalLengthCm'].unique()
df['SepalWidthCm'].unique()
df['PetalLengthCm'].unique()
df['PetalWidthCm'].unique()

# Data Exploration Analysis --->
# Create count plot using seaborn

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
sns.countplot(x='Species', data=df, palette=colors)
plt.show()
# this will create histogram with the count of each

df['SepalLengthCm'].hist(color='green')
df['SepalWidthCm'].hist(color='brown')
df['PetalLengthCm'].hist(color='yellow')
df['PetalWidthCm'].hist(color='red')
# Relationship between SepalLength and Species

plt.figure(figsize=(15, 8))
sns.boxplot(x='Species', y='SepalLengthCm', data=df.sort_values('SepalLengthCm', ascending=False))
# Correlation between Sepal Width and Species

df.plot(kind='scatter', x='SepalWidthCm', y='SepalLengthCm', color='red')
# Correlation between Width and Length of Sepal

sns.jointplot(x='SepalWidthCm', y='SepalLengthCm', data=df, size=5, color='red')
plt.show()
# Paired plotting with Histogram along with Scattered clustering

sns.pairplot(df, hue='Species', size=3)
plt.show()

# Create violin points for each feature

plt.figure(figsize=(14, 8))
for i, column in enumerate(df.columns[1:-1]):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='Species', y=column, data=df, inner='quartile', palette=sns.color_palette('pastel'))
    plt.title(f'{column} distribution by Species')
plt.show()
# Create swarm plots for each feature

plt.figure(figsize=(13, 8))
for i, column in enumerate(df.columns[1:-1]):
    plt.subplot(2, 2, i+1)
    sns.swarmplot(x='Species', y=column, data=df)
    plt.title(f'{column} distribution by class')
plt.show()

