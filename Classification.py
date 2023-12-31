import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
import plotly.express as pltex
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./iris.csv')
# df
df.head()
df.describe()
df.describe(include='all')
df.info()
df['Species'].value_counts()

# df.shape
# the type of matrix of data (150, 5) ---> rows x columns

# df.columns
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
    plt.subplot(2, 2, i + 1)
    sns.violinplot(x='Species', y=column, data=df, inner='quartile', palette=sns.color_palette('pastel'))
    plt.title(f'{column} distribution by Species')
plt.show()
# Create swarm plots for each feature

plt.figure(figsize=(13, 8))
for i, column in enumerate(df.columns[1:-1]):
    plt.subplot(2, 2, i + 1)
    sns.swarmplot(x='Species', y=column, data=df)
    plt.title(f'{column} distribution by class')
plt.show()

# Set the overall figure size i.e. the range in which the datapoints lie
plt.figure(figsize=(15, 15))

# Subplot 1
plt.subplot(4, 2, 1)
fig = df.boxplot(column='SepalLengthCm', color='red')
fig.set_title('Boxplot of Sepal Length')
fig.set_ylabel('SepalLengthCm')

# Subplot 2
plt.subplot(4, 2, 2)
fig = df.boxplot(column='SepalWidthCm', color='red')
fig.set_title('Boxplot of Sepal Width')
fig.set_ylabel('SepalWidthCm')

# Subplot 3
plt.subplot(4, 2, 3)
fig = df.boxplot(column='PetalLengthCm', color='red')
fig.set_title('Boxplot of Petal Length')
fig.set_ylabel('PetalLengthCm')

# Subplot 4
plt.subplot(4, 2, 4)
fig = df.boxplot(column='PetalWidthCm', color='red')
fig.set_title('Boxplot of Petal Width')
fig.set_ylabel('PetalWidthCm')

plt.show()

# Making a histogram chart to see how things are spread out

plt.figure(figsize=(24, 20))

# Subplot 1
plt.subplot(4, 2, 1)
fig = df['SepalLengthCm'].hist(bins=10, color='green')
fig.set_xlabel('IP Mean')
fig.set_title('Sepal Length')

# Subplot 2
plt.subplot(4, 2, 2)
fig = df['SepalWidthCm'].hist(bins=10, color='green')
fig.set_xlabel('Sepal Width')
fig.set_title('Sepal Width')

# Subplot 3
plt.subplot(4, 2, 3)
fig = df['PetalLengthCm'].hist(bins=10, color='green')
fig.set_xlabel('Petal Length')
fig.set_title('Petal Length')

# Subplot 4
plt.subplot(4, 2, 4)
fig = df['PetalWidthCm'].hist(bins=10, color='green')
fig.set_xlabel('Petal Width')
fig.set_title('Petal Width')

plt.show()

# Create a histogram with plotly
# Opens on a localhost server

fig1 = pltex.histogram(df, x='Species', color='SepalLengthCm')
fig1.show()

fig2 = pltex.histogram(df, x='Species', color='SepalWidthCm')
fig2.show()

fig3 = pltex.histogram(df, x='Species', color='PetalLengthCm')
fig3.show()

fig4 = pltex.histogram(df, x='Species', color='PetalWidthCm')
fig4.show()


# Create a scatter plot with 2 variables to distinguish Species

def scatter_plot(x, y):
    global df

    plt.figure(figsize=(15, 6))
    for Species in df['Species'].unique():
        plt.scatter(df[x][df['Species'] == Species], df[y][df['Species'] == Species], label=Species, edgecolor='k',
                    alpha=0.7)
    plt.xticks(rotation=0)

    plt.title("Scatter plot X:{} / Y:{}".format(x, y))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.show()


comb = combinations(['Species', 'PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm', 'SepalWidthCm'], 2)
combList = [list(i) for i in comb]

for col in combList:
    scatter_plot(col[0], col[1])

# Correlation Matrix computation

encodedDf = pd.get_dummies(df, columns=['Species'])
correlationMatrix = encodedDf.corr()
print(correlationMatrix)
# Display correlation using Heat Map

numericDf = df.select_dtypes(include=['number'])
correlationNumberMatrix = numericDf.corr()
print(correlationNumberMatrix)

# Creating Plot for it ---->
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(correlationNumberMatrix, annot=True, ax=ax, cmap='cividis')
plt.show()

# Model Training

xData = df.drop(columns=['Species'])
yData = df['Species']

# Splitting the dataset into training and test sets --->
xDataTrain, xDataTest, yDataTrain, yDataTest = train_test_split(xData, yData, test_size=0.3, random_state=1)

# Logistic Regression Model --->
model1 = LogisticRegression()
model1.fit(xDataTrain, yDataTrain)
acuLR = model1.score(xDataTest, yDataTest) * 100
print("Accuracy (Logistic Regression): ", acuLR)
# This will show us the accuracy of the Logistic Regression

# K-Nearest Neighbors
model2 = KNeighborsClassifier()
model2.fit(xDataTrain, yDataTrain)
acuKNC = model2.score(xDataTest, yDataTest) * 100
print("Accuracy (KNeighbors Classifier): ", acuKNC)

# Random Forest Classifier
model3 = RandomForestClassifier()
model3.fit(xDataTrain, yDataTrain)
acuRFC = model3.score(xDataTest, yDataTest) * 100
print("Accuracy (Random Forest Classifier): ", acuRFC)

# Decision Tree Classifier
model4 = DecisionTreeClassifier()
model4.fit(xDataTrain, yDataTrain)
acuDT = model4.score(xDataTest, yDataTest) * 100
print("Accuracy (Decision Tree Classifier): ", acuDT)

# The models are trained on the specific training data and tested
# Visualising the accuracy

plt.figure(figsize=(12, 6))
modelAccuracy = [acuLR, acuKNC, acuRFC, acuDT]
modelName = ['Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree']
plt.xlabel('Accuracy')
plt.ylabel('Models')
sns.barplot(x=modelAccuracy, y=modelName, palette='plasma')

# Conclusion: The Logistic Regression and the KNN models performed best achieving accuracy of 97.77%
