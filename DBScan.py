# importing dependencies

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

dataFrame = pd.read_csv('./iris.csv')

x = dataFrame[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
# irrespective of species

scaler = StandardScaler()
xSclaed = scaler.fit_transform(x)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dataFrame['DBSCAN_Cluster'] = dbscan.fit_predict(xSclaed)

plt.figure(figsize=(8, 6))
plt.scatter(dataFrame['SepalLengthCm'], dataFrame['SepalWidthCm'], c=dataFrame['DBSCAN_Cluster'], cmap='viridis')
# plt.scatter(dataFrame['sepalLength'], dataFrame['sepalWidth'], s=dataFrame['DBSCAN_Cluster'], cmap='viridis', s='')

plt.title("DBSCAN Clustering", fontsize=18)
plt.xlabel('Sepal Length', fontsize=14)
plt.ylabel('Sepal Width', fontsize=14)
plt.show()
