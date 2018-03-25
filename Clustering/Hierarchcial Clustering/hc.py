#Heirarchical Clustering

#Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using the dendrogram to find the optimal no of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#Fitting the hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s = 100, c = 'red',label = 'Careful')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s = 100, c = 'blue',label = 'Standard')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s = 100, c = 'green',label = 'Target')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s = 100, c = 'cyan',label = 'Careless')
plt.scatter(X[y_hc== 4,0],X[y_hc == 4,1], s = 100, c = 'magenta',label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual income (k$s)s')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()