import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Iris.csv')

df

df.head()

df.info()

df.isnull().sum()

target_data = df.iloc[:, 5]
target_data.head()

clustering_data = df.iloc[:, [1,2,3,4]]
clustering_data.head()

correlation_matrix = clustering_data.corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot = True)
plt.title('CORRELATION MATRIX')
plt.show()

df1 = df[df['Species']=='Iris-setosa']
df2 = df[df['Species']=='Iris-versicolor']
df3 = df[df['Species']=='Iris-virginica']

plt.scatter(df1['PetalLengthCm'],df1['PetalWidthCm'], color='steelblue' , label='Iris-setosa')
plt.scatter(df2['PetalLengthCm'],df2['PetalWidthCm'], color='lightblue', label='Iris-versicolor')
plt.scatter(df3['PetalLengthCm'],df3['PetalWidthCm'], color='navy' , label='Iris-virginica ')

plt.title('SCATTER PLOT ')
plt.legend()
plt.show()

df_imp = df.iloc[:,1:5]
k_meansclus = range(1,10)
sse = []

for k in k_meansclus :
  km = KMeans(n_clusters =k)
  km.fit(df_imp)
  sse.append(km.inertia_)

plt.title('The Elbow Method')
plt.plot(k_meansclus,sse, marker= 'o')
plt.xlabel('Number Of Clusters')
plt.ylabel('sse')
plt.show()

km1 = KMeans(n_clusters=3, random_state=0)
km1.fit(df_imp)

y_means = km1.fit_predict(df_imp)
y_means

centroids = km1.cluster_centers_
print("Centroids:\n", centroids)

df_imp = np.array(df_imp)
df_imp

plt.scatter(df_imp[y_means==0,2 ],df_imp[y_means==0,3 ], color='navy' , label='Iris-versicolor ')
plt.scatter(df_imp[y_means==1,2 ],df_imp[y_means==1,3 ], color='steelblue' , label='Iris-setosa')
plt.scatter(df_imp[y_means==2,2 ],df_imp[y_means==2,3 ], color='lightblue', label='Iris-virginica')
plt.scatter(centroids[:, 2], centroids[:, 3], c='red', label='Centroids')

plt.title('SCATTER PLOT ')
plt.legend()
plt.show()

y_means

from geopy.distance import geodesic

pip install geopy

