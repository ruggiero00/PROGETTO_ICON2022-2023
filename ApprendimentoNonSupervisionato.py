import plotly.express as px
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from scipy.spatial import distance
from sklearn.metrics import classification_report, confusion_matrix
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings("ignore")



###CLUSTERING

df = pd.read_csv("heart2.csv")
df.head()
print(df)

# x e y dati delle colonne DataFrame
fig = px.scatter(df, x='chol', y="trtbps", color='output', symbol="sex")
fig.show()

fig = px.scatter_3d(df, x='oldpeak', y="age",z='cp', color='output', symbol="sex")
fig.show()

###Cluster Analysis
###K-Means

# Normalizzazione dataframe
def normalize(df, features):
    result = df.copy()
    for feature_name in features:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

normalised_df = normalize(df,df.columns)

X_train, X_test, y_train, y_test = train_test_split( normalised_df.drop(["output"], axis=1),
normalised_df["output"], test_size=0.33, random_state=42)


wcss = []
clusters = 50
for i in range(1, clusters):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=500, n_init=10, random_state=123)
    kmeans.fit(X_train.values)
    wcss.append(kmeans.inertia_)

fig = go.Figure(data=go.Scatter(x=[i for i in range(1, clusters)], y=wcss))

fig.update_layout(title='WCSS vs. Cluster number',
                  xaxis_title='Clusters',
                  yaxis_title='WCSS')
fig.show()

kmeans = KMeans(n_clusters = 4, init="k-means++", max_iter = 500, n_init = 10, random_state = 42)
identified_clusters = kmeans.fit_predict(X_train.values)



copy_X_train = X_train.copy()
copy_X_train['Cluster'] = identified_clusters

distance_from_centroid = []
for values in  copy_X_train.values:
    distance_from_centroid.append(distance.euclidean(values[:-1],kmeans.cluster_centers_[int(values[-1])]))

copy_X_train['distance_from_centriods'] = distance_from_centroid

fig = px.scatter_3d(copy_X_train,  x='thalachh', y="oldpeak",z='age',
              color='Cluster', opacity = 0.8, size='distance_from_centriods',symbol=y_train, size_max=30)
fig.show()


for i in range(1,3):
    knn1 = KMeans(i)
    knn1.fit(X_train,y_train)
    pred1 = knn1.predict(X_test)
    print(f"For Knn-{i}: \n")
    print(classification_report(y_test,pred1))
    print('====================================')

km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2, 10))
visualizer.fit(normalised_df)  # Adatta i dati al visualizzatore
visualizer.show()

model=kmeans.fit(X_train)
pred_test = model.predict(X_test)
pred_train = model.predict(X_train)
print(pred_test)
print(pred_train)


