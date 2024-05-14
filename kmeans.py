import pandas as pd
import glob
import matplotlib.pyplot as plt
import os 

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def read_csv(csv_files):
    
    dataX = pd.DataFrame()
    dataY = pd.DataFrame()
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            dataX = pd.concat([dataX, df.iloc[:, 1:7]], ignore_index=True)
            dataY = pd.concat([dataY, df.iloc[:, -1]], ignore_index=True)
            
        else:
            print("File not found")
        
    return dataX,dataY


# Directory containing the CSV files
directory = 'harth/'
csv_pattern = '*.csv'

# Find all CSV files in the directory
csv_files = glob.glob(directory + csv_pattern)

X, Y = read_csv(csv_files[0:5])

# Transforming the data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(X)

# Creating the k-means clustering
k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(normalized_features)

X['cluster'] = clusters

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(normalized_features)
reduced_data = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
reduced_data['cluster'] = clusters

# Creating the scatter plot
import matplotlib.pyplot as plt
plt.scatter(reduced_data['PC1'], reduced_data['PC2'], c=reduced_data['cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering')
plt.colorbar(label='Cluster')
plt.show()