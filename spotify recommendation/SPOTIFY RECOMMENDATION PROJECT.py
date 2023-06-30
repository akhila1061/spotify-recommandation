#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/akhil/OneDrive/Desktop/spotify dataset.csv')

# Drop irrelevant columns
data = data.drop(['track_id', 'track_name', 'track_artist', 'track_album_id', 'track_album_name', 'track_album_release_date'], axis=1)

# Split features and target variable
X = data.drop(['playlist_genre', 'playlist_subgenre'], axis=1)
y = data[['playlist_genre', 'playlist_subgenre']]

# Encode categorical variables
label_encoder = LabelEncoder()
y['playlist_genre'] = label_encoder.fit_transform(y['playlist_genre'])

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
numeric_cols = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


# Print the pre-processed data
print("X_train:")
print(X_train.head())

print("\ny_train:")
print(y_train.head())

print("\nX_test:")
print(X_test.head())

print("\ny_test:")
print(y_test.head())


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
data = pd.read_csv('C:/Users/akhil/OneDrive/Desktop/spotify dataset.csv')

# Plotting track popularity distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['track_popularity'], bins=20, kde=True)
plt.title('Track Popularity Distribution')
plt.xlabel('Track Popularity')
plt.ylabel('Count')
plt.show()

# Plotting correlation matrix
corr_matrix = data[['track_popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Scatter plot of danceability and energy
plt.figure(figsize=(8, 6))
sns.scatterplot(x='danceability', y='energy', data=data, hue='playlist_genre')
plt.title('Danceability vs Energy')
plt.show()

# Box plot of valence by playlist genre
plt.figure(figsize=(10, 8))
sns.boxplot(x='playlist_genre', y='valence', data=data)
plt.title('Valence by Playlist Genre')
plt.xlabel('Genre')
plt.ylabel('Valence')
plt.xticks(rotation=45)
plt.show()

# Interactive scatter plot of tempo and loudness
fig = px.scatter(data, x='tempo', y='loudness', color='playlist_genre', hover_data=['track_name'])
fig.update_layout(title='Tempo vs Loudness', xaxis_title='Tempo', yaxis_title='Loudness')
fig.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('C:/Users/akhil/OneDrive/Desktop/spotify dataset.csv')

# Select relevant columns for clustering
columns_for_clustering = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                          'instrumentalness', 'liveness', 'valence', 'tempo']

# Filter data for clustering
cluster_data = data[columns_for_clustering]

# Perform feature scaling
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
cluster_data_pca = pca.fit_transform(cluster_data_scaled)

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(cluster_data_scaled)

# Add cluster labels to the data
data['cluster'] = clusters

# Plot clusters based on playlist genres
plt.figure(figsize=(10, 8))
sns.scatterplot(x=cluster_data_pca[:, 0], y=cluster_data_pca[:, 1], hue=data['playlist_genre'], palette='Set1')
plt.title('Clusters based on Playlist Genres')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Plot clusters based on playlist names
plt.figure(figsize=(10, 8))
sns.scatterplot(x=cluster_data_pca[:, 0], y=cluster_data_pca[:, 1], hue=data['playlist_name'], palette='Set1')
plt.title('Clusters based on Playlist Names')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[21]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the Spotify songs dataset
data = pd.read_csv('C:/Users/akhil/OneDrive/Desktop/spotify dataset.csv')

# Select relevant columns for genre segmentation
genre_data = data[['track_id', 'track_name', 'track_artist', 'playlist_genre']]

# Convert genre names to lowercase for consistency
genre_data['playlist_genre'] = genre_data['playlist_genre'].str.lower()

# Convert track names to string type to avoid TypeError
genre_data['track_name'] = genre_data['track_name'].astype(str)

# Group songs by genre and concatenate song names into a single string
genre_songs = genre_data.groupby('playlist_genre')['track_name'].apply(lambda x: ' '.join(x))

# Create TF-IDF matrix based on song names within each genre
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
tfidf_matrix = tfidf.fit_transform(genre_songs)

# Compute cosine similarity between genres
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommended genres for a given genre
def get_recommendations(genre, cosine_similarities, genre_songs):
    # Get the index of the given genre
    genre_index = genre_songs.index.get_loc(genre)
    
    # Get the cosine similarities for the given genre
    genre_similarities = cosine_similarities[genre_index]
    
    # Sort genres based on similarity scores
    similar_genres_indices = genre_similarities.argsort()[::-1]
    
    # Exclude the given genre itself from recommendations
    similar_genres_indices = similar_genres_indices[similar_genres_indices != genre_index]
    
    # Get top 5 similar genres
    top_similar_genres = genre_songs.index[similar_genres_indices][:5]
    
    return top_similar_genres

# Get recommendations for a specific genre
genre_recommendations = get_recommendations('pop', cosine_similarities, genre_songs)

# Print the recommended genres
print(f"Recommended genres for 'pop': {', '.join(genre_recommendations)}")


# In[ ]:




