# %%
# Import Libraries 
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline # Commented out for use in Google Colab

# Import Libraries for Unsupervised Learning on Tabular Data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # Standardize features by removing the mean and scaling to unit variance
from sklearn.pipeline import make_pipeline

import spot_creds
import spotipy

# Load Spotify credentials
client_id = spot_creds.client_id
client_secret = spot_creds.client_secret

# %%
# Create a function to load the data from two csv files and concatenate them
def load_data():
    # Load the data from the two csv files
    data1 = pd.read_csv('data/song_tables/SpotifyAudioFeaturesApril2019.csv')
    data2 = pd.read_csv('data/song_tables/SpotifyAudioFeaturesNov2018.csv')
    # Concatenate the two dataframes
    data = pd.concat([data1, data2], axis=0)
    return data

df = load_data()
df.head()



# %%
# Preprocess the data
def preprocess_data(df):
    df = df.dropna()
    numeric_df = df.drop(['track_id', 'track_name', 'artist_name'], axis=1)
    scaler = StandardScaler()
    pp_df = scaler.fit_transform(numeric_df)
    return pp_df, df, scaler

pp_df, original_df, scaler = preprocess_data(df)

# Fit the KMeans model
def fit_kmeans(df, n_clusters):
    model = KMeans(n_clusters=n_clusters)
    model.fit(df)
    return model

model = fit_kmeans(pp_df, 10)


# %%
# Visualize the clusters
def visualize_clusters(df, model):
    pca = PCA(n_components=2)
    pc = pca.fit_transform(df)
    plt.scatter(pc[:, 0], pc[:, 1], c=model.labels_)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('KMeans Clusters')
    plt.show()

visualize_clusters(pp_df, model)

# %%
# Show a sample of songs from each cluster
def show_cluster_samples(df, model, n_samples):
    df['cluster'] = model.labels_
    for cluster in range(model.n_clusters):
        print(f'Cluster {cluster}:')
        sample = df[df['cluster'] == cluster].sample(n=n_samples)
        print(sample[['track_name', 'artist_name', 'popularity']])
        print('\n')

show_cluster_samples(original_df, model, 3)


# %%
# Visualize clusters with most popular songs
def visualize_clusters_with_songs(df, original_df, model):
    pca = PCA(n_components=2)
    pc = pca.fit_transform(df)
    plt.scatter(pc[:, 0], pc[:, 1], c=model.labels_)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('KMeans Clusters')
    for cluster in range(model.n_clusters):
        cluster_indices = np.where(model.labels_ == cluster)[0]
        cluster_center = pc[cluster_indices].mean(axis=0)
        popular_song = original_df[original_df['cluster'] == cluster].sort_values('popularity', ascending=False).iloc[0]
        plt.text(cluster_center[0], cluster_center[1], popular_song['track_name'], fontsize=12)
    plt.show()

# Add 'cluster' column to the original dataframe
original_df['cluster'] = model.labels_
visualize_clusters_with_songs(pp_df, original_df, model)

# %%
# Recommend a song from a given cluster
def recommend_songs(df, model, cluster):
    indices = np.where(df['cluster'] == cluster)[0]
    songs = df.iloc[indices]
    song = songs.sample()
    return song[['track_name', 'artist_name', 'popularity']]

# Recommend a song from cluster 0
print(recommend_songs(original_df, model, 0))

# %%
'''This Block creates a function to:
1. Take a song name and artist name as input.
2. Get audio features for a song using the Spotify API.
3. Run the KMeans model to predict the cluster for the song.
4. Recommend a song from the same cluster as the input song.
'''

# Create a function to get audio features for a song using the Spotify API
def get_audio_features_from_spotify(song_name, artist_name, client_id, client_secret):
    # Initialize Spotipy API client
    sp = spotipy.Spotify(client_credentials_manager=spotipy.oauth2.SpotifyClientCredentials(client_id, client_secret))
    # Search for the song
    results = sp.search(q=f'track:{song_name} artist:{artist_name}', limit=1)
    # Get the track ID
    track_id = results['tracks']['items'][0]['id']
    # Get the audio features
    audio_features = sp.audio_features(track_id)[0]
    return audio_features

'''# Create a function to recommend a song from the same cluster as the input song
def recommend_song_from_cluster(df, model, song_name, artist_name, client_id, client_secret):
    # Get audio features for the input song
    audio_features = get_audio_features_from_spotify(song_name, artist_name, client_id, client_secret)
    # Preprocess the audio features
    scaler = StandardScaler()
    pp_audio_features = scaler.fit_transform(np.array(list(audio_features.values())[2:-1]).reshape(1, -1))
    # Predict the cluster for the input song
    cluster = model.predict(pp_audio_features)[0]
    # Recommend a song from the same cluster
    return recommend_songs(df, model, cluster)'''

def recommend_song_from_cluster(df, model, song_name, artist_name, client_id, client_secret):
    # Get audio features for the input song
    audio_features = get_audio_features_from_spotify(song_name, artist_name, client_id, client_secret)
    
    # Specify the relevant audio feature keys (example keys, adjust based on your model's needs)
    audio_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'popularity']
    relevant_features = [audio_features[key] for key in audio_keys if key in audio_features]
    
    # Preprocess the audio features
    scaler = StandardScaler()
    pp_audio_features = scaler.fit_transform(np.array(relevant_features).reshape(1, -1))
    
    # Predict the cluster for the input song
    cluster = model.predict(pp_audio_features)[0]
    
    # Recommend a song from the same cluster
    return recommend_songs(df, model, cluster)

# Test the function with "Affection" by Crystal Castles
print(recommend_song_from_cluster(original_df, model, 'Affection', 'Crystal Castles', client_id, client_secret))

# %%
# Create a function to get audio features for a song using the Spotify API
def get_audio_features_from_spotify(song_name, artist_name, client_id, client_secret):
    # Initialize Spotipy API client
    sp = spotipy.Spotify(client_credentials_manager=spotipy.oauth2.SpotifyClientCredentials(client_id, client_secret))
    # Search for the song
    results = sp.search(q=f'track:{song_name} artist:{artist_name}', limit=1)
    # Handle the case where the song is not found
    if not results['tracks']['items']:
        return None
    # Get the track ID
    track_id = results['tracks']['items'][0]['id']
    # Get the audio features
    audio_features = sp.audio_features(track_id)[0]
    return audio_features

# Create a function to recommend a song from the same cluster as the input song
def recommend_song_from_cluster(df, model, song_name, artist_name, client_id, client_secret):
    # Get audio features for the input song
    audio_features = get_audio_features_from_spotify(song_name, artist_name, client_id, client_secret)
    if not audio_features:
        return "Song not found in Spotify database."
    
    # Specify the relevant audio feature keys
    audio_keys = [ 
        'acousticness', 'danceability', 'duration_ms', 'energy', 
        'instrumentalness', 'key', 'liveness', 'loudness', 
        'mode', 'speechiness', 'tempo' ,'time_signature',
        'valence', 'popularity'
    ]
    
    # Ensure all keys are present in the audio features and fill missing keys with 0
    relevant_features = [audio_features.get(key, 0) for key in audio_keys]
    
    # Preprocess the audio features using the existing scaler
    pp_audio_features = scaler.transform(np.array(relevant_features).reshape(1, -1))
    
    # Predict the cluster for the input song
    cluster = model.predict(pp_audio_features)[0]
    
    # Recommend a song from the same cluster
    return recommend_songs(df, model, cluster)

# Test the function with "Affection" by Crystal Castles
print(recommend_song_from_cluster(original_df, model, 'Affection', 'Crystal Castles', client_id, client_secret))



# %%



