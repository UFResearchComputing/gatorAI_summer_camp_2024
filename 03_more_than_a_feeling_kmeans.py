# -*- coding: utf-8 -*-
"""03_more_than_a_feeling_kmeans.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/UFResearchComputing/gatorAI_summer_camp_2024/blob/main/03_more_than_a_feeling_kmeans.ipynb

# More Than a Feeling: K-Means

This notebook introduces unsupervised learning and clustering methods. It created clusters of songs based on Spotify features like dancability and popularity.
"""

!pip install spotipy

# Commented out IPython magic to ensure Python compatibility.
# Import Libraries
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# Import Libraries for Unsupervised Learning on Tabular Data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # Standardize features by removing the mean and scaling to unit variance
from sklearn.pipeline import make_pipeline

# Used to load Google Credentials
from google.colab import userdata
#import spot_creds
import spotipy

# Download the data from Kaggle and unzip into the data folder

!kaggle datasets download -d tomigelo/spotify-audio-features
!mkdir data
!unzip spotify-audio-features.zip -d data

# Create a function to load the data from two csv files and concatenate them
def load_data():
    # Load the data from the two csv files
    data1 = pd.read_csv('data/SpotifyAudioFeaturesApril2019.csv')
    data2 = pd.read_csv('data/SpotifyAudioFeaturesNov2018.csv')
    # Concatenate the two dataframes
    data = pd.concat([data1, data2], axis=0)
    return data

df = load_data()
df.head()

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

# Show a sample of songs from each cluster
def show_cluster_samples(df, model, n_samples):
    df['cluster'] = model.labels_
    for cluster in range(model.n_clusters):
        print(f'Cluster {cluster}:')
        sample = df[df['cluster'] == cluster].sample(n=n_samples)
        print(sample[['track_name', 'artist_name', 'popularity']])
        print('\n')

show_cluster_samples(original_df, model, 3)

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

# Recommend a song from a given cluster
def recommend_songs(df, model, cluster):
    indices = np.where(df['cluster'] == cluster)[0]
    songs = df.iloc[indices]
    song = songs.sample()
    return song[['track_name', 'artist_name', 'popularity']]

# Recommend a song from cluster 0
print(recommend_songs(original_df, model, 0))

# Recommend a song based on a given song from the model
def recommend_song(df, model, song_name):
    song = df[df['track_name'] == song_name].iloc[0]
    # Ensure we drop the same columns that were dropped during preprocessing
    song_features = song.drop(['track_id', 'track_name', 'artist_name', 'cluster']) # 'popularity'
    # Convert to DataFrame with appropriate column names
    song_features_df = pd.DataFrame([song_features], columns=song_features.index)
    song_features_scaled = scaler.transform(song_features_df)
    cluster = model.predict(song_features_scaled)[0]
    return recommend_songs(df, model, cluster)

# Recommend a song based on 'Shape of You' by Ed Sheeran
print(recommend_song(original_df, model, 'Breath'))

# Load Spotify credentials--replace part in quotes with what you called the credentials in the credential manager
# A pop up will ask you to grant access to the credentials if you don't have that already selected

client_id = userdata.get('spotify_client_id_matt')
client_secret = userdata.get('spotify_client_secret_matt')

# Pull audio features for a given song from Spotify and return them in a format that can be used by the model
def get_audio_features(track_name, artist_name, client_id, client_secret):
    # Initialize Spotipy client
    sp = spotipy.Spotify(client_credentials_manager=spotipy.oauth2.SpotifyClientCredentials(client_id, client_secret))
    # Search for the track
    results = sp.search(q=f'track:{track_name} artist:{artist_name}', type='track', limit=1)
    # Get the track ID
    track_id = results['tracks']['items'][0]['id']
    # Get the audio features
    audio_features = sp.audio_features(track_id)[0]
    # Get the popularity
    popularity = results['tracks']['items'][0]['popularity']
    # Convert the audio features and popularity to a DataFrame
    audio_features_df = pd.DataFrame([audio_features])
    audio_features_df['popularity'] = popularity
    # Drop irrelevant columns
    audio_features_df = audio_features_df.drop(['type', 'id', 'uri', 'track_href', 'analysis_url'], axis=1)
    # Rearrange columns to match the original DataFrame
    audio_features_df = audio_features_df[['acousticness','danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence', 'popularity']]
    return audio_features_df

# Get audio features for 'With or Without You' by U2
audio_features = get_audio_features('With or Without You', 'U2', client_id, client_secret)
print(audio_features)

# Recommend a song based on the new song 'With or Without You' by U2
def recommend_song_from_audio_features(df, model, audio_features, scaler):
    # Scale the audio features
    audio_features_scaled = scaler.transform(audio_features)
    # Predict the cluster
    cluster = model.predict(audio_features_scaled)[0]
    return recommend_songs(df, model, cluster)

# Recommend a song based on 'With or Without You' by U2
print(recommend_song_from_audio_features(original_df, model, audio_features, scaler))

# Cheese