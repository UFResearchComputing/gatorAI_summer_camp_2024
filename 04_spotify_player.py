# -*- coding: utf-8 -*-
"""04_spotify_player.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/UFResearchComputing/gatorAI_summer_camp_2024/blob/main/04_spotify_player.ipynb

# Spotify Song Player

This notebook authenticates with Spotify and handles the playback of a song.
"""

!pip install spotipy

# Import the necessary libraries
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Used to load Google Credentials
from google.colab import userdata

# Set up Spotipy (Replace with your actual credentials and redirect URI)
client_id = userdata.get('client_id')
client_secret = userdata.get('client_secret')

redirect_uri = 'https://localhost:8888/callback'  # Or any suitable redirect URI

scope = "user-modify-playback-state user-read-playback-state"  # Request permission to control playback

# Create SpotifyOAuth object
sp_oauth = SpotifyOAuth(client_id=client_id, client_secret=client_secret,
                        redirect_uri=redirect_uri, scope=scope, cache_path=None)

# Get authorization URL and prompt the user to authorize
auth_url = sp_oauth.get_authorize_url()
print("Please visit this URL to authorize your application:", auth_url)

# Get authorization code from the user (You'll need a way to capture this)
auth_code = input("Enter the authorization code: ")

# Get access token
token_info = sp_oauth.get_access_token(auth_code)
access_token = token_info['access_token']

# Create Spotify object with the access token
spotify = spotipy.Spotify(auth=access_token)

# Get the active device list
devices = spotify.devices()

def play_song(track_uri):
    if not devices['devices']:
        print("No active devices found. Please make sure Spotify is running on one of your devices.")
    else:
        # Select the first active device
        device_id = devices['devices'][0]['id']
        device_name = devices['devices'][0]['name']
        print(f"Using device: {device_name} with ID {device_id}")

        # Get the URI of the song you want to play
        uri = track_uri

        # Play the song on the selected device
        spotify.start_playback(device_id=device_id, uris=[uri])
        print(f"Started playback on {device_name}")

# Example usage
# play_song("spotify:track:3AFpIlNR4DEJ1WC7qyOHU8")  # Play 'Affection' by Crystal Castles

play_song("spotify:track:3AFpIlNR4DEJ1WC7qyOHU8")

def get_song_uri(song_name, artist_name):
  results = spotify.search(q=f'track:{song_name} artist:{artist_name}', type='track')
  if results['tracks']['items']:
    return results['tracks']['items'][0]['uri']
  else:
    return None

get_song_uri("Happy", "Pharrel Williams")

