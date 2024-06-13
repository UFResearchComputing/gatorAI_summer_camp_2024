# %%
# Make a small program that takes an input of an emotion and outputs the title of a song that matches that emotion.

# %%
# Import the necessary libraries
import pandas as pd


# %%
# The ID of the Google Sheet and the gid of the worksheet
sheet_id = "1A8YVKxvEnF9OFdePQqdyBwzUG_o0n-WmJPprzdz74JU"
worksheet_gid = "1445035333"

# Construct the URL to download the Google Sheet as a CSV file
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={worksheet_gid}"

# Download the Google Sheet as a CSV file
df = pd.read_csv(url)

# Save the DataFrame as a CSV file in the current directory
df.to_csv("music_key.csv", index=False)

# %%
df.head()

# %%
# Drop the first column of the DataFrame

df = df.drop(columns=["Timestamp"])

df.head()

# %%
# Replace Column Header 1 with "Angry" and Column Header 2 with "Happy"
df.columns = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

df.head()

# %%
# Create a Function that takes the user input of an emotion and output a random song title that matches that emotion

def get_song_title(emotion):
    # Check if the emotion is "Angry" or "Happy"
    emotion = emotion.capitalize()
    if emotion == "Angry":
        # Get a random song title from the "Angry" column
        song_title = df["Angry"].sample().values[0]
        return song_title, emotion
    elif emotion == "Fear":
        # Get a random song title from the "Fear" column
        song_title = df["Fear"].sample().values[0]
        return song_title, emotion
    elif emotion == "Happy":
        # Get a random song title from the "Happy" column
        song_title = df["Happy"].sample().values[0]
        return song_title, emotion
    elif emotion == "Neutral":
        # Get a random song title from the "Neutral" column
        song_title = df["Neutral"].sample().values[0]
        return song_title, emotion
    elif emotion == "Sad":
        # Get a random song title from the "Sad" column
        song_title = df["Sad"].sample().values[0]
        return song_title, emotion
    elif emotion == "Surprise":
        # Get a random song title from the "Surprise" column
        song_title = df["Surprise"].sample().values[0]
        return song_title, emotion
    else:
        return "Invalid emotion! Please enter one of the following emotions: Angry, Fear, Happy, Neutral, Sad, Surprise"


# %%
# Get the user input of an emotion
emotion = input("Please enter an emotion: ")

# Get the song title that matches the emotion
song_title, emotion = get_song_title(emotion)

# Print the song title
print(f"The song title that matches the emotion '{emotion}' is: {song_title}")

# %%



