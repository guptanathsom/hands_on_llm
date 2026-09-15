import pandas as pd
from urllib import request
from gensim.models import Word2Vec

# 1. Download and parse data
data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')
lines = data.read().decode("utf-8").split('\n')[2:]
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]


songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode("utf-8").split('\n')
songs = [s.rstrip().split('\t') for s in songs_file]

songs_df = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])

songs_df['id'] = songs_df['id'].str.strip()

songs_df = songs_df.set_index('id')

# 2. Train the Word2Vec model on playlists
# Note: gensim 4.0+ uses 'vector_size' (older book code may write 'size')
model = Word2Vec(
    playlists,
    vector_size=32,  # Dimensions of song embedding vectors
    window=20,       # Max distance between current and predicted song in a playlist
    negative=50,     # Negative sampling rate
    min_count=1,     # Include all songs
    workers=4        # Parallel threads
)

# 3. Recommendation Function for a single song
def recommend_similar_songs(song_id, top_n=5):
    song_id = str(song_id)
    similar = model.wv.most_similar(positive=[song_id], topn=top_n)
    similar_ids = [item[0] for item in similar]
    return songs_df.loc[similar_ids]

# Test: Find similar songs to song ID '21'
target_id = '21'
print("Target Song:")
print(songs_df.loc[target_id])
print("\nRecommended Songs:")
print(recommend_similar_songs(target_id))