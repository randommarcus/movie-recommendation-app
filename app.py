import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('rotten_tomatoes_movies_2025.csv')

# Features for contend-based filtering
features = ['name', 'description','genre', 'type', 'actors','directors', 'content_rating', 'release_date', 'producers', 'synopsis', 'tomatometer_movie_rating', 'tomatometer_ratings_count', 'tomatometer_reviews_count']
for feature in features:
    df[feature] = df[feature].fillna('')

# Combine features into a single string for each movie
def combine_features(row):
    return row['name'] + ' ' + row['description'] + ' ' + row['genre'] + ' ' + row['type'] + ' ' + row['actors'] + ' ' + row['directors'] + ' ' + row['content_rating'] + ' ' + row['release_date'] + ' ' + row['producers'] + ' ' + row['synopsis'] + ' ' + str(row['tomatometer_movie_rating']) + ' ' + str(row['tomatometer_ratings_count']) + ' ' + str(row['tomatometer_reviews_count'])
df['combined_features'] = df.apply(combine_features, axis=1)

# Create a TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)