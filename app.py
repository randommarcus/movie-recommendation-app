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

# Create a Series with movie titles as index and DataFrame indices as values
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def get_recommendations(name, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the name
    idx = indices[name]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['name'].iloc[movie_indices]

