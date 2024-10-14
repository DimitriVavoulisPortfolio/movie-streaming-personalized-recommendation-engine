import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import os

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file = os.path.join(current_dir, 'ratings.csv')
movies_file = os.path.join(current_dir, 'movies.csv')
output_file = os.path.join(current_dir, 'preprocessed_data.npz')

# Load data
ratings_df = pd.read_csv(ratings_file)
movies_df = pd.read_csv(movies_file)

# Merge dataframes
df = pd.merge(ratings_df, movies_df, on='movieId')

# Convert ratings to binary (1 for ratings >= 3, 0 otherwise)
df['rating_binary'] = (df['rating'] >= 3).astype(int)

# Handle missing values
df = df.dropna()

# Process genres
df['genres'] = df['genres'].fillna('').apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df['genres'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

# Process temporal data
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['time_of_day'] = df['timestamp'].dt.hour

# Create user and movie features
user_ratings = df.groupby('userId')['rating'].agg(['mean', 'count']).reset_index()
user_ratings.columns = ['userId', 'user_avg_rating', 'user_rating_count']

movie_ratings = df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_ratings.columns = ['movieId', 'movie_avg_rating', 'movie_rating_count']

df = pd.merge(df, user_ratings, on='userId')
df = pd.merge(df, movie_ratings, on='movieId')

# Prepare final dataset
features = ['userId', 'movieId', 'day_of_week', 'time_of_day', 
            'user_avg_rating', 'user_rating_count', 
            'movie_avg_rating', 'movie_rating_count']
X = pd.concat([df[features], genre_df], axis=1)
y = df['rating_binary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
np.savez(output_file, 
         X_train=X_train.values, y_train=y_train.values,
         X_test=X_test.values, y_test=y_test.values,
         feature_names=X.columns.values,
         genre_names=mlb.classes_)

print(f"Preprocessed data saved to {output_file}")
print(f"Shape of training data: {X_train.shape}")
print(f"Shape of testing data: {X_test.shape}")
