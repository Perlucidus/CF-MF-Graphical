import pandas as pd
import numpy as np
from scipy import sparse
from copy import deepcopy


def load_ratings_mf(ratings: pd.DataFrame):
    rating_matrix = dict()
    user_data = dict()
    movie_data = dict()
    for idx, row in ratings.iterrows():
        u = int(row['userId'])
        m = int(row['movieId'])
        if u not in user_data:
            user_data[u] = []
        if m not in movie_data:
            movie_data[m] = []
        user_data[u].append(m)
        movie_data[m].append(u)
        rating_matrix[u, m] = row['rating']
    return rating_matrix, user_data, movie_data


def filter_ratings(ratings: pd.DataFrame, min_user_ratings=5, min_movie_ratings=0):
    if min_movie_ratings > 0:
        ratings = ratings.groupby('movieId').filter(lambda x: len(x) > min_movie_ratings)
    if min_user_ratings > 0:
        ratings = ratings.groupby('userId').filter(lambda x: len(x) > min_user_ratings)
    return ratings


def serialize_data(ratings: pd.DataFrame):
    users = ratings['userId'].unique()
    movies = ratings['movieId'].unique()
    user2serial = {u: i for i, u in enumerate(users)}
    movie2serial = {m: i for i, m in enumerate(movies)}
    ratings['userId'] = ratings['userId'].apply(user2serial.get)
    ratings['movieId'] = ratings['movieId'].apply(movie2serial.get)
    return ratings


def generalized_train_test_split(ratings: pd.DataFrame, test_size=0.2):
    train = []
    test = []
    for user, user_data in ratings.groupby('userId'):
        idx = np.zeros(len(user_data), dtype=np.bool)
        train_split = np.random.choice(len(user_data), size=int(len(user_data) * test_size), replace=False)
        idx[train_split.astype(np.int64)] = True
        train.append(user_data[~idx])
        test.append(user_data[idx])
    return pd.concat(train), pd.concat(test)


def user_train_test_split(ratings: pd.DataFrame, test_size=0.2):
    users = ratings['userId'].unique()
    np.random.shuffle(users)
    split = int(users.shape[0] * test_size)
    tr = ratings.loc[ratings['userId'].isin(users[split:])]
    te = ratings.loc[ratings['userId'].isin(users[:split])]
    return tr, te
