import numpy as np
import pandas as pd
import scipy.sparse as sparse
import bottleneck as bn
import surprise
from preprocessing import filter_ratings, user_train_test_split
from copy import deepcopy


alpha = 0.75
reg = 6


def split_users(data, train_proportion=0.8):
    users = data.groupby('userId')
    tr, te = [], []
    for _, user_data in users:
        idx = np.zeros(len(user_data), dtype=np.bool)
        train_split = np.random.choice(len(user_data), size=int(len(user_data) * train_proportion), replace=False)
        idx[train_split.astype(np.int64)] = True
        tr.append(user_data[idx])
        te.append(user_data[~idx])
    return pd.concat(tr), pd.concat(te)


def serialize(ratings, user2serial, movie2serial):
    user_id = ratings['userId'].apply(user2serial.get)
    movie_id = ratings['movieId'].apply(movie2serial.get)
    return pd.DataFrame(data={'userId': user_id, 'movieId': movie_id}, columns=['userId', 'movieId'])


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    tp = 1 / np.log2(np.arange(2, k + 2))
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def main(movielens_version):
    surprise_data = surprise.Dataset.load_builtin(movielens_version)
    data = pd.DataFrame(surprise_data.raw_ratings, columns=['userId', 'movieId', 'rating', 'timestamp'])
    data.drop(columns=['timestamp'], inplace=True)
    data = filter_ratings(data)
    train, test = user_train_test_split(data)
    unique_users = data['userId'].unique()
    unique_train_movies = train['movieId'].unique()
    user2serial = {u: i for i, u in enumerate(unique_users)}
    movie2serial = {m: i for i, m in enumerate(unique_train_movies)}
    test = test[test['movieId'].isin(unique_train_movies)]
    test_movies_tr, test_movies_te = split_users(test)
    train = serialize(train, user2serial, movie2serial)
    test_movies_tr = serialize(test_movies_tr, user2serial, movie2serial)
    test_movies_te = serialize(test_movies_te, user2serial, movie2serial)
    train = sparse.csr_matrix((np.ones_like(train['userId']), (train['userId'], train['movieId'])))
    test_movies_tr = sparse.csr_matrix((np.ones_like(test_movies_tr['userId']),
                                        (test_movies_tr['userId'], test_movies_tr['movieId'])),
                                       shape=(test_movies_tr['userId'].max() + 1, len(unique_train_movies)))
    test_movies_te = sparse.csr_matrix((np.ones_like(test_movies_te['userId']),
                                        (test_movies_te['userId'], test_movies_te['movieId'])),
                                       shape=(test_movies_te['userId'].max() + 1, len(unique_train_movies)))
    corr = np.asarray(train.T.dot(train).todense(), dtype=np.float32)
    mu = np.diag(corr) / train.shape[0]
    scaled_var = np.diag(corr) - train.shape[0] * np.square(mu)
    corr -= mu[:, np.newaxis] * (mu * train.shape[0])
    rescaling = np.power(scaled_var, alpha / 2)
    scaling = 1 / rescaling
    corr = scaling[:, np.newaxis] * corr * scaling
    corr_diag = deepcopy(np.diag(corr))
    diag_ind = np.diag_indices(corr.shape[0])
    corr[diag_ind] = corr_diag + reg
    mrf = np.linalg.inv(corr)
    mrf /= -np.diag(mrf)
    mrf[diag_ind] = 0
    mrf = scaling[:, np.newaxis] * mrf * rescaling
    x = test_movies_tr.toarray()
    predictions = x @ mrf
    predictions[x.nonzero()] = -np.inf
    print(np.nanmean(NDCG_binary_at_k_batch(predictions, test_movies_te)))


if __name__ == '__main__':
    main('ml-100k')
    main('ml-1m')
