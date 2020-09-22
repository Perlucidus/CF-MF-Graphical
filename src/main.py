from gmf import GaussianMF
from pmf import PoissonMF
from orf import OrdinalRandomFields
from preprocessing import load_ratings_mf, filter_ratings, serialize_data, generalized_train_test_split
from sklearn.model_selection import train_test_split
import surprise
import pandas as pd
import numpy as np


def evaluate(users, ratings, model):
    predictions = {(u, i): model.predict(u, i) for u, i in ratings}
    rmse = np.sqrt(sum(np.square(predictions[u, m] - r) for (u, m), r in ratings.items()) / len(ratings))
    mae = sum(np.abs(predictions[u, m] - r) for (u, m), r in ratings.items()) / len(ratings)
    user_ratings = {}
    user_retrieval = retrieval(users, ratings, model)
    for (u, i), r in ratings.items():
        if u not in user_ratings:
            user_ratings[u] = {}
        user_ratings[u][i] = r
    ndcg = []
    for user in users:
        rank = [x[0] for x in user_retrieval[user]][:100]
        dcg = sum([ratings.get((user, item), 0) >= 3.5 / np.log(i + 2) for i, item in enumerate(rank)])
        rank = sorted(user_ratings[user], key=user_ratings[user].get, reverse=True)[:100]
        idcg = sum([ratings.get((user, item), 0) >= 3.5 / np.log(i + 2) for i, item in enumerate(rank)])
        if idcg != 0:
            ndcg.append(dcg / idcg)
    return mae, rmse, np.mean(ndcg)


def retrieval(users, ratings, model):
    user_predictions = {user: {} for user in users}
    items = set()
    for _, item in ratings:
        if item in items:
            continue
        items.add(item)
        for user in users:
            user_predictions[user][item] = model.predict(user, item)
    return {user: sorted(user_predictions[user].items(), key=lambda x: x[-1], reverse=True) for user in users}


def fusion(users, ratings, pmf, svdpp):
    pmf_retrieval = retrieval(users, ratings, pmf)
    svdpp_pred = {user: {} for user in users}
    items = set()
    for _, item in ratings:
        if item in items:
            continue
        items.add(item)
        for user in users:
            svdpp_pred[user][item] = svdpp.predict(user, item).est
    svdpp_retrieval = {user: sorted(svdpp_pred[user].items(), key=lambda x: x[-1], reverse=True) for user in users}
    pmf_max = max([x[1] for user in users for x in pmf_retrieval[user]])
    pmf_min = min([x[1] for user in users for x in pmf_retrieval[user]])
    pmf_retrieval_scaled = {user: [(x[0], (x[1] - pmf_min) / (pmf_max - pmf_min)) for x in rank]
                     for user, rank in pmf_retrieval.items()}
    svdpp_max = max([x[1] for user in users for x in svdpp_retrieval[user]])
    svdpp_min = min([x[1] for user in users for x in svdpp_retrieval[user]])
    svdpp_retrieval_scaled = {user: [(x[0], (x[1] - svdpp_min) / (svdpp_max - svdpp_min)) for x in rank]
                              for user, rank in svdpp_retrieval.items()}
    user_retrieval = {user: [] for user in users}
    for user in users:
        pmf_rank = dict(pmf_retrieval_scaled[user])
        svdpp_rank = dict(svdpp_retrieval_scaled[user])
        rank = {}
        for item, r in pmf_rank.items():
            if item in svdpp_rank:
                rank[item] = (r + svdpp_rank[item]) / 2
            else:
                rank[item] = r
        for item, r in svdpp_rank.items():
            if item not in rank:
                rank[item] = r
        user_retrieval[user] = list(sorted(rank, key=rank.get, reverse=True))
    user_ratings = {}
    for (u, i), r in ratings.items():
        if u not in user_ratings:
            user_ratings[u] = {}
        user_ratings[u][i] = r
    ndcg = []
    for user in users:
        rank = [x[0] for x in svdpp_retrieval[user]][:100]
        dcg = sum([ratings.get((user, item), 0) >= 3.5 / np.log(i + 2) for i, item in enumerate(rank)])
        rank = sorted(user_ratings[user], key=user_ratings[user].get, reverse=True)[:100]
        idcg = sum([ratings.get((user, item), 0) >= 3.5 / np.log(i + 2) for i, item in enumerate(rank)])
        if idcg != 0:
            ndcg.append(dcg / idcg)
    print('svd++ ndcg', np.mean(ndcg))
    ndcg = []
    for user in users:
        rank = user_retrieval[user][:100]
        dcg = sum([ratings.get((user, item), 0) >= 3.5 / np.log(i + 2) for i, item in enumerate(rank)])
        rank = sorted(user_ratings[user], key=user_ratings[user].get, reverse=True)[:100]
        idcg = sum([ratings.get((user, item), 0) >= 3.5 / np.log(i + 2) for i, item in enumerate(rank)])
        if idcg != 0:
            ndcg.append(dcg / idcg)
    return np.mean(ndcg)


if __name__ == '__main__':
    for movielens_version in ('ml-100k', 'ml-1m'):
        print(movielens_version)
        surprise_data = surprise.Dataset.load_builtin(movielens_version)
        df = pd.DataFrame(surprise_data.raw_ratings, columns=['userId', 'movieId', 'rating', 'timestamp'])
        df = filter_ratings(df, 0, 0)
        df = serialize_data(df)
        train, test = generalized_train_test_split(df)
        train_ratings, train_users, train_items = load_ratings_mf(train)
        test_ratings, test_users, test_items = load_ratings_mf(test)

        # svdpp = surprise.SVDpp()
        # print(surprise.model_selection.cross_validate(svdpp, surprise_data, cv=2, return_train_measures=True))
        svdpp = surprise.SVDpp()
        reader = surprise.Reader(rating_scale=(0.5, 5))
        trainset = surprise.Dataset.load_from_df(train.drop(columns=['timestamp']), reader).build_full_trainset()
        svdpp.fit(trainset)

        gmf = GaussianMF(num_iterations=30, num_features=30, lambda_u=1e2, lambda_i=1e2, exposure='Bernoulli')
        gmf.fit(train_ratings, train_users, train_items)
        print(f'GMF Train Evaluation', evaluate(train_users, train_ratings, gmf))  # Evaluate train data
        print('GMF Test Evaluation', evaluate(test_users, test_ratings, gmf))  # Evaluate test data

        pmf = PoissonMF(num_features=5, num_iterations=15, smoothness=100,
                        shape_act=1, rate_act=0.3, shape_pop=1, rate_pop=0.3, shape_pref=1, shape_attr=1)
        pmf.fit(train_ratings, train_users, train_items)
        print('PMF Train Evaluation', evaluate(train_users, train_ratings, pmf))  # Evaluate train data
        print('PMF Test Evaluation', evaluate(test_users, test_ratings, pmf))  # Evaluate test data

        # mrf = LinearByLinearMRF(train_ratings, train_users, train_items)
        # mrf.fit()
        # print(mrf.evaluate(train_ratings))
        # print(mrf.evaluate(test_ratings))

        # orf = OrdinalRandomFields(train_ratings, train_users, train_items)
        # orf.fit()
        # print(orf.evaluate(train_ratings))
        # print(orf.evaluate(test_ratings))

        print(fusion(test_users, test_ratings, pmf, svdpp))
