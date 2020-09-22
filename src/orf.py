import numpy as np
import scipy.optimize as optimize
import scipy.special as special
import scipy.stats as stats
from itertools import combinations
from pathlib import Path
import pickle


class OrdinalRandomFields:
    def __init__(self, ratings, users, items, **kwargs):
        self.ratings = ratings
        self.users = users
        self.items = items
        self.user_size = max(self.users) + 1
        self.item_size = max(self.items) + 1
        self.num_iter = kwargs.get('num_iter', 1)
        self.latent_size = kwargs.get('latent_size', 50)
        self.reg = kwargs.get('reg', 0.1)
        self.mean_rating = np.mean(list(self.ratings.values()))
        self.user_means = {u: np.mean([self.ratings[u, i] for i in self.users[u]]) for u in self.users}
        self.item_means = {i: np.mean([self.ratings[u, i] for u in self.items[i]]) for i in self.items}
        self.bias = None
        self.user_logistic_latent = None
        self.item_logistic_latent = None
        self.user_features = None
        self.item_features = None
        self.user_interaction_features = None
        self.item_interaction_features = None
        self.weights = None

    def fit(self):
        f = Path('orf.txt')
        if f.is_file():
            with f.open('rb') as p:
                self.bias, self.user_logistic_latent, self.item_logistic_latent, self.user_features, \
                self.item_features, self.user_interaction_features, self.item_interaction_features = pickle.load(p)
        else:
            print('Fit ordinal')
            self.fit_ordinal()
            print('Pearson features')
            self.pearson_features()
            with f.open('wb') as p:
                pickle.dump((self.bias, self.user_logistic_latent, self.item_logistic_latent, self.user_features,
                             self.item_features, self.user_interaction_features, self.item_interaction_features), p)
        self.weights = {u: {(i, j): np.random.normal(0, 0.01) for i, j in f}
                        for u, f in self.item_interaction_features.items()}
        eta = 5
        for t in range(self.num_iter):
            print(f'Iteration {t + 1}')
            for user in self.users:
                for i, j in self.weights[user]:
                    self.weights[user][i, j] += eta * self.gradient(user, i, j)

    def gradient(self, user, i, j):
        g = self.item_interaction_features[user][i, j]
        for rating in np.arange(0, 5, 0.5) + 0.5:
            g -= self.local_likelihood(user, i, rating) * self.item_interaction_features[user][i, j]
        return g

    def normalization(self, user, item):
        s = 0
        for rating in np.arange(0, 5, 0.5) + 0.5:
            q = self.ordinal(user, item, rating)
            x = np.exp(np.sum(self.weights[user][i, j] * self.item_interaction_features[user][i, j]
                              for i, j in self.item_interaction_features[user] if i == item or j == item))
            s += q * x
        return s

    def local_likelihood(self, user, item, rating):
        q = self.ordinal(user, item, rating)
        x = np.exp(np.sum(self.weights[user][i, j] * self.item_interaction_features[user][i, j]
                          for i, j in self.item_interaction_features[user] if i == item or j == item))
        return q * x / self.normalization(user, item)

    def predict(self, user, item):
        best = 0
        best_p = 0
        try:
            for rating in np.arange(0, 5, 0.5) + 0.5:
                p = self.local_likelihood(user, item, rating)
                if p > best_p:
                    best = rating
                    best_p = p
        except IOError:
            if user in self.users:
                return self.user_means[user]
            if item in self.items:
                return self.item_means[item]
            return self.mean_rating
        return best

    def evaluate(self, ratings):
        return {
            'RMSE': np.sqrt(sum(np.square(self.predict(u, m) - r) for (u, m), r in ratings.items()) / len(ratings)),
            'MAE': sum(np.abs(self.predict(u, m) - r) for (u, m), r in ratings.items()) / len(ratings)
        }

    def ordinal(self, user, item, rating):
        mu = self.bias[user, item] + self.user_logistic_latent[user] @ self.item_logistic_latent[item]
        user_var = np.var([self.ratings[user, i] for i in self.users[user]])
        item_var = np.var([self.ratings[u, item] for u in self.items[item]])
        std = np.sqrt(user_var + item_var)
        r_ceil = np.ceil(rating * 2) / 2
        r_floor = np.floor(rating * 2) / 2
        return special.expit((r_ceil - mu) / std) - special.expit((r_floor - mu) / std)

    def ordinal_loss(self, weights):
        weights = weights.reshape(-1, self.latent_size)
        p = weights[:self.user_size]
        q = weights[self.user_size:]
        loss = np.sum(np.square(self.ratings[u, i] - self.bias[u, i] - p[u] @ q[i])
                      + self.reg * (np.square(p[u]) + np.square(q[i])) for u, i in self.ratings)
        p_grad = 2 * self.reg * p
        q_grad = 2 * self.reg * q
        for u in self.users:
            p_grad[u] += 2 * np.sum((self.ratings[u, i] - self.bias[u, i] - p[u] @ q[i]) * -q[i] for i in self.users[u])
        for i in self.items:
            q_grad[i] += 2 * np.sum((self.ratings[u, i] - self.bias[u, i] - p[u] @ q[i]) * -p[u] for u in self.items[i])
        return loss, np.concatenate([p_grad, q_grad]).reshape(-1)

    def fit_ordinal(self):
        user_bias = {u: self.user_means[u] - self.mean_rating for u in self.users}
        item_bias = {i: self.item_means[i] - self.mean_rating for i in self.items}
        self.bias = {(u, i): self.mean_rating + user_bias[u] + item_bias[i] for u, i in self.ratings}
        p = np.random.uniform(size=(self.user_size, self.latent_size))
        q = np.random.uniform(size=(self.item_size, self.latent_size))
        weights = np.concatenate([p, q])
        weights, _, _ = optimize.fmin_l_bfgs_b(self.ordinal_loss, weights, maxiter=50)
        weights = weights.reshape(-1, self.latent_size)
        self.user_logistic_latent = weights[:self.user_size]
        self.item_logistic_latent = weights[self.user_size:]

    def pearson_features(self):
        self.user_interaction_features = {}
        self.item_interaction_features = {}
        g = lambda x: special.expit(np.abs(x))
        self.user_features = {u: {i: g(self.ratings[u, i] - self.user_means[u]) for i in self.users[u]}
                              for u in self.users}
        self.item_features = {i: {u: g(self.ratings[u, i] - self.item_means[i]) for u in self.items[i]}
                              for i in self.items}
        # for i in self.items:
        #     self.user_interaction_features[i] = {(u, v): g((self.ratings[u, i] - self.user_means[u])
        #                                                    - (self.ratings[v, i] - self.user_means[v]))
        #                                          for u, v in combinations(self.users[i], 2)}
        for u in self.users:
            neighbors = {(i, j): set(self.items[i]).intersection(self.items[j])
                         for i, j in combinations(self.users[u], 2)}
            pearson = {(i, j): stats.pearsonr([self.ratings[v, i] for v in neighbors[i, j]],
                                              [self.ratings[v, j] for v in neighbors[i, j]])
                       for i, j in combinations(self.users[u], 2)}
            best = sorted(pearson, key=pearson.get, reverse=True)[:10]
            self.item_interaction_features[u] = {(i, j): g((self.ratings[u, i] - self.item_means[i])
                                                           - (self.ratings[u, j] - self.item_means[j]))
                                                 for i, j in best}
