from mf import MFModel
import numpy as np
from tqdm import tqdm


class GaussianMF(MFModel):
    def __init__(self, **kwargs):
        """
        :param kwargs:
        exposure - the exposure model
        lambda_u - Inverse standard deviation for user preferences
        lambda_i - Inverse standard deviation for item attributes
        """
        super().__init__(**kwargs)
        self.lambda_u = float(kwargs.get('lambda_u', 1))
        self.lambda_i = float(kwargs.get('lambda_i', 1))
        self.exposure = kwargs.get('exposure', 'Bernoulli')
        self.user_preferences = None
        self.item_attributes = None
        self.default_user_preferences = None
        self.default_item_preferences = None

    def inverse_probability_weighting(self, users, items):
        """
        Computes the propensity scores for users to be exposed to an item
        :return: Propensity scores
        """
        if self.exposure == 'Bernoulli':
            return {(u, i): len(items[i]) / len(users) for u in users for i in items}
        elif self.exposure == 'Uniform':
            return {(u, i): 1 / (len(users) * len(items)) for u in users for i in items}
        else:
            raise NotImplementedError(f'Unsupported exposure model {self.exposure}')

    def loss(self, ratings, users, items, propensity):
        return sum((ratings[u, i] - self.predict(u, i)) ** 2 / propensity[u, i]
                   for (u, i), r in ratings.items()) \
               + self.lambda_u * sum(np.linalg.norm(self.user_preferences[u]) ** 2 for u in users) \
               + self.lambda_i * sum(np.linalg.norm(self.item_attributes[i]) ** 2 for i in items)

    def gaussian_matrix_factorization(self, ratings, users, items, propensity):
        """
        Performs matrix factorization to find user preferences and item attributes
        :return: User and item intrinsic scores (user preferences and item attributes)
        """
        # Randomly initialize intrinsic scores
        theta_u = {u: np.random.rand(self.num_features) for u in users}
        theta_i = {i: np.random.rand(self.num_features) for i in items}
        # Pre-compute outer product
        theta_i_outer = {i: np.outer(theta_i[i], theta_i[i]) for i in items}
        with tqdm(total=self.num_iterations) as progress:
            for t in range(self.num_iterations):
                progress.set_description(f'GMF Iteration {t + 1}/{self.num_iterations}')
                for u in users:
                    x = sum(theta_i_outer[i] / propensity[u, i] for i in users[u]) \
                        + self.lambda_i * np.eye(self.num_features)
                    theta_u[u] = np.linalg.inv(x) @ sum(ratings[u, i] * theta_i[i] / propensity[u, i] for i in users[u])
                # Pre-compute outer product
                theta_u_outer = {u: np.outer(theta_u[u], theta_u[u]) for u in users}
                for i in items:
                    x = sum(theta_u_outer[u] / propensity[u, i] for u in items[i]) \
                        + self.lambda_u * np.eye(self.num_features)
                    theta_i[i] = np.linalg.inv(x) @ sum(ratings[u, i] * theta_u[u] / propensity[u, i] for u in items[i])
                # Pre-compute outer product
                theta_i_outer = {i: np.outer(theta_i[i], theta_i[i]) for i in items}
                self.user_preferences, self.item_attributes = theta_u, theta_i  # Update intrinsic scores
                progress.set_postfix_str(f'Loss: {self.loss(ratings, users, items, propensity)}')
                progress.update()
        return theta_u, theta_i

    def fit(self, ratings, users, items):
        propensity = self.inverse_probability_weighting(users, items)
        self.user_preferences, self.item_attributes = \
            self.gaussian_matrix_factorization(ratings, users, items, propensity)
        self.default_user_preferences = np.mean(list(self.user_preferences.values()), axis=0)
        self.default_item_preferences = np.mean(list(self.item_attributes.values()), axis=0)

    def predict(self, user, item):
        """
        Predicts the user rating for the item
        :param user: user id
        :param item: item id
        :return: Rating of the user for the item
        """
        user_pref = self.user_preferences.get(user, self.default_user_preferences)
        item_attr = self.item_attributes.get(item, self.default_item_preferences)
        return user_pref @ item_attr
