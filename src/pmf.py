from mf import MFModel
import numpy as np
from scipy import special
from tqdm import trange


class PoissonMF(MFModel):
    def __init__(self, **kwargs):
        """
        :param kwargs:
        smoothness - Smoothness for random initialization
        shape_act - Shape of user activity
        rate_act - Rate of user activity
        shape_pop - Shape of item popularity
        rate_pop - Rate of item popularity
        shape_pref - Shape of user preferences
        shape_attr - Shape of item attributes
        """
        super().__init__(**kwargs)
        self.smoothness = int(kwargs.get('smoothness', 100))
        self.user_activity_base_shape = float(kwargs.get('shape_act', 0.3))
        self.user_activity_base_rate = float(kwargs.get('rate_act', 1))
        self.item_popularity_base_shape = float(kwargs.get('shape_pop', 0.3))
        self.item_popularity_base_rate = float(kwargs.get('rate_pop', 1))
        self.user_preference_base_shape = float(kwargs.get('shape_pref', 0.3))
        self.item_attribute_base_shape = float(kwargs.get('shape_attr', 0.3))
        self.user_preferences_shape = None
        self.user_preferences_rate = None
        self.item_attributes_shape = None
        self.item_attributes_rate = None
        self.default_user_preferences_shape = None
        self.default_user_preferences_rate = None
        self.default_item_attributes_shape = None
        self.default_item_attributes_rate = None

    def fit(self, ratings, users, items):
        # Randomly initialize shapes and rates
        def param_init(n):
            return self.smoothness / np.random.gamma(self.smoothness, 1 / self.smoothness, n)

        pref_shp = {u: param_init(self.num_features) for u in users}  # User preference shape
        pref_rte = {u: param_init(self.num_features) for u in users}  # User preference rate
        act_rte = {u: param_init(1) for u in users}  # User activity rate
        attr_shp = {i: param_init(self.num_features) for i in items}  # Item attribute shape
        attr_rte = {i: param_init(self.num_features) for i in items}  # Item attribute rate
        pop_rte = {i: param_init(1) for i in items}  # Item popularity rate
        # Compute initial user activity shape and item popularity shape
        act_shp = self.user_activity_base_shape + self.user_preference_base_shape * self.num_features
        pop_shp = self.item_popularity_base_shape + self.item_attribute_base_shape * self.num_features
        for _ in trange(self.num_iterations, desc='PMF'):
            # Update the multinomial
            phi = {(u, i): np.exp(
                special.digamma(pref_shp[u]) - np.log(pref_rte[u]) +
                special.digamma(attr_shp[i]) - np.log(attr_rte[i])
            ) for u, i in ratings}
            phi = {ui: z / sum(z) for ui, z in phi.items()}
            for u in users:  # Update user preferences shape, rate and user activity rate
                pref_shp[u] = self.user_preference_base_shape + sum(ratings[u, i] * phi[u, i] for i in users[u])
                pref_rte[u] = (act_shp / act_rte[u]) + sum(attr_shp[i] / attr_rte[i] for i in users[u])
                act_rte[u] = (self.user_activity_base_shape / self.user_activity_base_rate)
                act_rte[u] += sum(pref_shp[u] / pref_rte[u])
            for i in items:  # Update item attributes shape, rate and item popularity rate
                attr_shp[i] = self.item_attribute_base_shape + sum(ratings[u, i] * phi[u, i] for u in items[i])
                attr_rte[i] = (pop_shp / pop_rte[i]) + sum(pref_shp[u] / pref_rte[u] for u in items[i])
                pop_rte[i] = self.item_popularity_base_shape / self.item_popularity_base_rate
                pop_rte[i] += sum(attr_shp[i] / attr_rte[i])
        self.user_preferences_shape = pref_shp
        self.user_preferences_rate = pref_rte
        self.item_attributes_shape = attr_shp
        self.item_attributes_rate = attr_rte
        self.default_user_preferences_shape = np.mean(list(self.user_preferences_shape.values()), axis=0)
        self.default_user_preferences_rate = np.mean(list(self.user_preferences_rate.values()), axis=0)
        self.default_item_attributes_shape = np.mean(list(self.item_attributes_shape.values()), axis=0)
        self.default_item_attributes_rate = np.mean(list(self.item_attributes_rate.values()), axis=0)

    def predict(self, user, item):
        """
        Predicts the user rating for the item
        :param user: user id
        :param item: item id
        :return: Rating of the user for the item
        """
        # Log gamma expectation
        user_pref_shp = self.user_preferences_shape.get(user, self.default_user_preferences_shape)
        user_pref_rte = self.user_preferences_rate.get(user, self.default_user_preferences_rate)
        item_attr_shp = self.item_attributes_shape.get(item, self.default_item_attributes_shape)
        item_attr_rte = self.item_attributes_rate.get(item, self.default_item_attributes_rate)
        rating = sum(
            np.exp(special.digamma(user_pref_shp) - np.log(user_pref_rte) +
                   special.digamma(item_attr_shp) - np.log(item_attr_rte))
        )
        return rating
