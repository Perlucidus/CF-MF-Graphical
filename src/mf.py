from abc import ABC, abstractmethod


class MFModel(ABC):
    def __init__(self, **kwargs):
        """
        :param kwargs:
        num_iterations - Number of iterations to perform
        num_features - Number of latent variables
        """
        self.num_iterations = int(kwargs.get('num_iterations', 10))
        self.num_features = int(kwargs.get('num_features', 30))

    @abstractmethod
    def fit(self, ratings, users, items):
        """
        Fits the model
        :param ratings: Ratings sparse coordinate matrix
        :param users: Dictionary of user, list of items the user has been exposed to
        :param items: Dictionary of item, list of users that have been exposed to it
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, user, item):
        """
        Predicts the user rating for the item
        :param user: user id
        :param item: item id
        :return: Rating of the user for the item
        """
        raise NotImplementedError()
