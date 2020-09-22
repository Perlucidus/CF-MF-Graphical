import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rating_plot(ratings, title):
    """
    Plot ratings histogram
    :param ratings: Ratings dataframe
    :param title: Graph title
    """
    rating_count = dict()
    for idx, row in ratings.iterrows():
        rating = row['rating']
        rating_count[rating] = rating_count.get(rating, 0) + 1
    rating_count = list(rating_count.items())
    rating_count.sort(key=lambda k: k[0])
    rating, counts = list(zip(*rating_count))
    # plt.xlabel('Rating')
    # plt.ylabel('Count')
    plt.xticks(0.5 + np.arange(10) / 2)
    # plt.title(title)
    plt.bar(rating, counts)
    plt.show()


if __name__ == '__main__':
    for ml in ('ml-100k', 'ml-1m'):
        rating_plot(pd.read_csv(f'../data/{ml}/ratings.csv'), ml)
