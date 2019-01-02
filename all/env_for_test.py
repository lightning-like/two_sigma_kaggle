"""
env for local test use file from example before pull data
"""

import os
import pickle
from typing import Tuple

import pandas as pd


# todo try download all data from submit
class Env:
    """
    emulate kernal
    """

    module_path = os.path.dirname(__file__)
    input_market_data = module_path + '/input_market_data'
    input_news_data_1 = module_path + '/input_news_data_1'
    input_news_data_2 = module_path + '/input_news_data_2'
    out_data = module_path + '/out_data'

    result = []

    def __init__(self):
        with open(self.input_market_data, 'rb') as f:
            self.market = pickle.load(f)  # type: pd.DataFrame

        with open(self.input_news_data_1, 'rb') as f:
            f1 = pickle.load(f)  # type: pd.DataFrame
        with open(self.input_news_data_2, 'rb') as f:
            f2 = pickle.load(f)  # type: pd.DataFrame

        self.news = pd.concat([f1, f2])

        with open(self.out_data, 'rb') as f:
            self.out = pickle.load(
                    f)  # type: Tuple[pd.DataFrame, pd.DataFrame , DataFrame ]

    def get_prediction_days(self):
        return self.out

    def get_training_data(self):
        return self.market.copy(), self.news.copy()

    def predict(self, predict):
        self.result.append(predict)

    def write_submission_file(self):
        print(self.result)


def make_env():
    return Env()


def create_data():
    with open('news_df.dms', 'rb') as f:
        news = pd.read_csv(f)
    with open('market_df.dms', 'rb') as f:
        market = pd.read_csv(f)

    with open('input_market_data', 'wb') as f:
        pickle.dump(market, f)

    with open('input_news_data_1', 'wb') as f:
        pickle.dump(news.loc[:int(len(news) / 2)], f)

    with open('input_news_data_2', 'wb') as f:
        pickle.dump(news.loc[int(len(news) / 2):], f)

    tmplates = ['market.dms', 'news.dms', 'result.dms']

    predictdate = []
    for i in range(20):
        date_ = []
        for tmp in tmplates:
            with open(str(i) + tmp, 'rb') as f:
                date_.append(pd.read_csv(f))
        predictdate.append(date_)

    with open('out_data', 'wb') as f:
        pickle.dump(predictdate, f)


def speed_test():
    from timeit import default_timer as timer

    start = timer()
    with open('news_df.dms', 'rb') as f:
        news = pd.read_csv(f)
    print(len(news))
    download_market = timer()
    print(download_market - start, ' seconds for market csv')
    with open('market_df.dms', 'rb') as f:
        market = pd.read_csv(f)
    print(len(market))
    download_cvs = timer()
    print(download_cvs - start, ' seconds for all csv')

    with open('input_market_data', 'rb') as f:
        market = pickle.load(f)  # type: pd.DataFrame
    download_pkl_market = timer()
    print(download_pkl_market - download_cvs, ' seconds for pkl market')
    with open('input_news_data_1', 'rb') as f:
        f1 = pickle.load(f)  # type: pd.DataFrame
    download_pkl_news_1 = timer()
    print(download_pkl_market - download_pkl_news_1,
          ' seconds for pkl news 1')
    with open('input_news_data_2', 'rb') as f:
        f2 = pickle.load(f)  # type: pd.DataFrame
    download_pkl_news_2 = timer()
    print(download_pkl_news_2 - download_pkl_news_1,
          ' seconds for pkl news 2')

    news = f1.append(f2,ignore_index=True)
    download_concat = timer()
    print(download_pkl_news_2 - download_concat, ' seconds for all news')


if __name__ == '__main__':

    speed_test()
