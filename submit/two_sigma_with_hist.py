"""
solution of kaggle competion.
We save all predict data and add it to data
From all data we create feature for train and predict

model of prediction conteains of pipeline. With tested on cv. for bad cv we just
off model
"""
from typing import List

import pandas as pd

try:
    from kaggle.competitions import twosigmanews
except Exception as error:
    print('local test')
    from all import env_for_test as twosigmanews


class Data:
    """
    save all data and create feature
    """

    news_cols_agg = {
        'urgency':            ['min', 'count'],
        'takeSequence':       ['max'],
        'bodySize':           ['min', 'max', 'mean', 'std'],
        'wordCount':          ['min', 'max', 'mean', 'std'],
        'sentenceCount':      ['min', 'max', 'mean', 'std'],
        'companyCount':       ['min', 'max', 'mean', 'std'],
        'marketCommentary':   ['min', 'max', 'mean', 'std'],
        'relevance':          ['min', 'max', 'mean', 'std'],
        'sentimentNegative':  ['min', 'max', 'mean', 'std'],
        'sentimentNeutral':   ['min', 'max', 'mean', 'std'],
        'sentimentPositive':  ['min', 'max', 'mean', 'std'],
        'sentimentWordCount': ['min', 'max', 'mean', 'std'],
        'noveltyCount12H':    ['min', 'max', 'mean', 'std'],
        'noveltyCount24H':    ['min', 'max', 'mean', 'std'],
        'noveltyCount3D':     ['min', 'max', 'mean', 'std'],
        'noveltyCount5D':     ['min', 'max', 'mean', 'std'],
        'noveltyCount7D':     ['min', 'max', 'mean', 'std'],
        'volumeCounts12H':    ['min', 'max', 'mean', 'std'],
        'volumeCounts24H':    ['min', 'max', 'mean', 'std'],
        'volumeCounts3D':     ['min', 'max', 'mean', 'std'],
        'volumeCounts5D':     ['min', 'max', 'mean', 'std'],
        'volumeCounts7D':     ['min', 'max', 'mean', 'std']
        }

    # columns
    RETURN_10_NEXT = 'returnsOpenNextMktres10'
    TIME = 'time'
    ASSET = 'assetCode'
    UNIVERSE = 'universe'
    GROUP_COLS = [TIME, RETURN_10_NEXT, ASSET, UNIVERSE, 'assetName']

    def TRAIN_COL(self, columns: List):
        self.TRAIN_COLS = [col for col in columns
                           if col not in self.GROUP_COLS]

    def __init__(self,
                 market: pd.DataFrame,
                 news: pd.DataFrame):
        """
        define start data and base columns
        """

        self._ALLMarket = market  # type: pd.DataFrame
        self._ALLNews = news  # type: pd.DataFrame

        # 2sec
        self.not_univers_ticker, self.univers_ticker = [[
            ticker
            for ticker, data_df_a in m_pd.groupby(self.ASSET)]
            for u, m_pd in self._ALLMarket.groupby(self.UNIVERSE)]

    def add_data(self,
                 market: pd.DataFrame,
                 news: pd.DataFrame):
        """
        save new data in atr. !! there are no check be carefully
        """
        self._ALLMarket = self._ALLMarket.append(market, ignore_index=True)
        self._ALLNews = self._ALLNews.append(news, ignore_index=True)

    def get_features(self, ticker):
        market = self._ALLMarket[
            self._ALLMarket[self.ASSET] == ticker]  # 0.2sec
        market.loc[:, self.TIME + 'tmp'] = pd.to_datetime(
                market[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)

        news = self._ALLNews[self._ALLNews['assetCodes'].str.contains(
                "'" + ticker + "'",
                regex=False)]
        news.loc[:, self.ASSET] = ticker
        news.loc[:, self.TIME + 'tmp'] = pd.to_datetime(
                news[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)
        news = news.groupby(
                [self.TIME + 'tmp', self.ASSET]
                ).agg(self.news_cols_agg)

        # todo aggregate look on kernal https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data

        # todo create features

        return market.join(news, on=[self.TIME + 'tmp', self.ASSET])


if __name__ == '__main__':
    ENV = twosigmanews.make_env()

    market_df, news_df = ENV.get_training_data()
    data_obj = Data(market_df, news_df)

    for ticker in data_obj.univers_ticker[:1]:
        #todo  join diff tickers

        ticker_date = data_obj.get_features(ticker)
        ticker_date.to_csv(ticker + "feature.csv")
