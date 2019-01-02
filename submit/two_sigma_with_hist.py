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


def FSmaCrossing(close: pd.Series,
                 window) -> pd.DataFrame:
    f_sma_1 = close.rolling(window=window, min_periods=window).mean()
    f_sma_2 = close.rolling(window=2 * window,
                            min_periods=2 * window).mean()

    result = (f_sma_1 - f_sma_2) / f_sma_2
    return result.to_frame()


class Data:
    """
    save all data and create feature
    """

    news_cols_agg = {
        'urgency':            ['min', 'max', 'count'],
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

        market.loc[:, self.TIME ] = pd.to_datetime(
                market[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)

        self._ALLMarket = market[market[self.TIME]> 20101010] # type: pd.DataFrame

        news.loc[:, self.TIME] = pd.to_datetime(
                news[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)
        self._ALLNews = news[news[self.TIME]> 20101010]  # type: pd.DataFrame

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
        market.loc[:, self.TIME] = pd.to_datetime(
                market[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)
        self._ALLMarket = self._ALLMarket.append(market, ignore_index=True)
        news.loc[:, self.TIME] = pd.to_datetime(
                news[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)
        self._ALLNews = self._ALLNews.append(news, ignore_index=True)

    def get_features(self, ticker):
        market = self._ALLMarket[
            self._ALLMarket[self.ASSET] == ticker]  # 0.2sec


        try:
            news = self._ALLNews[self._ALLNews['assetCodes'].str.contains(
                    "'" + ticker + "'",
                    regex=False)]

            news = news.groupby(
                    [self.TIME]
                    ).agg(self.news_cols_agg)

            market.join(news, on=[self.TIME])
        except Exception as error:
            print("there is no news for " , ticker)
        #todo spped up
        result = pd.DataFrame()
        result.loc[:, 'bartrend'] = market['close'] / market['open']
        result.loc[:, 'crosSMA10Close'] = FSmaCrossing(market['close'], 10)
        result.loc[:, 'crosSMA50Close'] = FSmaCrossing(market['close'], 50)
        result.loc[:, 'crosSMA50open'] = FSmaCrossing(market['open'], 50)
        result.loc[:, 'gap'] = market['close'].shift(1) / market['open']
        result.loc[:, 'volume'] = FSmaCrossing(market['volume'], 50)
        result.loc[:, ('urgency', 'min')] = market[('urgency', 'min')].fillna(0)
        result.loc[:, ('urgency', 'max')] = market[('urgency', 'max')].fillna(0)
        result.loc[:, ('urgency', 'std')] = market[('urgency', 'std')].fillna(0)
        result.loc[:, ('urgency sma', 'max')] = market[
            ('urgency', 'max')].ffill().mean(20)
        result.loc[:, ('urgency', 'min')] = market[
            ('urgency', 'min sma')].ffill().mean(20)

        result.loc[:, 'companyCount'] = market['companyCount', 'max'].fillna(
            0)

        #todo add sentiment feature

        # todo aggregate look on kernal https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data

        # todo create features

        return market


if __name__ == '__main__':
    ENV = twosigmanews.make_env()

    market_df, news_df = ENV.get_training_data()
    data_obj = Data(market_df, news_df)
    print(data_obj.univers_ticker)
    market = data_obj._ALLMarket
    result = []

    for ticker , df in  market.groupby(data_obj.ASSET):
        ret = df.loc[250:,'returnsOpenNextMktres10'] # type: pd.Series
        profit = ret.mean()
        sharpe = profit/ ret.std()
        result.append([ticker,sharpe])


    # for ticker in data_obj.univers_ticker:
    #     # todo  join diff tickers
    #     if ticker == 'ARRO.O':
    #         data_obj.get_features(ticker).to_csv(ticker+'.csv')
