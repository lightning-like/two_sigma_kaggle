"""
solution of kaggle competion.
We save all predict data and add it to data
From all data we create feature for train and predict

model of prediction conteains of pipeline. With tested on cv. for bad cv we just
off model
"""
from typing import List

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from all.env_for_test import FAST

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
        'urgency':            ['min', 'max', 'std'],
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

        market.loc[:, self.TIME] = pd.to_datetime(
                market[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)

        self._ALLMarket = market[
            market[self.TIME] > 20101010]  # type: pd.DataFrame

        news.loc[:, self.TIME] = pd.to_datetime(
                news[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)

        self._ALLNews = news[news[self.TIME] > 20101010]  # type: pd.DataFrame

        self._split_data()

    def _split_data(self):
        self._ALLMarket_dict = {ticker: df
                                for ticker, df in self._ALLMarket.groupby(
                self.ASSET)}

        self._ALLNews_dict = {ticker: df
                              for tickers, df in self._ALLNews.groupby(
                'assetCodes') for ticker in eval(tickers)}

        dict_vol = {}
        for ticker, data in self._ALLMarket_dict.items():
            vol = data.iloc[-250:]['close'] * data.iloc[-250:]['volume']
            un = data.iloc[-250:][self.UNIVERSE]

            if FAST or (un.sum() == 250 and len(data) >= 1568):
                dict_vol[ticker] = (vol.mean())

        dict_vol = dict_vol.items()
        self.dict_vol = sorted(dict_vol, key=lambda x: x[1], reverse=True)

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

        self._split_data()



    def get_features(self, ticker):

        try:
            market = self._ALLMarket_dict[ticker]
            news = self._ALLNews_dict[ticker]
        except KeyError as error:
            return pd.DataFrame()

        news = news.groupby(
                [self.TIME]
                ).agg(self.news_cols_agg)

        market = market.join(news, on=[self.TIME])


        result = pd.DataFrame(index=market[self.TIME])
        result.loc[:, 'bartrend'] = market['close'] / market['open']
        result.loc[:, 'crosSMA10Close'] = FSmaCrossing(market['close'], 10)
        result.loc[:, 'crosSMA50Close'] = FSmaCrossing(market['close'], 50)
        result.loc[:, 'crosSMA50open'] = FSmaCrossing(market['open'], 50)
        result.loc[:, 'gap'] = market['close'].shift(1) / market['open']
        result.loc[:, 'volume'] = FSmaCrossing(market['volume'], 50)
        result.loc[:, 'urgency min'] = market[('urgency', 'min')]
        result.loc[:, 'urgency max'] = market[('urgency', 'max')]
        result.loc[:, 'urgency std'] = market[('urgency', 'std')]
        result.loc[:, 'urgency sma max'] = market[
            ('urgency', 'max')].ffill().rolling(window=20).mean()
        result.loc[:, 'urgency sma min'] = market[
            ('urgency', 'min')].ffill().rolling(window=20).mean()

        result.loc[:, 'companyCount'] = market[
            ('companyCount', 'max')]

        result.loc[:, 'sentimentNegative max'] = market[
            ('sentimentNegative', 'max')]

        result.loc[:, 'sentimentNegative min'] = market[
            ('sentimentNegative', 'min')]

        result.loc[:, 'sentimentNegative min'] = market[
            ('sentimentNegative', 'min')]

        result.loc[:, 'sentimentNegative std'] = market[
            ('sentimentNegative', 'std')]

        result.loc[:, 'sentimentNegative std'] = market[
            ('sentimentNegative', 'mean')].ffill().rolling(window=20).mean()

        #todo add correlations between stocks
        result = result.fillna(0)



        return result




def get_singl_ans(X: pd.DataFrame, y: list):
    index_ = X.index
    try:
        probs = nn.predict_proba(X)
    except Exception as error:
        probs = [None for i in range(len(index_))]
    result = pd.Series(
            [np.random.choice(list(range(len(y))), p=prob)
             for prob in probs],
            index=index_,
            name='ans _col')
    return result





nn = MLPClassifier(hidden_layer_sizes=(100, 100),
                   activation='tanh',
                   # keep progress between .fit(...) calls
                   warm_start=True,
                   # make only 1 iteration on each .fit(...)
                   max_iter=20)

if __name__ == '__main__':
    ENV = twosigmanews.make_env()

    market_df, news_df = ENV.get_training_data()
    data_obj = Data(market_df, news_df)
    market = data_obj._ALLMarket
    result = []
    from timeit import default_timer as timer

    df = []
    for ticker, tuple_ in list(data_obj.dict_vol):
        df.append(data_obj.get_features(ticker))
        download_market = timer()

    df = pd.concat(df, axis=1)

    a  = get_singl_ans(df, data_obj.dict_vol)

    ret = []
    for i in range(len(data_obj.dict_vol)):
        r = data_obj._ALLMarket_dict[data_obj.dict_vol[i][0]]['returnsOpenNextMktres10']
        r.index  = a.index[-len(r):]
        ret.append( r[a == i])


    print(pd.concat(ret))
    # for i in range(1):
    #     start = timer()
    #     ans = [get_singl_ans(df, data_obj.dict_vol) for _ in range(50)]
    #
    #
    #
    #     #todo calc reword
    #     #todo filter best samples
    #     for a in ans:
    #         download_market = timer()
    #         print(download_market - start, ' seconds fit ')
    #         nn.partial_fit(df, a,  list(range(len(data_obj.dict_vol))) ) #todo add sell *2 - len()
    #     #todo add cross validation
    #     download_market = timer()
    #     print(download_market - start, ' seconds totala ')
    #     # todo in prodaction cacl reword and use propabity as y [-1,1] for small number of stock
    #     # todo check posobile of save model between sessions
