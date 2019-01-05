"""
solution of kaggle competion.
We save all predict data and add it to data
From all data we create feature for train and predict

model of prediction conteains of pipeline. With tested on cv. for bad cv we just
off model
"""
import warnings
from timeit import default_timer as timer
from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)

base = 0.08397673393697565

CAT_ANS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]

try:
    from kaggle.competitions import twosigmanews
    FAST =False
except Exception as error:
    print('local test')
    from all.env_for_test import FAST
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
        self.dict_vol = sorted(dict_vol, key=lambda x: x[1], reverse=True)[:10]

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

        result = pd.DataFrame()
        # result.loc[:, 'bartrend'] = market['close'] / market['open']
        # result.loc[:, 'crosSMA10Close'] = FSmaCrossing(market['close'], 10)
        # result.loc[:, 'crosSMA50Close'] = FSmaCrossing(market['close'], 50)
        # result.loc[:, 'crosSMA50open'] = FSmaCrossing(market['open'], 50)
        # result.loc[:, 'gap'] = market['close'].shift(1) / market['open']
        # result.loc[:, 'volume'] = FSmaCrossing(market['volume'], 50)
        # result.loc[:, 'urgency min'] = market[('urgency', 'min')]
        # result.loc[:, 'urgency max'] = market[('urgency', 'max')]
        # result.loc[:, 'urgency std'] = market[('urgency', 'std')]
        # result.loc[:, 'urgency sma max'] = market[
        #     ('urgency', 'max')].ffill().rolling(window=20).mean()
        # result.loc[:, 'urgency sma min'] = market[
        #     ('urgency', 'min')].ffill().rolling(window=20).mean()
        #
        # result.loc[:, 'volume p'] = market['volume'] * market['close']
        #
        # result.loc[:, 'companyCount'] = market[
        #     ('companyCount', 'max')]
        #
        # result.loc[:, 'sentimentNegative max'] = market[
        #     ('sentimentNegative', 'max')]
        #
        # result.loc[:, 'sentimentNegative min'] = market[
        #     ('sentimentNegative', 'min')]
        #
        # result.loc[:, 'sentimentNegative min'] = market[
        #     ('sentimentNegative', 'min')]
        #
        # result.loc[:, 'sentimentNegative std'] = market[
        #     ('sentimentNegative', 'std')]
        #
        # result.loc[:, 'sentimentNegative std'] = market[
        #     ('sentimentNegative', 'mean')].ffill().rolling(window=20).mean()

        result.loc[:, 'r'] = market[self.RETURN_10_NEXT]
        result.loc[:, 'ticker'] = list(self._ALLMarket_dict.keys()).index(
                ticker)
        # todo add correlations between stocks
        result = result.fillna(0)
        result.index = market[self.TIME]

        return result


def get_singl_ans(X: pd.DataFrame, n):
    index_ = X.index
    try:

        probs = nn.predict_proba(X)

    except Exception as error:
        probs = [None for _ in range(len(index_))]
    func = np.random.choice
    result = np.vstack(map(lambda x: func(CAT_ANS, n, p=x), probs))

    return result


nn = MLPClassifier(hidden_layer_sizes=(100, 100),
                   activation='tanh',
                   # keep progress between .fit(...) calls
                   warm_start=True,
                   # make only 1 iteration on each .fit(...)
                   max_iter=1)

if __name__ == '__main__':
    ENV = twosigmanews.make_env()

    market_df, news_df = ENV.get_training_data()
    data_obj = Data(market_df, news_df)


    df_cv = []
    df = []
    for ticker, tuple_ in data_obj.dict_vol:
        f = data_obj.get_features(ticker)
        df.append(f[:-250])
        df_cv.append(f[-250:])


    df = pd.concat(df, axis=0, ignore_index=True)
    df_cv = pd.concat(df_cv, axis=0, ignore_index=True)

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    df = pd.DataFrame(scaler.fit_transform(df),
                      columns=df.columns)
    df_cv = pd.DataFrame(scaler.transform(df_cv),
                         columns=df_cv.columns)

    ret = []
    for i in data_obj.dict_vol:
        r = data_obj._ALLMarket_dict[i[0]][
                ['returnsOpenNextMktres10',
                 'time']
            ].iloc[-250:]
        ret.append(r)

    returns_c = pd.concat(ret, ignore_index=True)
    bl_score = returns_c.groupby('time').sum().values
    bl_score = bl_score.mean() / bl_score.std()
    top_s = abs(returns_c).groupby('time').sum().values
    top_s = top_s.mean() / top_s.std()

    ret = []

    for i in data_obj.dict_vol:
        r = data_obj._ALLMarket_dict[i[0]][
                ['returnsOpenNextMktres10', 'time']
            ].iloc[:-250]
        ret.append(r)


    returns_ = pd.concat(ret, ignore_index=True)
    bl_score_t = returns_.groupby('time').sum().values
    bl_score_t = bl_score_t.mean() / bl_score_t.std()

    bl_score_t_top = abs(returns_).groupby('time').sum().values
    bl_score_t_top = bl_score_t_top.mean() / bl_score_t_top.std()
    #
    # y_ = (returns_['returnsOpenNextMktres10'] > 0).astype(int) * 2 - 1
    # y_ = y_ * 100
    #
    # nn.partial_fit(df,
    #                y_,
    #                CAT_ANS
    #                )

    ans = []
    for golbal_loop in range(100):
        for iteration_ in range(10):
            start = timer()

            ans += [ a for a in get_singl_ans(df, 30).T]
            print('time  gen ', start - timer())
            all_scores = []
            for a in ans:
                score_df = returns_
                score_df['res'] = returns_['returnsOpenNextMktres10'] * a / 100
                score_df = score_df.groupby('time').sum()['res'].values
                score = score_df.mean() / score_df.std()
                all_scores.append(score)

            quant = np.quantile(all_scores, q=0.8)
            print(quant,
                  np.quantile(all_scores, q=0.1),
                  ' vs ',
                  bl_score_t,
                  'and',
                  bl_score_t_top)
            print('time  score ', start - timer())
            ans = [a for i, a in enumerate(ans) if all_scores[i] > quant]
            print('time  filt ', start - timer())
            x = np.vstack([df.values for _ in range(len(ans))])
            y = np.hstack(ans)
            print('time  append ', start - timer())
            nn.partial_fit(x,
                           y,
                           CAT_ANS
                           )
            print('time  fit ', start - timer())
        pred = nn.predict(df_cv)
        ret = []

        score_df = returns_c
        score_df['res'] = score_df['returnsOpenNextMktres10'] * pred / 100
        score_df = score_df.groupby('time').sum()['res'].values
        score = score_df.mean() / score_df.std()
        print("_______________CV______________", score, ' VS ', bl_score,
              ' and ', top_s, ' tree ', base)

        pd.DataFrame(y).to_csv('best ans')

        # todo save prediction
        # todo check posobile of save model between sessions
