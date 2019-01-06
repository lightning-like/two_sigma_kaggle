"""
solution of kaggle competion.
We save all predict data and add it to data
From all data we create feature for train and predict

model of prediction conteains of pipeline. With tested on cv. for bad cv we just
off model
"""
import time
import warnings
from typing import List

import lightgbm as lgb
import pandas as pd
from sklearn import preprocessing

params = {"objective":        "binary",
          "metric":           "binary_logloss",
          "num_leaves":       60,
          "max_depth":        -1,
          "learning_rate":    0.01,
          "bagging_fraction": 0.9,  # subsample
          "feature_fraction": 0.9,  # colsample_bytree
          "bagging_freq":     5,  # subsample_freq
          "bagging_seed":     2018,
          "verbosity":        -1
          }

warnings.filterwarnings("ignore", category=DeprecationWarning)

CAT_ANS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]

try:
    from kaggle.competitions import twosigmanews

    FAST = False
except Exception as error:
    print('local test')
    from all import env_for_test as twosigmanews
    from all.env_for_test import FAST


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
    dict_vol = {}
    news_cols_agg = {
        'urgency':           ['min', 'max', 'std'],
        'companyCount':      ['min', 'max', 'mean', 'std'],
        'sentimentNegative': ['min', 'max', 'mean', 'std'],
        'sentimentNeutral':  ['min', 'max', 'mean', 'std'],
        'sentimentPositive': ['min', 'max', 'mean', 'std'],
        }

    # columns
    RETURN_10_NEXT = 'returnsOpenNextMktres10'
    TIME = 'time'
    ASSET = 'assetCode'
    UNIVERSE = 'universe'
    GROUP_COLS = [TIME, RETURN_10_NEXT, ASSET, UNIVERSE, 'assetName']

    NEWS_COL = [TIME, ASSET] + list(news_cols_agg)

    def TRAIN_COL(self, columns: List):
        self.TRAIN_COLS = [col for col in columns
                           if col not in self.GROUP_COLS]

    def __init__(self,
                 market: pd.DataFrame,
                 news: pd.DataFrame):
        """
        define start data and base columns
        """
        start = time.time()
        self.data_train = self.prepare_data(market, news)
        print(time.time() - start, 'prep')
        self.data_train.sort_values(by=[self.ASSET, self.TIME], inplace=True)
        self.data_predict = self.data_train[
            self.data_train[self.TIME] > 20160901]
        self.code_asset = {i: asset
                           for i, asset in
                           enumerate(self.data_train[self.ASSET].unique())}

        self.asset_code = {asset: i
                           for i, asset in
                           enumerate(self.data_train[self.ASSET].unique())}

    def prepare_data(self,
                     market: pd.DataFrame,
                     news: pd.DataFrame) -> pd.DataFrame:

        market.loc[:, self.TIME] = pd.to_datetime(
                market[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)

        assets = set(market[self.ASSET].unique())
        news.loc[:, self.TIME] = pd.to_datetime(
                news[self.TIME]
                ).dt.strftime("%Y%m%d").astype(int)



        news.loc[:, self.ASSET] = news[
            'assetCodes'].map(lambda x: next(iter((eval(x) & assets)), None))

        news = news[self.NEWS_COL]

        news = news.groupby(  # type: pd.DataFrame
                [self.TIME, self.ASSET]
                ).agg(self.news_cols_agg)

        data = market.join(news,
                           how='left',
                           on=[self.TIME, self.ASSET])

        return data

        # todo calc crosscorr and add to feature

    def add_data(self,
                 market: pd.DataFrame,
                 news: pd.DataFrame):
        """
        save new data in atr. !! there are no check be carefully
        for predict data
        """
        self.data_predict.append(self.prepare_data(market, news))
        self.data_predict.sort_values(by=[self.ASSET, self.TIME], inplace=True)

    def get_features(self, predict=False) -> pd.Series:

        if predict:
            market = self.data_predict
            date_ = market[self.TIME].iloc[-1]
        else:
            market = self.data_train
            date_ = 20100101

        result = pd.DataFrame()

        result.loc[:, 'open std'] = market['open'].rolling(window=20).std()
        result.loc[:, 'open cross'] = FSmaCrossing(result.loc[:, 'open std'],
                                                   10)
        result.loc[:, 'bartrend'] = market['close'] / market['open']
        result.loc[:, 'crosSMA10Close'] = FSmaCrossing(market['close'], 10)
        result.loc[:, 'crosSMA20Close'] = FSmaCrossing(market['close'], 20)
        result.loc[:, 'crosSMA20open'] = FSmaCrossing(market['open'], 20)
        result.loc[:, 'gap'] = market['close'].shift(1) / market['open']
        result.loc[:, 'volume'] = FSmaCrossing(market['volume'], 50)
        result.loc[:, 'urgency min'] = market[('urgency', 'min')]
        result.loc[:, 'urgency max'] = market[('urgency', 'max')]
        result.loc[:, 'urgency std'] = market[('urgency', 'std')]
        result.loc[:, 'urgency sma max'] = market[
            ('urgency', 'max')].ffill().rolling(window=20).mean()
        result.loc[:, 'urgency sma min'] = market[
            ('urgency', 'min')].ffill().rolling(window=20).mean()

        result.loc[:, 'volume p'] = market['volume'] * market['close']

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

        result.loc[:, 'universe'] = market['universe']

        result.loc[:, 'sentimentNegative std'] = market[
            ('sentimentNegative', 'mean')].ffill().rolling(window=20).mean()

        result.loc[:, 'r'] = market[self.RETURN_10_NEXT]
        result.loc[:, self.ASSET] = market[self.ASSET].map(
                lambda x: self.asset_code[x])
        # todo add correlations between stocks
        result = result.fillna(0)
        result.loc[:, self.TIME] = market[self.TIME]

        return result[result[self.TIME] >= date_]


def mark_ans(r):
    y_ = (r > 0).astype(int) * 2 - 1
    y_ = y_
    r_a = abs(r)
    y_[r_a > r_a.quantile()] /= 2
    r_a[r_a > r_a.quantile()] /= 2
    y_[r_a > r_a.quantile()] /= 2

    # calc category ans for best sharpe
    # todo add outlayre detection
    # todo add gausClust for diff category

    return y_


class Model():

    def __init__(self):
        start = time.time()
        self.ENV = twosigmanews.make_env()
        print(time.time() - start, " init")
        market_df, news_df = self.ENV.get_training_data()
        print(time.time() - start, " load")
        self.data_obj = Data(market_df, news_df)
        print(time.time() - start, " save")
        self.last_date = market_df[self.data_obj.TIME].iloc[-1]

    def train(self, last_date):

        # get train data
        f = self.data_obj.get_features()
        self.f = f
        date_ = last_date - 10000
        df = f[f[self.data_obj.TIME] < date_]
        cv_ = f[(f[self.data_obj.TIME] >= date_) &
                (f[self.data_obj.TIME] <= last_date)]
        self.TRAIN_COL = [c for c in df.columns
                          if c not in [self.data_obj.TIME, 'r' , 'universe']]

        r10t = df['r']
        r10cv = cv_['r']
        y_ = mark_ans(r10t)
        y_c = mark_ans(r10cv)

        cv_ = cv_[self.TRAIN_COL]
        df = df[self.TRAIN_COL]

        self.scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        df = pd.DataFrame(self.scaler.fit_transform(df),
                          columns=df.columns)
        cv_ = pd.DataFrame(self.scaler.transform(cv_),
                           columns=cv_.columns)

        lgtrain, lgval = lgb.Dataset(df, y_), lgb.Dataset(cv_, y_c)
        self.lgbmodel = lgb.train(params, lgtrain, 2000,
                                  valid_sets=[lgtrain, lgval],
                                  early_stopping_rounds=100, verbose_eval=200)

        # todo repeat with diff ans get category seporatly  out std  move to other category

    def test(self, start_date):
        data = self.f[self.f[self.data_obj.TIME] > start_date]
        r = data[['r', self.data_obj.TIME , 'universe']]

        data = data[self.TRAIN_COL]

        preds = self.lgbmodel.predict(
                data,
                num_iteration=self.lgbmodel.best_iteration)
        r['res'] = r['r'] * preds
        r = r[r['universe'] == 1]
        result = r.groupby(self.data_obj.TIME).sum()
        r = result['r']
        result = result['res']

        print(result.mean() / result.std() , '___vs___' , r.mean()/r.std() )

    def predict(self):
        print("generating predictions...")
        preddays = self.ENV.get_prediction_days()
        for marketdf, newsdf, predtemplatedf in preddays:
            start = time.time()
            self.data_obj.add_data(marketdf, newsdf)

            print('time add', start - time.time())

            f = self.data_obj.get_features(True)

            result = self.scaler.transform(f)
            print('time feature', start - time.time())

            preds = self.lgbmodel.predict(
                    result,
                    num_iteration=self.lgbmodel.best_iteration)

            f['ans'] = preds
            f[self.data_obj.ASSET].map(lambda x: self.data_obj.code_asset[x])

            predtemplatedf = predtemplatedf.merge(
                    f[['ans', self.data_obj.ASSET]],
                    how='left').drop(
                    'confidenceValue', axis=1).fillna(0).rename(
                    columns={'ans': 'confidenceValue'})

            self.ENV.predict(predtemplatedf)

        self.ENV.write_submission_file()


if __name__ == '__main__':

    model = Model()
    model.train(model.last_date)
    #model.test(model.last_date - 10000)
    model.predict()
