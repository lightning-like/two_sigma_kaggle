"Baseline"

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC

#todo prepare data
#todo make feature
#todo pipline


try:
    from kaggle.competitions import twosigmanews
except Exception as error:
    print('local test')
    from all import env_for_test as twosigmanews

PARAMS_SVC = {
    'C':                       1,
    'decision_function_shape': 'ovr',
    'kernel':                  'linear'
    }

RETURN_10_NEXT = 'returnsOpenNextMktres10'
TIME = 'time'
ASSET = 'assetCode'



class Data():

    ALLMarket = None
    ALLNews = None

    def prepare_data(self, market_df: pd.DataFrame, news_df: pd.DataFrame):
        print('preparing data...')
        # a bit of feature engineering
        # TODO NEED ADD MA , SHIFTS

        if self.ALLMarket is None:
            self.ALLMarket = market_df
        else:
            self.ALLMarket = pd.concat([self.ALLMarket,market_df])

        if self.ALLNews is None:
            self.ALLNews = market_df
        else:
            self.ALLNews = pd.concat([self.ALLNews, news_df])


        market_df[TIME] =pd.to_datetime(market_df[TIME]).dt.strftime("%Y%m%d").astype(int)
        market_df['bartrend'] = market_df['close'] / market_df['open']
        market_df['average'] = (market_df['close'] + market_df['open']) / 2
        market_df['pricevolume'] = market_df['volume'] * market_df['close']

        news_df[TIME] =  pd.to_datetime(news_df[TIME]).dt.strftime("%Y%m%d").astype(int)
        news_df[ASSET] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
        news_df['position'] = news_df['firstMentionSentence'] / news_df[
            'sentenceCount']
        news_df['coverage'] = news_df['sentimentWordCount'] / news_df['wordCount']

        # get rid of extra junk from news data
        droplist = ['sourceTimestamp', 'firstCreated', 'sourceId', 'headline',
                    'takeSequence', 'provider', 'firstMentionSentence',
                    'sentenceCount', 'bodySize', 'headlineTag', 'marketCommentary',
                    'subjects', 'audiences', 'sentimentClass',
                    'assetName', 'assetCodes', 'urgency', 'wordCount',
                    'sentimentWordCount']
        news_df.drop(droplist, axis=1, inplace=True)
        market_df.drop(['assetName', 'volume'], axis=1, inplace=True)

        # combine multiple news reports for same assets on same day
        news_gp = news_df.groupby([TIME, ASSET], sort=False).aggregate(
                np.mean).reset_index()

        # join news reports to market data, note many assets will have many
        # days without news data
        return pd.merge(market_df, news_gp, how='left', on=[TIME, ASSET],
                        copy=False).fillna(0)  # , right_on=['time', 'assetCodes'])


DATA = Data()

def init_data():
    ENV = twosigmanews.make_env()
    market_df, news_df = ENV.get_training_data()
    data_df = DATA.prepare_data(market_df, news_df)
    return data_df, ENV


def get_ans(all_data) -> pd.Series:
    result = sum(_answer_sma(all_data, window)
                 for window in range(10, 30 + 1, 1))  # type:  pd.Series
    result = result.apply(np.sign)

    return result


def _answer_sma(all_data: pd.DataFrame, window):
    close = all_data['close']
    f_sma = close.rolling(window=window, min_periods=window).mean()
    b_sma = f_sma.shift(-window)
    b_sma.fillna(close, inplace=True)

    indicator = f_sma - b_sma
    signal = _get_intersections(indicator, close)
    return signal


def _get_intersections(indicator, time_series):
    indicator_shift = indicator.shift(1)

    rolling = time_series.rolling(5, center=True)
    range_ = np.arange(len(time_series) - 4)
    # find the index number of rolling argmax and argmin

    roll_argmax = rolling.apply(np.argmax)[2:-2].T.astype(int) + range_
    roll_argmin = rolling.apply(np.argmin)[2:-2].T.astype(int) + range_

    # find the index of buy and sell points (where two sma intersect)
    # sell_index = result[(indicator >= 0) & (indicator_shift < 0)].index
    # buy_index = result[(indicator < 0) & (indicator_shift >= 0)].index

    # find local argmax and argmin in the buy and sell points
    sell_index = roll_argmax[(indicator >= 0) & (indicator_shift < 0)]
    buy_index = roll_argmin[(indicator < 0) & (indicator_shift >= 0)]

    result = pd.Series(len(time_series) * [0], index=time_series.index)
    result[result.index[sell_index]] = -1
    result[result.index[buy_index]] = 1

    return result.astype(int)


def train(data_df):
    d = data_df[data_df['universe'] == 1]
    tickers_df = d.groupby(ASSET)
    svms = {}
    scalers = {}
    for tiker, data_t in list(tickers_df):
        #todo add columns from other asset
        #todo add crosvalidation
        #todo on/off active
        s = SVC(**PARAMS_SVC)
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x = data_t[TRAIN_COLS]
        y = get_ans(data_t)
        x = scaler.fit_transform(x)
        try:
            s.fit(x, y)
            svms[tiker] = s
            scalers[tiker] = scaler
        except ValueError as error:
            print(error)
    return svms, scalers


def predict(svms, scalers, market_df, news_df, pred_template_df):

    data_df_predict = DATA.prepare_data(market_df, news_df)

    predicts= {}
    result = []
    for asset in pred_template_df[ASSET]:
        data_df_predict_a = data_df_predict[data_df_predict[ASSET] == asset]
        data_df_predict_a = data_df_predict_a[TRAIN_COLS]
        try:
            x = scalers[asset].transform(data_df_predict_a)
            pred = svms[asset].predict(x)[0]
            if pred != 0:
                predicts[asset] = pred
            else:
                predicts[asset] = predicts.get(asset,0) * 0.95

            result.append(predicts[asset])
        except KeyError as error:
            result.append(0)
    pred_template_df['confidenceValue'] = result
    return pred_template_df


if __name__ == '__main__':

    data_df, ENV = init_data()

    GROUP_COLS = [TIME, RETURN_10_NEXT, ASSET, 'universe', 'assetName']

    TRAIN_COLS = [col for col in data_df.columns
                  if col not in GROUP_COLS]

    svms, scalers = train(data_df)

    pred_days = ENV.get_prediction_days()
    for market_df, news_df, pred_template_df in pred_days:
        pred_template_df = predict(
                svms,
                scalers,
                market_df,
                news_df,
                pred_template_df)

        ENV.predict(pred_template_df)

    ENV.write_submission_file()
