"Baseline"

import lightgbm as lgb
import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews

print('Training lightgbm')

# money
PARAMS = {"objective":        "binary",
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

RETURN_10_NEXT = 'returnsOpenNextMktres10'
TIME = 'time'
ASSET = 'assetCode'


def prepare_data(market_df: pd.DataFrame, news_df: pd.DataFrame):
    print('preparing data...')
    # a bit of feature engineering
    # TODO NEED ADD MA , SHIFTS
    market_df[TIME] = market_df.time.dt.strftime("%Y%m%d").astype(int)
    market_df['bartrend'] = market_df['close'] / market_df['open']
    market_df['average'] = (market_df['close'] + market_df['open']) / 2
    market_df['pricevolume'] = market_df['volume'] * market_df['close']

    news_df[TIME] = news_df.time.dt.strftime("%Y%m%d").astype(int)
    news_df[ASSET] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
    news_df['position'] = news_df['firstMentionSentence'] / news_df[
        'sentenceCount']
    news_df['coverage'] = news_df['sentimentWordCount'] / news_df['wordCount']

    # filter pre-2012 data, no particular reason
    market_df = market_df.loc[market_df[TIME] > 20120000]

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


def get_ans(all_data: pd.DataFrame) -> pd.Series:
    return (all_data[RETURN_10_NEXT] > 0).astype(int)


class model():
    def __init__(self, Xt, Yt, Xv, Yv):
        # todo replase SVM for each asset seporatly
        # for svm not sure how confidence calc

        lg_train, lg_val = lgb.Dataset(Xt, Yt[:, 0]), lgb.Dataset(Xv, Yv[:, 0])

        self.lgbm = lgb.train(PARAMS,
                              lg_train,
                              2000,
                              valid_sets=[lg_train, lg_val],
                              early_stopping_rounds=100,
                              verbose_eval=200)

    def predict(self, Xp):
        preds = lgb_model.predict(Xp,
                                  num_iteration=lgb_model.best_iteration)

        preds = preds * 2 - 1

        return pd.DataFrame({'ast': data_df[ASSET], 'conf': preds})


if __name__ == '__main__':

    ENV = twosigmanews.make_env()
    (market_df, news_df) = ENV.get_training_data()
    data_df = prepare_data(market_df, news_df)
    del market_df, news_df  # save the precious memory

    print('building training set...')

    GROUP_COLS = [TIME, ASSET, 'universe', RETURN_10_NEXT]

    TRAIN_COLS = [col for col in data_df.columns
                  if col not in GROUP_COLS]

    dates = data_df[TIME].unique()

    train = range(len(dates))[:int(0.85 * len(dates))]
    train = data_df[TIME].isin(dates[train])

    val = range(len(dates))[int(0.85 * len(dates)):]
    val = data_df[TIME].isin(dates[val])

    # we be classifyin
    data_df[RETURN_10_NEXT] = get_ans(data_df)

    # train data
    Xt = data_df[TRAIN_COLS].loc[train].values
    Yt = data_df[[RETURN_10_NEXT]].loc[train].values

    # validation data
    Xv = data_df[TRAIN_COLS].loc[val].values
    Yv = data_df[[RETURN_10_NEXT]].loc[val].values

    pred_days = ENV.get_prediction_days()

    lgb_model = model(Xt, Yt, Xv, Yv)

    for market_df, news_df, pred_template_df in pred_days:
        data_df = prepare_data(market_df, news_df)
        Xp = data_df[TRAIN_COLS].values
        preds_df = lgb_model.predict()
        pred_template_df['confidenceValue'][
            pred_template_df[ASSET].isin(preds_df.ast)] = preds_df[
            'conf'].values
        ENV.predict(pred_template_df)

    ENV.write_submission_file()
