import numpy as np
import pandas as pd

try:
    from kaggle.competitions import twosigmanews
except Exception as error:
    print('local test')
    from all import env_for_test as twosigmanews

env = twosigmanews.make_env()
(marketdf, newsdf) = env.get_training_data()

print('preparing data...')


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def prepare_data(marketdf, newsdf):
    # a bit of feature engineering
    marketdf['time'] =pd.to_datetime(
            marketdf['time']
                ).dt.strftime("%Y%m%d").astype(int)
    marketdf['bartrend'] = marketdf['close'] / marketdf['open']
    marketdf['average'] = (marketdf['close'] + marketdf['open']) / 2
    marketdf['pricevolume'] = marketdf['volume'] * marketdf['close']

    newsdf['time'] = pd.to_datetime(
            newsdf['time']
                ).dt.strftime("%Y%m%d").astype(int)
    newsdf['assetCode'] = newsdf['assetCodes'].map(lambda x: list(eval(x))[0])
    newsdf['position'] = newsdf['firstMentionSentence'] / newsdf[
        'sentenceCount']
    newsdf['coverage'] = newsdf['sentimentWordCount'] / newsdf['wordCount']

    # filter pre-2012 data, no particular reason
    marketdf = marketdf.loc[marketdf['time'] > 20120000]

    # get rid of extra junk from news data
    droplist = ['sourceTimestamp', 'firstCreated', 'sourceId', 'headline',
                'takeSequence', 'provider', 'firstMentionSentence',
                'sentenceCount', 'bodySize', 'headlineTag', 'marketCommentary',
                'subjects', 'audiences', 'sentimentClass',
                'assetName', 'assetCodes', 'urgency', 'wordCount',
                'sentimentWordCount']
    newsdf.drop(droplist, axis=1, inplace=True)
    marketdf.drop(['assetName', 'volume'], axis=1, inplace=True)

    # combine multiple news reports for same assets on same day
    newsgp = newsdf.groupby(['time', 'assetCode'], sort=False).aggregate(
        np.mean).reset_index()

    # join news reports to market data, note many assets will have many days without news data
    return pd.merge(marketdf, newsgp, how='left', on=['time', 'assetCode'],
                    copy=False)  # , right_on=['time', 'assetCodes'])


cdf = prepare_data(marketdf, newsdf)
del marketdf, newsdf  # save the precious memory

#################################################################################
print('building training set...')
targetcols = ['returnsOpenNextMktres10']
traincols = [col for col in cdf.columns if
             col not in ['time', 'assetCode', 'universe'] + targetcols]

dates = cdf['time'].unique()
train = range(len(dates))[:int(0.85 * len(dates))-250]
val = range(len(dates))[int(0.85 * len(dates))-250:-250]

test = range(len(dates))[-250:]

# we be classifyin
cdf['tst'] = cdf[targetcols[0]]
cdf[targetcols[0]] = (cdf[targetcols[0]] > 0).astype(int) *2 -1


r = abs(cdf['tst'])
cdf[targetcols[0]][r > r.quantile()] /= 2
r[r > r.quantile()] /= 2
cdf[targetcols[0]][r > r.quantile()] /= 2
# train data
Xt = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[train])].values
Yt = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[train])].values

# validation data
Xv = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[val])].values
Yv = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[val])].values

# validation data
Xtest = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[test])].values
Ytest = cdf[['tst','time']].loc[cdf['time'].isin(dates[test])]


print(Xt.shape, Xv.shape)

#######################################################
##
## LightGBM
##
#######################################################
import lightgbm as lgb

print('Training lightgbm')

# money
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

lgtrain, lgval = lgb.Dataset(Xt, Yt[:, 0]), lgb.Dataset(Xv, Yv[:, 0])
lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval],
                     early_stopping_rounds=100, verbose_eval=200)

preds = lgbmodel.predict(Xtest, num_iteration=lgbmodel.best_iteration)

Ytest['res'] = Ytest['tst']* preds
res = Ytest.groupby('time').sum()['res'].values
print(res.mean()/res.std()) #0.14923083887714048
# ############################################################
# print("generating predictions...")
# preddays = env.get_prediction_days()
# for marketdf, newsdf, predtemplatedf in preddays:
#     cdf = prepare_data(marketdf, newsdf)
#     Xp = cdf[traincols].fillna(0).values
#     preds = lgbmodel.predict(Xp, num_iteration=lgbmodel.best_iteration) * 2 - 1
#     predsdf = pd.DataFrame({'ast': cdf['assetCode'], 'conf': preds})
#     predtemplatedf['confidenceValue'][
#         predtemplatedf['assetCode'].isin(predsdf.ast)] = predsdf['conf'].values
#     env.predict(predtemplatedf)
#
# env.write_submission_file()
















