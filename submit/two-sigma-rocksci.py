# This Python 3 environment comes with many helpful analytics
# libraries installed
# It is defined by the kaggle/python docker
# image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter)
# will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

try:
    from kaggle.competitions import twosigmanews
except ModuleNotFoundError as module_error:
    from all import env_for_test as twosigmanews

import numpy as np


def make_random_predictions(predictions_df):
    """
    baseline
    """
    predictions_df.confidenceValue = 2.0 * np.random.rand(
            len(predictions_df)) - 1.0


def fit_model(market_train_df, news_train_df):
    # print(market_train_df.head())
    #
    # print(market_train_df.tail())
    #
    # print(news_train_df.head())
    pass


if __name__ == '__main__':

    # You can only call make_env() once, so don't lose it!
    env = twosigmanews.make_env()

    # You can only iterate through a result from `get_prediction_days()` once
    # so be careful not to lose it once you start iterating.
    days = env.get_prediction_days()

    market_train_df, news_train_df = env.get_training_data()
    fit_model(market_train_df, news_train_df)

    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        make_random_predictions(predictions_template_df)
        env.predict(predictions_template_df)
    print('Done!')

    env.write_submission_file()
