"""
env for local test use file from example before pull data
"""

import os

import pandas as pd


# todo try download all data from submit
class Env:
    """
    emulate kernal
    """

    module_path = os.path.dirname(__file__)
    file_list = [
        module_path + '/marketdata_sample.csv',
        module_path + '/news_sample.csv'
        ]

    result = []

    def __init__(self):
        self.data = []
        for file_name in self.file_list:
            with open(file_name, 'rb') as f:
                data = pd.read_csv(f)
                self.data.append(data)

    class IterDays:
        """
        for generate fit->predict data
        """
        class Result:
            """
            To save result per any date
            """
            def __init__(self, len_):
                self.confidenceValue = [0 for _ in range(len_)]

            def __len__(self):
                return len(self.confidenceValue)

        def __init__(self, data):
            self.limit = 2
            self.counter = 0
            self.data = data
            self.result = self.Result(len(data))

        def __next__(self):
            if self.counter < self.limit:
                self.counter += 1
                return self.data + [self.result]
            else:
                raise StopIteration

        def __iter__(self):
            return self

    def get_prediction_days(self):
        return self.IterDays(self.data)

    def get_training_data(self):
        return tuple(self.data)

    def predict(self, predict):
        self.result.append(predict)

    def write_submission_file(self):
        print(self.result)


def make_env():
    return Env()


if __name__ == '__main__':

    env = make_env()
    dates = env.get_prediction_days()
    print(dates)

    (market_train_df, news_train_df) = env.get_training_data()

    print(market_train_df.head())

    print(market_train_df.tail())

    print(news_train_df.head())
