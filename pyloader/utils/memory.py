import sys
import datetime
import pandas as pd
import numpy as np
from utils.basics import BasicOperation
from utils.preprocessing import Preprocessing


class Memory:
    FORGET_NO = "no"  # remember all data
    FORGET_SLI = "sliding"  # sliding window


    def __init__(self, path, cur_date, positive_window_size, model, columns, features, \
                 label, forget_type, dropna, delay, negative_window_size, \
                 bl_regression, label_days, bl_transfer, date_format, bl_ssd):
        self.start_date = cur_date - datetime.timedelta(
            days=(positive_window_size))
        self.positive_window = datetime.timedelta(days=positive_window_size)
        self.forget_type = forget_type
        self.features = features
        self.label = label
        self.dropna_option = dropna
        self.negative_window = datetime.timedelta(days=negative_window_size)
        self.positive_window_size = positive_window_size
        self.basic_oper = BasicOperation(path, cur_date, model, columns, bl_ssd)
        self.date_format = date_format

        self.bl_regression = bl_regression
        self.bl_transfer = bl_transfer
        self.label_days = datetime.timedelta(days=label_days)
        self.new_inst_start_index = 0
        (self.df, self.cur_date) = self.basic_oper.read_data(
            1, self.features, self.dropna_option, self.date_format)
        self.ret_df = self.df
        self.start_date = self.cur_date - \
                datetime.timedelta(days=(self.positive_window_size))

    def buffering(self):
        for i in range(1, self.positive_window_size):
            (df_delta, cur_date) = self.basic_oper.read_data(
                1, self.features, self.dropna_option, self.date_format)
            df_failed = df_delta[df_delta['failure'] == 1]
            if len(df_failed.index) > 0:
                self.df = self.labeling(df_failed['serial_number'].values,
                                        None)
            self.df = pd.concat([self.df, df_delta])

            self.cur_date = cur_date

        # 1-phase down-sampling
        self.ret_df = self.cleaning(self.df)

        self.new_inst_start_index = 0

        self.start_date = self.cur_date - \
                datetime.timedelta(days=(self.positive_window_size))

    def cleaning(self, df):
        ret_df = df.copy()
        ret_df = ret_df[~(
            (ret_df['failure'] == 0) &
            (ret_df['date'] < (self.cur_date - self.negative_window)))]
        return ret_df

    def __labeling(self, x):
        if x['date'] >= (self.cur_date - self.label_days):
            x['failure'] = 1
        return x

    def __labeling_reg(self, x):
        if x['date'] >= (self.cur_date - self.label_days):
            x['failure'] = 1 - (self.cur_date - x['date']).days * 1.0 / (
                self.label_days.days + 1.0)
        return x

    def labeling(self, sns, keep_delay):
        self.df = self.df.reset_index(drop=True)
        index = np.where(self.df['serial_number'].isin(sns))
        if self.bl_regression:
            self.df.update(self.df.iloc[index].apply(
                self.__labeling_reg, axis=1))
        else:
            self.df.update(self.df.iloc[index].apply(self.__labeling, axis=1))
        return self.df

    def data_management(self, keep_delay=None, delay=False):
        (df_delta, cur_date) = self.basic_oper.read_data(
            1, self.features, self.dropna_option, self.date_format)
        if delay:
            df_failed = df_delta[df_delta['failure'] == 1]
            if len(df_failed.index) > 0:
                self.df = self.labeling(df_failed['serial_number'].values,
                                        None)
        else:
            self.df = self.df[self.df['failure'] == 1]
        self.cur_date = cur_date
        # Remember all data
        if self.forget_type == self.FORGET_NO:
            self.new_inst_start_index = len(self.df.index)
            self.df = pd.concat([self.df, df_delta])
        # Sliding window
        elif self.forget_type == self.FORGET_SLI:
            # keep samples in the positive window
            self.df = self.df[self.df['date'] > self.start_date]

            # 1-phase down-sampling
            self.ret_df = self.cleaning(self.df)

            self.new_inst_start_index = len(self.ret_df.index)
            self.start_date = self.cur_date - \
                    datetime.timedelta(days=(self.positive_window_size))
            self.ret_df = pd.concat([self.ret_df, df_delta])
            self.df = pd.concat([self.df, df_delta])
        else:
            print(
                "Unknown sliding window options, it should be \'fixed\' or \'variable\'."
            )
            sys.exit(1)
        return cur_date
