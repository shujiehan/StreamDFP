import pandas as pd
import datetime
import numpy as np


class BasicOperation:
    def __init__(self, path, cur_date, model, columns, bl_ssd):
        self.path = path
        self.cur_date = cur_date
        self.model = model
        self.columns = columns
        self.bl_ssd = bl_ssd
        if self.bl_ssd:
            self.failed_df = pd.read_csv(path + "ssd_failure_label.csv")
            self.failed_df['failure_time'] = pd.to_datetime(self.failed_df['failure_time']).dt.date
            self.failed_df = self.failed_df[['disk_id', 'model', 'failure_time']]
            self.failed_df = self.failed_df[self.failed_df['model'].isin(model)]
            self.failed_df['failure_time'] = pd.to_datetime(self.failed_df['failure_time'])

    def read_data(self, window_size, features, drop, date_format):
        df_all = pd.DataFrame()
        for i in range(window_size):
            if self.columns == "all":
                df = pd.read_csv(self.path + self.cur_date.strftime(date_format) +
                                 ".csv")
            else:
                df = pd.read_csv(self.path + self.cur_date.strftime(date_format) +
                                 ".csv")
                df = df[self.columns]
            if self.model is not None:
                df = df[df['model'].isin(self.model)]
            if self.bl_ssd:
                # Fixe mixed types in a column
                df = df.replace('\\N', np.nan)
                if drop:
                    df = df.dropna(how='any', axis=0)
                # remove samples after failure occurrences
                df['failure'] = np.where(df['disk_id'].isin(self.failed_df[
                            self.failed_df['failure_time'] < self.cur_date]['disk_id'].values), -1, 0)
                df = df[df['failure'] == 0]
                # label the samples
                df['failure'] = np.where(df['disk_id'].isin(self.failed_df[
                            self.failed_df['failure_time'] == self.cur_date]['disk_id'].values), 1, 0)
                # convert disk_id to string
                df['disk_id'] = 's' + df['disk_id'].astype('str')
                df.rename({'ds': 'date', 'disk_id': 'serial_number'}, axis=1, inplace=True)
                tmp_cols = df[features].select_dtypes(include='object').columns
                if len(tmp_cols) > 0:
                    for col in tmp_cols:
                        df[col] = df[col].astype('float64')
            df_all = pd.concat([df_all, df])
            self.cur_date += datetime.timedelta(days=1)
        df_all['date'] = pd.to_datetime(df_all['date'], format=date_format)
        return (df_all, self.cur_date)
