import pandas as pd
import datetime


class BasicOperation:
    def __init__(self, path, cur_date, model, columns):
        self.path = path
        self.cur_date = cur_date
        #self.manufacturer = manufacturer
        self.model = model
        self.columns = columns

    def read_data(self, window_size, features, drop):
        df_all = pd.DataFrame()
        for i in range(window_size):
            if self.columns == "all":
                df = pd.read_csv(self.path + self.cur_date.isoformat()[0:10] +
                                 ".csv")
            else:
                df = pd.read_csv(self.path + self.cur_date.isoformat()[0:10] +
                                 ".csv")
                df = df[self.columns]
            if self.model is not None:
                df = df[df['model'].isin(self.model)]
            #elif self.manufacturer is not None:
            #    df = df[df['model'].str[:2] == self.manufacturer]
            if drop:
                df = df.dropna(how='any', axis=0)
            df_all = pd.concat([df_all, df])
            self.cur_date += datetime.timedelta(days=1)
        df_all['date'] = pd.to_datetime(df_all['date'], format='%Y-%m-%d')
        return (df_all, self.cur_date)
