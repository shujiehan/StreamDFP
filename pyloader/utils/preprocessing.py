import pandas as pd
import numpy as np
from utils.basics import BasicOperation


class Preprocessing:
    def __init__(self, raw_data, label, path_metadata, path_failed_disks_sn):
        """
        Parameters:
            raw_data: need to be preprocessed
            label: 'failure' for disk failure; 'smart_5_raw' for latent sector errors
            path_metadata: The path where stored all disks metadata
            path_failed_disks_sn: The path where stored failed disks sn.
        """
        self.raw_data = raw_data
        self.label = label
        self.path_metadata = path_metadata
        try:
            self.metadata = pd.read_csv(path_metadata, index_col=0)
        except:
            self.metadata = pd.DataFrame(columns=[
                'serial_number', 'model', 'collect_days', 'start_date',
                'failed_date'
            ])
        self.path_failed_disks_sn = path_failed_disks_sn
        try:
            self.failed_sn = pd.read_csv(path_failed_disks_sn, index_col=0)
        except:
            self.failed_sn = pd.DataFrame(columns=['serial_number'])
        self.delta_meta = pd.DataFrame()
        self.delta_sn = []

    def update_raw_data(self, raw_data):
        self.raw_data = raw_data

    def filter_failed_disks(self):
        sn = self.raw_data[self.raw_data['failure'] == 1][['serial_number']]
        self.failed_sn = pd.concat([self.failed_sn, sn]).drop_duplicates()
        self.failed_sn = self.failed_sn.reset_index(drop=True)

    def __update_failed_date(self, x, option):
        if option == 'failed_sn':
            if x['failure'] == 1:
                return x['start_date']
            return np.nan
        if option == 'metadata':
            this_df = self.delta_meta[self.delta_meta['serial_number'] == x[
                'serial_number']]
            if this_df['failure'].values == 1:
                return this_df['date']
            return np.nan

    def __update_collect_days(self, x):
        if x['serial_number'] in self.delta_sn:
            return x['collect_days'] + 1
        return x['collect_days']

    def update_metadata(self):
        self.delta_meta = self.raw_data[[
            'serial_number', 'model', 'date', 'failure'
        ]]
        self.delta_sn = self.delta_meta['serial_number'].values
        meta_sn = self.metadata['serial_number'].values

        if len(self.metadata.index) > 0:
            # update existing sn in metadata
            self.metadata['collect_days'] = self.metadata.apply(
                self.__update_collect_days, axis=1)
            self.metadata['failed_date'] = self.metadata.apply(
                self.__update_failed_date, axis=1, option='metadata')
        ## drop existing sn in metadata for delta_meta
        self.delta_meta = self.delta_meta[
            ~self.delta_meta['serial_number'].isin(meta_sn)]
        if len(self.delta_meta.index) > 0:
            ## update delta_meta for concatenating it to metadata
            self.delta_meta['collect_days'] = 1
            self.delta_meta = self.delta_meta.rename(
                index=str, columns={'date': 'start_date'})
            self.delta_meta = self.delta_meta.reset_index(drop=True)
            self.delta_meta['failed_date'] = self.delta_meta.apply(
                self.__update_failed_date, axis=1, option='failed_sn')
            self.delta_meta = self.delta_meta.drop(['failure'], axis=1)

        if self.metadata.empty:
            self.metadata = self.metadata.append(self.delta_meta)
        else:
            if self.delta_meta.empty is False:
                self.metadata = pd.concat([self.metadata, self.delta_meta])
            self.metadata = self.metadata.reset_index(drop=True)

    @staticmethod
    def append_to_csv(df, csv_file_path, sep=",", index=False, index_col=0):
        import os
        if not os.path.isfile(csv_file_path):
            df.to_csv(csv_file_path, mode='a', index=index, sep=sep)
        elif len(df.columns) != len(
                pd.read_csv(
                    csv_file_path, nrows=1, sep=sep,
                    index_col=index_col).columns):
            raise Exception("Columns do not match!! Dataframe has " + str(
                len(df.columns)
            ) + " columns. CSV file has " + str(
                len(
                    pd.read_csv(
                        csv_file_path, nrows=1, sep=sep, index_col=index_col)
                    .columns)) + " columns.")
        elif not (df.columns == pd.read_csv(
                csv_file_path, nrows=1, sep=sep,
                index_col=index_col).columns).all():
            raise Exception(
                "Columns and column order of dataframe and csv file do not match!!"
            )
        else:
            df.to_csv(
                csv_file_path, mode='a', index=index, sep=sep, header=False)
