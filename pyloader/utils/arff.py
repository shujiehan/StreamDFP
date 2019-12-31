#!/usr/bin/env python
# -*-coding:utf-8-*-

import argparse
import numpy as np
import sys
import pandas as pd


class Arff:
    '''
      Inputs pandas data frame, determines attributes' types and converts
      to an ARFF file.
    '''

    def __init__(self, delimiter=',', bl_no_label=False, bl_regression=False):
        self.attribute_types = {}
        self.delimiter = delimiter
        self.bl_no_label = bl_no_label
        self.bl_regression = bl_regression

    def dump(self, name, df, output):
        self.read_csv(df)
        self.determine_attribute_types()
        self.write_arff(name, output)

    def read_csv(self, df):
        df = df.replace(np.nan, '', regex=True)
        self.columns = df.columns
        self.data = df.values

    def determine_attribute_types(self):
        for (i, attribute) in enumerate(self.columns):
            unique = list(set(self.data[:, i]))
            # Shujie add: customize for imbalance data
            if not self.bl_regression:
                if attribute == "failure" and len(unique) == 1:
                    unique = ['c0', 'c1']
                elif attribute == "errors" and len(unique) == 1:
                    unique = ['c0', 'c1']

            unique_value_index = 0
            while (unique_value_index < len(unique) and self.is_numeric(
                    str(unique[unique_value_index])) is not False):
                unique_value_index += 1

            self.attribute_types[attribute] = 'numeric'

            if (unique_value_index < len(unique)):
                unique.sort()
                unique = ["'%s'" % value for value in unique]
                self.attribute_types[attribute] = '{' + ','.join(unique) + '}'

                column_data = np.copy(self.data[:, i])
                for (data_index, value) in enumerate(column_data):
                    column_data[data_index] = "'%s'" % str(value)

                self.data[:, i] = column_data

    def write_arff(self, name, output):
        new_file = open(output, 'w')

        # Write relation
        new_file.write('@relation ' + str(name) + '\n\n')

        self.write_attributes(new_file)

        # Prepare data
        lines = []
        for row in self.data:
            str_row = ",".join(str(item) for item in row)
            lines.append(str_row)

        # Write data
        new_file.write('\n@data\n\n')
        new_file.write('\n'.join(lines))
        new_file.close()

    def is_numeric(self, lit):
        'Return value of numeric literal string or ValueError exception'
        if not len(lit):
            return 0
        # Handle '0'
        if lit == '0':
            return 0
        # Hex/Binary
        litneg = lit[1:] if lit[0] == '-' else lit
        if litneg[0] == '0':
            if litneg[1] in 'xX':
                return int(lit, 16)
            elif litneg[1] in 'bB':
                return int(lit, 2)
            else:
                try:
                    return int(lit, 8)
                except ValueError:
                    pass

        # Int/Float/Complex
        try:
            return int(lit)
        except ValueError:
            pass
        try:
            return float(lit)
        except ValueError:
            pass

        return False

    def write_attributes(self, new_file):
        if self.bl_no_label:
            self._write_attributes_without_labels(new_file)
        else:
            self._write_attributes_with_labels(new_file)

    def _write_attributes_without_labels(self, new_file):
        for index, column in enumerate(self.columns):
            new_file.write("@attribute col%i %s\n" %
                           (index, self.attribute_types[column]))

    def _write_attributes_with_labels(self, new_file):
        for column in self.columns:
            new_file.write(
                "@attribute %s %s\n" % (column, self.attribute_types[column]))


if __name__ == '__main__':
    df = pd.read_csv("2015-01-01.csv")
    arff = Arff()
    arff.dump("2015-01-01", df, "2015-01-01.arff")
