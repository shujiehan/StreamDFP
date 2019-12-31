import math
from abc import ABCMeta, abstractmethod


class Instance(metaclass=ABCMeta):
    """
    A class is to handle an instance.
    """

    def __init__(self, weight, sn, data, class_name, num_classes):
        """
        Parameters:
            1. instance
            2. weight, res -> DenseInstanceData
            3. weight, attribute_valus, index_values, number_attributes -> SparseInstanceData
        """
        self.weight = weight
        self.sn = sn
        self.instance_header = list(data.index)
        self.instance_data = list(data)  # pd.Series
        self.class_name = class_name
        self.num_classes = num_classes
        self.predicted_votes = None

    def keep_predicted_votes(self, votes):
        self.predicted_votes = votes

    def get_predicted_votes(self):
        return self.predicted_votes

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def attribute(self, inst_att_index):
        """
        Return the attribute with the given index.
        """
        #return self.instance_header.attribute(inst_att_index)
        return self.instance_header[inst_att_index]

    def index_of_attribute(self, attribute):
        #return self.instance_header.index_of(attribute)
        return self.instance_header.index(attribute)

    def delete_attribute_at(self, i):
        del self.instance_header[i]
        del self.instance_data[i]
        #self.instance_data.delete_attribute_at(i)

    def insert_attribute_at(self, i, attribute, value):
        self.instance_header(i, attribute)
        self.instance_data(i, value)
        #self.instance_data.insert_attribute_at(i)

    def num_attributes(self):
        return len(self.instance_header)

    def value(self, *args):
        """
        Parameters:
            args:
                the index of instance data;
                or it can be the attribute name

        """
        if len(args) == 1 and isinstance(args[0], int):
            return self.instance_data[args[0]]
        elif len(args) == 1 and isinstance(args[0], str):
            index = self.instance_header.index(args[0])
            return self.instance_data[index]

    def is_missing(self, *args):
        """
        Parameters:
            args:
                the index of instance data;
                or it can be the attribute

        """
        if len(args) == 1 and isinstance(args[0], int):
            index = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            index = self.instance_header.index(args[0])
        return self.instance_data[index] == None

    def set_missing(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            index = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            index = self.instance_header.index(args[0])
        self.set_value(index, None)

    def num_values(self):
        return len(self.instance_data)

    def index(self, i):
        return self.instance_data[i]

    def set_value(self, *args, value):
        """
        Parameters:
            args:
                the index of instance data;
                or it can be the attribute

        """
        if len(args) == 1 and isinstance(args[0], int):
            index = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            index = self.instance_header.index(args[0])
        self.instance_data[index] = value

    def class_value(self):
        return self.instance_data[self.class_index()]

    def class_index(self):
        class_index = self.instance_header.index(self.class_name)
        return class_index

    def get_num_classes(self):
        return self.num_classes

    def class_is_missing(self):
        return math.isnan(self.instance_data[self.class_index()])

    def class_attribute(self):
        return self.class_name

    def set_class_value(self, value):
        self.instance_data[self.class_index()] = value

    def dataset(self):
        """
        Return: the instances
        """
        return self.instance_header

    def attribute_is_nominal(self, inst_att_index):
        if isinstance(float(self.instance_data[int(inst_att_index)]), float):
            return False
        return True
