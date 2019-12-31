import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from instances.instance import Instance
from instances.instances import Instances
from itertools import islice


class AbstractPredict(metaclass=ABCMeta):
    """
    This class is used to unit test for validating correctness.
    """

    def __init__(self):
        # keep a sequence of instances of one disk
        # dict{sn:Instances}
        self.keep_delay = {}
        return

    def keep(self, inst, queue_size):
        if inst.sn in self.keep_delay.keys():
            self.keep_delay[inst.sn].enqueue(inst)
        else:
            instances = Instances(inst.sn, queue_size)
            instances.enqueue(inst)
            self.keep_delay[inst.sn] = instances

    def inspect(self, data, class_name, num_classes, inspect_start_idx,
                validation_window):
        sns = data['serial_number'].values
        data = data.drop(['serial_number'], axis=1)
        if inspect_start_idx > 0:
            for index, row in islice(data.iterrows(), inspect_start_idx, None):
                inst = Instance(1, sns[index], row, class_name, num_classes)
                self.keep(inst, validation_window)
