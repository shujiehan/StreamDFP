import sys
import datetime
import getopt
import pandas as pd
import pickle
from core_utils.abstract_predict import AbstractPredict
from utils.memory import Memory
from utils.arff import Arff


class Simulate(AbstractPredict):
    def __init__(self, path, date_format, start_date, positive_window_size, #manufacturer, \
            disk_model, columns, features, label, forget_type, bl_delay=False, \
            dropna=False, negative_window_size=6, validation_window=6, \
            bl_regression=False, label_days=None, bl_transfer=False, bl_ssd=False):
        super().__init__()
        self.memory = Memory(path, start_date, positive_window_size,  #manufacturer,\
            disk_model, columns, features, label, forget_type, dropna, bl_delay, \
            negative_window_size, bl_regression, label_days, bl_transfer, date_format, bl_ssd)
        if not bl_transfer:
            self.memory.buffering()
            self.data = self.memory.ret_df.drop(['model', 'date'], axis=1)
        else:
            self.data = self.memory.ret_df.drop(['model', 'date'], axis=1)
        self.data = self.data.reset_index(drop=True)
        self.class_name = label[0]
        self.num_classes = 2
        self.bl_delay = bl_delay
        self.validation_window = validation_window

    def load(self):
        # Load Data from Memory class and backtracking delayed instances
        self.memory.data_management(self.keep_delay, self.bl_delay)

        self.data = self.memory.ret_df.drop(['model', 'date'], axis=1)
        self.data = self.data.reset_index(drop=True)

    def delay_evaluate(self):
        pop_sn = []
        i = 0
        for sn, instances in self.keep_delay.items():
            instances.dequeue()
            if len(instances.queue) == 0:
                pop_sn.append(sn)
            i += 1
        for sn in pop_sn:
            self.keep_delay.pop(sn)

    def run(self):
        self.inspect(self.data, self.class_name, self.num_classes,
                     self.memory.new_inst_start_index, self.validation_window)


def run_simulating(start_date, path, path_load, path_save, train_path,
                   test_path, file_format, iter_days, model, features, label,
                   columns, forget_type, positive_window_size, bl_delay,
                   bl_load, bl_save, negative_window_size, validation_window,
                   bl_regression, label_days, bl_transfer, bl_ssd, date_format):
    if file_format == "arff":
        arff = Arff(bl_regression=bl_regression)
    if bl_load:
        with open(path_load, 'rb') as f:
            sim = pickle.load(f)
        print(sim.memory.cur_date)
        date = sim.memory.cur_date
        #sim.load()
    else:
        print(start_date)
        sim = Simulate(path, date_format, start_date, positive_window_size, model, columns,
                       features, label, forget_type, bl_delay, True,
                       negative_window_size, validation_window, bl_regression,
                       label_days, bl_transfer, bl_ssd)
        if not bl_transfer:
            fname = (sim.memory.cur_date -
                     datetime.timedelta(days=1)).isoformat()[0:10]

            if file_format == "arff":
                if not bl_regression:
                    sim.data['failure'] = sim.data['failure'].map({
                        0: 'c0',
                        1: 'c1'
                    })
                arff.dump(fname, sim.data, train_path + fname + ".arff")
            elif file_format == "csv":
                sim.data.to_csv(train_path + fname + ".csv", index=False)
            if test_path is not None and sim.memory.new_inst_start_index > 0:
                if file_format == "arff":
                    arff.dump(fname,
                              sim.data[sim.memory.new_inst_start_index:],
                              test_path + fname + ".arff")
                elif file_format == "csv":
                    sim.data[sim.memory.new_inst_start_index:].to_csv(
                        test_path + fname + ".csv", index=False)
            sim.run()
        else:
            print(sim.memory.cur_date)
            fname = (sim.memory.cur_date -
                     datetime.timedelta(days=1)).isoformat()[0:10]
            if test_path is not None:
                if file_format == "arff":
                    if not bl_regression:
                        sim.data['failure'] = sim.data['failure'].map({
                            0: 'c0',
                            1: 'c1'
                        })
                    arff.dump(fname,
                              sim.data[sim.memory.new_inst_start_index:],
                              test_path + fname + ".arff")
                elif file_format == "csv":
                    sim.data[sim.memory.new_inst_start_index:].to_csv(
                        test_path + fname + ".csv", index=False)
            for i in range(1, positive_window_size):
                sim.load()
                print(sim.memory.cur_date)
                fname = (sim.memory.cur_date -
                         datetime.timedelta(days=1)).isoformat()[0:10]
                if test_path is not None:
                    if file_format == "arff":
                        if not bl_regression:
                            sim.data['failure'] = sim.data['failure'].map({
                                0:
                                'c0',
                                1:
                                'c1'
                            })
                        arff.dump(fname,
                                  sim.data[sim.memory.new_inst_start_index:],
                                  test_path + fname + ".arff")
                    elif file_format == "csv":
                        sim.data[sim.memory.new_inst_start_index:].to_csv(
                            test_path + fname + ".csv", index=False)
            if file_format == "arff":
                arff.dump(fname, sim.data, train_path + fname + ".arff")
            elif file_format == "csv":
                sim.data.to_csv(train_path + fname + ".csv", index=False)
            sim.run()

    if bl_load is False and bl_delay:
        for i in range(validation_window):
            sim.load()
            print(sim.memory.cur_date)
            fname = (sim.memory.cur_date -
                     datetime.timedelta(days=1)).isoformat()[0:10]

            if file_format == "arff":
                if not bl_regression:
                    sim.data['failure'] = sim.data['failure'].map({
                        0: 'c0',
                        1: 'c1'
                    })
                arff.dump(fname, sim.data, train_path + fname + ".arff")
            elif file_format == "csv":
                sim.data.to_csv(train_path + fname + ".csv", index=False)
            if test_path is not None and sim.memory.new_inst_start_index > 0:
                if file_format == "arff":
                    arff.dump(fname,
                              sim.data[sim.memory.new_inst_start_index:],
                              test_path + fname + ".arff")
                elif file_format == "csv":
                    sim.data[sim.memory.new_inst_start_index:].to_csv(
                        test_path + fname + ".csv", index=False)
            sim.run()

    for ite in range(0, iter_days):
        print(sim.memory.cur_date)
        date = sim.memory.cur_date
        if bl_delay:
            sim.load()
            sim.delay_evaluate()
            fname = (sim.memory.cur_date -
                     datetime.timedelta(days=1)).isoformat()[0:10]

            if file_format == "arff":
                if not bl_regression:
                    sim.data['failure'] = sim.data['failure'].map({
                        0: 'c0',
                        1: 'c1'
                    })
                arff.dump(fname, sim.data, train_path + fname + ".arff")
            elif file_format == "csv":
                sim.data.to_csv(train_path + fname + ".csv", index=False)
            if test_path is not None and sim.memory.new_inst_start_index > 0:
                if file_format == "arff":
                    arff.dump(fname,
                              sim.data[sim.memory.new_inst_start_index:],
                              test_path + fname + ".arff")
                elif file_format == "csv":
                    sim.data[sim.memory.new_inst_start_index:].to_csv(
                        test_path + fname + ".csv", index=False)
            sim.run()
        else:
            sim.load()
            fname = (sim.memory.cur_date -
                     datetime.timedelta(days=1)).isoformat()[0:10]

            if file_format == "arff":
                if not bl_regression:
                    sim.data['failure'] = sim.data['failure'].map({
                        0: 'c0',
                        1: 'c1'
                    })
                arff.dump(fname, sim.data, train_path + fname + ".arff")
            elif file_format == "csv":
                sim.data.to_csv(train_path + fname + ".csv", index=False)
            if test_path is not None and sim.memory.new_inst_start_index > 0:
                if file_format == "arff":
                    arff.dump(fname,
                              sim.data[sim.memory.new_inst_start_index:],
                              test_path + fname + ".arff")
                elif file_format == "csv":
                    sim.data[sim.memory.new_inst_start_index:].to_csv(
                        test_path + fname + ".csv", index=False)
            sim.run()
    if bl_save:
        with open(path_save, 'wb') as f:
            pickle.dump(sim, f)


def usage(arg):
    print(arg, ":h [--help]")
    print("-s <start_date> [--start_date <start_date>]")
    print("-p <path_dataset> [--path <path_dataset>]")
    print("-l <path_load> [--path_load <path_load>]")
    print("-v <path_save> [--path_save <path_save>]")
    print("-c <path_features> [--path_features <path_features>]")
    print("-r <train_data_path> [--train_path <train_data_path>]")
    print("-e <test_data_path> [--test_path <test_data_path>]")
    print("-f <file_format> [--format <file_format>]")
    print("-o <option> [--option <option>]")
    print("-i <iter_days> [--iter_days <iter_days>]")
    print("-d <disk_model> [--disk_model <disk_model>]")
    print("-t <forget_type> [--forget_type <forget_type>]")
    print(
        "-w <positive_window_size> [--positive_window_size <positive_window_size>]"
    )
    print(
        "-L <negative_window_size> [--negative_window_size <negative_window_size>]"
    )
    print("-V <validation_window> [--validation_window <validation_window>]")
    print("-a <label_days> [--label_days <label_days>]")
    print("-F <date_format> [--date_format <date_format>]")
    print()
    print("Details:")
    print("path_load = load the Simulate class for continuing to process data")
    print(
        "path_save = save the Simulate class for continuing to process data next"
    )
    print(
        "file_format = file format of saving the processed data, arff by default"
    )
    print(
        "option = 1: enable regression (classification by default); 2: enable loading the Simulate class; 3: enable saving the Simulate class; 4: enable labeling; 5: enable transfer learning"
    )
    print(
        "forget_type = \"no\" (keep all historical data) or \"sliding\" (sliding window), \"sliding\" by default"
    )
    print(
        "positive_window_size = size of the sliding time window, 30 days by default"
    )
    print(
        "negative_window_size = size of the window for negative samples in 1-phase downsampling, 7 days by default"
    )
    print(
        "validation_window = size of window for evaluation, 30 days by default"
    )
    print("label_days = number of extra labeled days")


def get_parms():
    str_start_date = "2015-01-01"
    date_format = "%Y-%m-%d"
    path = "~/trace/smart/all/"
    train_path = "./train/"
    test_path = None
    path_load = None
    path_save = None
    bl_delay = False
    bl_load = False
    bl_save = False
    bl_regression = False
    bl_transfer = False
    bl_ssd = False
    option = {
        1: "bl_regression",
        2: "bl_load",
        3: "bl_save",
        4: "bl_delay",
        5: "bl_transfer",
        6: "bl_ssd"
    }

    file_format = "arff"
    iter_days = 5
    #manufacturer = None  #'ST'
    #model = 'ST4000DM000'
    model = []
    features = [
        'smart_1_normalized', 'smart_5_raw', 'smart_5_normalized',
        'smart_9_raw', 'smart_187_raw', 'smart_197_raw', 'smart_197_normalized'
    ]
    corr_attrs = []
    path_features = None
    label = ['failure']
    forget_type = "sliding"
    label_days = None
    positive_window_size = 30
    negative_window_size = 7
    validation_window = 30

    try:
        (opt, args) = getopt.getopt(
            sys.argv[1:], "hs:p:l:v:c:r:e:f:o:i:d:t:w:L:V:a:F:", [
                "help", "start_date", "path", "path_load", "path_save",
                "path_features", "train_path", "test_path", "file_format",
                "option", "iter_days", "disk_model", "forget_type",
                "positive_window_size", "negative_window_size",
                "validation_window", "label_days", "date_format"
            ])
    except:
        usage(sys.argv[0])
        print("getopts exception")
        sys.exit(1)

    for o, a in opt:
        if o in ("-h", "--help"):
            usage(sys.argv[0])
            sys.exit(0)
        elif o in ("-s", "--start_date"):
            str_start_date = a
        elif o in ("-p", "--path"):
            path = a
        elif o in ("-l", "--path_load"):
            path_load = a
        elif o in ("-v", "--path_save"):
            path_save = a
        elif o in ("-c", "--path_features"):
            path_features = a
        elif o in ("-f", "--file_format"):
            file_format = a
        elif o in ("-r", "--train_path"):
            train_path = a
        elif o in ("-e", "--test_path"):
            test_path = a
        elif o in ("-o", "--option"):
            ops = a.split(",")
            for op in ops:
                if int(op) == 1:
                    bl_regression = True
                elif int(op) == 2:
                    bl_load = True
                elif int(op) == 3:
                    bl_save = True
                elif int(op) == 4:
                    bl_delay = True
                elif int(op) == 5:
                    bl_transfer = True
                elif int(op) == 6:
                    bl_ssd = True
        elif o in ("-i", "--iter_days"):
            iter_days = int(a)
        elif o in ("-d", "--disk_model"):
            model = a.split(",")
        elif o in ("-t", "--forget_type"):
            forget_type = a
        elif o in ("-w", "--positive_window_size"):
            positive_window_size = int(a)
        elif o in ("-L", "--negative_window_size"):
            negative_window_size = int(a)
        elif o in ("-V", "--validation_window"):
            validation_window = int(a)
        elif o in ("-a", "--label_days"):
            label_days = int(a)
        elif o in ("-F", "--date_format"):
            date_format = a

    if str_start_date.find("-") != -1:
        start_date = datetime.datetime.strptime(str_start_date, "%Y-%m-%d")
    else:
        start_date = datetime.datetime.strptime(str_start_date, "%Y%m%d")
    if path_features is not None:
        features = []
        with open(path_features, "r") as f:
            for line in f.readlines():
                features.append(line.strip())
        print(features)

    if bl_ssd:
        columns = ['ds', 'model', 'disk_id'] + features
    else:
        columns = ['date', 'model', 'serial_number'] + label + features
    return (start_date, path, path_load, path_save, train_path, test_path,
            file_format, bl_delay, bl_load, bl_save, iter_days, model,
            features, label, columns, forget_type, positive_window_size,
            negative_window_size, validation_window, bl_regression, label_days,
            bl_transfer, bl_ssd, date_format)


if __name__ == "__main__":
    (start_date, path, path_load, path_save, train_path, test_path,
     file_format, bl_delay, bl_load, bl_save, iter_days, disk_model, features,
     label, columns, forget_type, positive_window_size, negative_window_size,
     validation_window, bl_regression, label_days, bl_transfer, bl_ssd, date_format) = get_parms()

    run_simulating(start_date, path, path_load, path_save, train_path,
                   test_path, file_format, iter_days, disk_model, features,
                   label, columns, forget_type, positive_window_size, bl_delay,
                   bl_load, bl_save, negative_window_size, validation_window,
                   bl_regression, label_days, bl_transfer, bl_ssd, date_format)
