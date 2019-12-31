# StreamDFP

StreamDFP is a general stream mining framework for disk failure prediction with concept-drift adaptation. It includes feature extraction, labeling of samples, as well as training of a prediction model. 

StreamDFP is designed to support a variety of learning algorithms, based on three key techniques: online labeling, concept-drift-aware training, and general prediction.

We implement the prototype of StreamDFP in two parts. The first part is implemented in Python for preprocessing. We realize feature extraction, buffering, labeling, and first-phase downsampling. The second part is written in Java. We read the processed data from local file system for second-phase downsampling and training. We realize incremental algorithms and change detectors based on [Massive Online Analysis (MOA)](https://moa.cms.waikato.ac.nz/).

## Prerequisite

- Python3: Please install [numpy](https://numpy.org/) and [pandas](https://pandas.pydata.org/).
- Java: jdk-1.8.0

## Dataset

We use the following four disk models in public dataset [Backblaze](https://www.backblaze.com/b2/hard-drive-test-data.html):

- Seagate ST3000DM001
- Seagate ST4000DM000
- Seagate ST12000NM0007
- Hitachi HDS722020ALA330

You can also use other disk models for testing.

## Usage

### Preprocessing in Python

Please first go to `pyloader/` :

```
python run.py
-s <start_date> [--start_date <start_date>]
-a <label_days> [--label_days <label_days>]
-p <path_dataset> [--path <path_dataset>]
-r <train_data_path> [--train_path <train_data_path>]
-e <test_data_path> [--test_path <test_data_path>]
-c <path_features> [--path_features <path_features>]
-o <option> [--option <option>] (1: enable regression (classification by default))
```

For more details, please run `python run.py -h` or refer to an example script `run_hi7_loader.sh`.

### Training and prediction in Java

Please go back to `StreamDFP/`:

```
java -cp simulate/target/simulate-2019.01.0-SNAPSHOT.jar:moa/target/moa-2019.01.0-SNAPSHOT.jar simulate.Simulate
-s <start_date> 
-p <train_data_path>
-t <test_data_path>
-g [enable regression task]
-a <classifier>
-L <label_days>
-D <down_sample_ratio>
```

For more details, please refer to an example script `run_hi7.sh`.

### Example

Using *Hitachi HDS722020ALA330* as an example:

Assume the dataset storing `~/trace/smart/all/`

#### Classification:

1. open `pyloader/`;

2. run the script `run_hi7_loader.sh` to process 10-day data;

3. go back to `StreamDFP/`;

4. run the script `run_hi7.sh` to training prediction model of ARF and predict disk failures;

5. parse the results by running `python parse.py hi7_example/example.txt`

6. output the following results:

|   days    |    FP     |   FPR    | F1-score  | Precision |  Recall   |
| :-------: | :-------: | :------: | :-------: | :-------: | :-------: |
| 11.710000 | 22.200001 | 0.473107 | 26.220090 | 16.235855 | 68.095238 |

#### Regression

1. open `pyloader/`;

2. run the script `run_hi7_reg_loader.sh` to process 10-day data;

3. go back to `StreamDFP/`;

4. run the script `run_hi7_reg.sh` to training prediction model of ARF and predict disk failures;

5. parse the results by running `python parse_reg.py hi7_example/example.txt`

6. output the following results:

| days_mean | days_std | days_max | days_min  |
| :-------: | :------: | :------: | :-------: |
| 0.302072  | 5.017107 | 9.206787 | -7.110260 |

## Contact

Please email to Shujie Han (sjhan@cse.cuhk.edu.hk) if you have any questions.

