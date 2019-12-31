#!bin/bash
# run and dump pickle
# 5: load, 6: save
START_DATE="2014-09-30"
TRAIN_PATH="./pyloader/hi7_train_p30n7v30_fix_20_reg/"
TEST_PATH="./pyloader/hi7_test_p30n7v30_fix_20_reg/"
ITER_DAYS=10
VALIDATION_WINDOW=30
CLF_NAME="trees.FIMTDD"

REPORT_DIR="hi7_example_reg/"
LABEL_DAYS=20

if [ ! -d ${REPORT_DIR} ]; then
mkdir $REPORT_DIR
fi

LAMBDA=1
DOWN_SAMPLE=6
SEED=1
RES_NAME="example.txt"
PATH_REPORT="${REPORT_DIR}${RES_NAME}"
TIME_PATH="${REPORT_DIR}time_${RES_NAME}"

CMD="java -cp simulate/target/simulate-2019.01.0-SNAPSHOT.jar:moa/target/moa-2019.01.0-SNAPSHOT.jar \
  simulate.Simulate -s $START_DATE -i $ITER_DAYS -p $TRAIN_PATH -t $TEST_PATH -l 2 -a ($CLF_NAME) -b $LAMBDA \
  -D $DOWN_SAMPLE -V $VALIDATION_WINDOW -r $SEED -g -L $LABEL_DAYS"
echo "$CMD > $PATH_REPORT"
time ($CMD > $PATH_REPORT) 2>> $TIME_PATH
