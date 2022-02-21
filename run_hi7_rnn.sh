#!bin/bash
START_DATE="2014-09-30"
TRAIN_PATH="./pyloader/hi7_train_p30n7v30_fix_20/"
TEST_PATH="./pyloader/hi7_test_p30n7v30_fix_20/"
ITER_DAYS=10
VALIDATION_WINDOW=30

CLF_NAME="meta.RNN"
REPORT_DIR="hi7_rnn/"

if [ ! -d ${REPORT_DIR} ]; then
mkdir $REPORT_DIR
fi

LEARNING_RATE=0.5
DOWN_SAMPLE=2
NUM_RESET=100
THRESHOLD=0.5
CLIP=5
SEED=1
RES_NAME="example.txt"
PATH_REPORT="${REPORT_DIR}${RES_NAME}"
TIME_PATH="${REPORT_DIR}time_${RES_NAME}"

CMD="java -cp simulate/target/simulate-2019.01.0-SNAPSHOT.jar:moa/target/moa-2019.01.0-SNAPSHOT.jar \
  simulate.Simulate -s $START_DATE -i $ITER_DAYS -p $TRAIN_PATH -t $TEST_PATH \
  -a ($CLF_NAME -r $LEARNING_RATE -s $NUM_RESET -o -c $CLIP) -H $THRESHOLD \
  -D $DOWN_SAMPLE -V $VALIDATION_WINDOW -r $SEED"
echo "$CMD > $PATH_REPORT"
time ($CMD > $PATH_REPORT) 2>> $TIME_PATH
