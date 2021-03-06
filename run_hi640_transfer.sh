#!bin/bash
START_DATE="2014-09-30"
TRAIN_PATH="./pyloader/hi7_train_p30n7v30_fix_20/"
TEST_PATH="./pyloader/hi7_test_p30n7v30_fix_20/"
TRANSFER_START_DATE="2015-12-15"
TRANSFER_TRAIN_PATH="./pyloader/hi640_train_p30n7v30_fix_20_transfer/"
TRANSFER_TEST_PATH="./pyloader/hi640_test_p30n7v30_fix_20_transfer/"
ITER_DAYS=400
TRANSFER_ITER_DAYS=400
NUM_JOBS=-1
ENSEMBLE_SIZE=30
GRACE_PERIOD=50
SPLIT_CONFIDENCE=1e-7
VALIDATION_WINDOW=30
POSITIVE_WINDOW=30

CLF_NAME="meta.AdaptiveRandomForest"
TRANSFER_CLF_NAME="transfer.HomOTL"
REPORT_DIR="hi640_transfer/"

if [ ! -d ${REPORT_DIR} ]; then
mkdir $REPORT_DIR
fi

DRIFT_DELTA=1e-5
WARNING_DELTA=1e-4
DOWN_SAMPLE=10
LAMBDA=6
TRANSFER_DOWN_SAMPLE=29
LAMBDA_TRANSFER=10
DISCOUNT=0.99872 # Discounting weight parameter
SEED=1
RES_NAME="example.txt"
PATH_REPORT="${REPORT_DIR}${RES_NAME}"
TIME_PATH="${REPORT_DIR}time_${RES_NAME}"

CMD="java -cp simulate/target/simulate-2019.01.0-SNAPSHOT.jar:moa/target/moa-2019.01.0-SNAPSHOT.jar \
  simulate.Simulate -s $START_DATE -i $ITER_DAYS -p $TRAIN_PATH -t $TEST_PATH -a ($CLF_NAME -a $LAMBDA -s $ENSEMBLE_SIZE \
  -l (ARFHoeffdingTree -g $GRACE_PERIOD -c $SPLIT_CONFIDENCE) -j $NUM_JOBS -x (ADWINChangeDetector -a $DRIFT_DELTA) \
  -p (ADWINChangeDetector -a $WARNING_DELTA)) -D $DOWN_SAMPLE -O ${TRANSFER_DOWN_SAMPLE} -V $VALIDATION_WINDOW -r $SEED \
  -T -I $TRANSFER_ITER_DAYS -A ($TRANSFER_CLF_NAME -n ($CLF_NAME -a $LAMBDA_TRANSFER -s $ENSEMBLE_SIZE \
  -l (ARFHoeffdingTree -g $GRACE_PERIOD -c $SPLIT_CONFIDENCE) \
  -j $NUM_JOBS -x (ADWINChangeDetector -a $DRIFT_DELTA) -p (ADWINChangeDetector -a $WARNING_DELTA)) -w $DISCOUNT) -S $TRANSFER_START_DATE \
  -x $TRANSFER_TRAIN_PATH -y $TRANSFER_TEST_PATH -P $POSITIVE_WINDOW"
echo "$CMD > $PATH_REPORT"
time ($CMD > $PATH_REPORT) 2>> $TIME_PATH
