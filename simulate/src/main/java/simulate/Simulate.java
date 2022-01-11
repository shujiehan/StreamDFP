package simulate;

import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.transfer.HomOTL;
import moa.core.ObjectRepository;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.BasicRegressionPerformanceEvaluator;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class Simulate extends AbstractPredict{
    public Date curDate;
    public String testPath;
    public String trainPath;
    public String fileName;
    public LearningPerformanceEvaluator globalEvaluator;
    public LearningPerformanceEvaluator localEvaluator;
    public AbstractClassifier scheme;
    public boolean blDelay;
    public Instances trainingData;
    public Instances testData;
    public int numClasses;
    public int iterations;
    public boolean blModelMeasurement;
    public double downSampleRatio;

    // Shujie added on Nov. 2
    public Date transferCurDate;
    public String targetTrainPath;
    public String targetTestPath;
    public String targetFileName;
    public Instances targetTrainingData;
    public Instances targetTestData;
    public int transferIterations;
    public AbstractClassifier transferScheme;
    public int positiveWindowSize;
    public double transferDownSampleRatio;

    public double threshold;

    private SimpleDateFormat dateFormat;

    public ClassOption abstractClassifierOption = new ClassOption("abstractClassifier", 'a',
            "Abstract Classifier", AbstractClassifier.class,
            "meta.AdaptiveRandomForest -l (ARFHoeffdingTree -g 50 -c 0.0000001) -s 30 -j -1");
    public StringOption testPathOption = new StringOption("testPath", 't', "Test data path.",
            "/home/shujiehan/MySoftwares/streamDFP/pyloader/test/");

    public StringOption trainingPathOption = new StringOption("trainingPath", 'p',
            "Training data path.", "/home/shujiehan/MySoftwares/streamDFP/pyloader/train/");

    public StringOption startDateOption = new StringOption("startDate", 's', "Start Date.",
            "2015-01-30");

    public IntOption iterationsOption = new IntOption("iterations", 'i', "number of iterations", 30);
    public IntOption cindexOption = new IntOption("cindex", 'c', "class index", 1);

    public FlagOption blDelayOption = new FlagOption("blDelay", 'd', "disable delay validation");

    public IntOption labelWindowOption = new IntOption("labelWindowSize", 'L',
            "label window size", 6);

    public IntOption validationWindowOption = new IntOption("validationWindowSize", 'V',
            "validation window size.", 30);

    public FloatOption downSampleRatioOption = new FloatOption("downSampleRatio", 'D',
            "down sampling ratio", 5);

    public IntOption randomSeedOption = new IntOption("randomSeed", 'r', "random seed", 1);

    public FlagOption modelMeasurementOption = new FlagOption("modelMeasurements", 'm',
            "enable model measurements");
    public FlagOption regressionOption = new FlagOption("regressionTask", 'g', "regression task");

    // Shujie added on Nov. 2
    public FlagOption transferOption = new FlagOption("transferLearningTask", 'T', "transfer learning task");
    public IntOption transferIterationsOption = new IntOption("transferIterations", 'I',
            "number of iterations for transfer learning", 30);
    public ClassOption transferClassifierOption = new ClassOption("transferClassifier", 'A',
            "Abstract Classifier", AbstractClassifier.class,
            "transfer.HomOTL -n (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -g 50 -c 0.0000001)) -w 0.5");
    public FloatOption transferDownSampleRatioOption = new FloatOption("transferDownSampleRatio", 'O',
            "down sampling ratio for transfer learning", 5);

    public StringOption transferCurDateOption = new StringOption("transferCurDate", 'S',
            "Transfer learning start Date.", "2015-01-30");

    public StringOption targetTrainingPathOption = new StringOption("targetTrainingPath", 'x',
            "Training data path for the target domain in transfer learning task", "pyloader/target_train/");

    public StringOption targetTestPathOption = new StringOption("targetTestPath", 'y',
            "Test data path for the target domain in transfer learning task", "pyloader/target_test/");

    public IntOption positiveWindowSizeOption = new IntOption("positiveWindow", 'P',
            "Window size that buffers the positive samples.", 30);

    public FloatOption thresholdOption = new FloatOption("thresholdForPrediction", 'H',
            "Threshold for determining positive and negative class", 0.5, 0, 1);


    public Simulate () {
        super();
        prepareForUse();
    }

    public void init() throws Exception {
        Date startDate = new Date();
        this.dateFormat= new SimpleDateFormat("yyyy-MM-dd");
        String strStartDate = startDateOption.getValue();
        try {
            startDate = this.dateFormat.parse(strStartDate);
        } catch (ParseException e) {
            e.printStackTrace();
        }
        this.curDate = startDate;
        this.testPath = testPathOption.getValue();
        this.trainPath = trainingPathOption.getValue();
        this.fileName = strStartDate + ".arff";
        this.cindex = cindexOption.getValue();
        this.iterations = iterationsOption.getValue();
        this.threshold = thresholdOption.getValue();

        blDelayOption.set();
        this.blDelay = blDelayOption.isSet();
        this.blModelMeasurement = this.modelMeasurementOption.isSet();
        this.blRegression = this.regressionOption.isSet();

        this.numClasses = 2;
        this.scheme = (AbstractClassifier) ClassOption.cliStringToObject(
                this.abstractClassifierOption.getValueAsCLIString(), AbstractClassifier.class, null);

        this.randomSeed = this.randomSeedOption.getValue();
        this.scheme.prepareForUse();
        this.scheme.setRandomSeed(this.randomSeed);
        this.scheme.resetLearning();

        if (blRegression) {
            this.labelWindow = labelWindowOption.getValue(); // Fixed window size for regression
        }
        this.validationWindow = validationWindowOption.getValue();
        this.globalEvaluator = resetEvaluator();
        this.localEvaluator = resetEvaluator();

        this.downSampleRatio = downSampleRatioOption.getValue();
        this.keepDelayInstances = new HashMap<>();
        this.trainingData = load(trainPath + fileName, cindex);

        InstancesHeader ih = new InstancesHeader(this.trainingData);
        this.scheme.setModelContext(ih);

        this.samplingRandom = new Random(this.randomSeed);

        updateModel(this.trainingData, this.scheme, this.downSampleRatio);

        this.blTransfer = this.transferOption.isSet();
        if (blTransfer) {
            this.targetTrainPath = targetTrainingPathOption.getValue();
            this.targetTestPath = targetTestPathOption.getValue();
            this.transferCurDate = this.dateFormat.parse(transferCurDateOption.getValue());
            this.targetFileName = this.dateFormat.format(transferCurDate) + ".arff";
            targetTestData = load(targetTestPath + targetFileName, cindex);
            this.transferIterations = transferIterationsOption.getValue();
            this.transferDownSampleRatio = transferDownSampleRatioOption.getValue();
            this.transferScheme = (AbstractClassifier) ClassOption.cliStringToObject(
                    this.transferClassifierOption.getValueAsCLIString(),
                    AbstractClassifier.class, null);
            this.positiveWindowSize = positiveWindowSizeOption.getValue();
            this.transferScheme.prepareForUse();
            this.transferScheme.setRandomSeed(this.randomSeed);
            this.transferScheme.resetLearning();
            ih = new InstancesHeader(this.targetTestData);
            this.transferScheme.setModelContext(ih);
        }
    }

    public LearningPerformanceEvaluator resetEvaluator() {
        if (blRegression) {
            return resetRegressionEvaluator();
        } else {
            return resetClassificationEvaluator();
        }
    }

    public BasicRegressionPerformanceEvaluator resetRegressionEvaluator() {
        BasicRegressionPerformanceEvaluator evaluator = new BasicRegressionPerformanceEvaluator();
        evaluator.setLabelWindowSize(labelWindow);
        return evaluator;
    }

    public BasicClassificationPerformanceEvaluator resetClassificationEvaluator () {
        BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();
        evaluator.reset(this.numClasses);
        evaluator.precisionRecallOutputOption.set();
        evaluator.precisionPerClassOption.set();
        evaluator.recallPerClassOption.set();
        evaluator.f1PerClassOption.set();
        evaluator.confusionMatrixOption.set();
        evaluator.falseAlarmOption.set();
        evaluator.setThreshold(threshold);
        return evaluator;
    }

    public void loadData() {
        curDate = new Date(curDate.getTime() + 1000*24*60*60);
        this.fileName = this.dateFormat.format(curDate) + ".arff";
        testData = load(testPath + fileName, cindex);
        trainingData = load(trainPath + fileName, cindex);
    }

    // for transfer learning
    public void loadData(String domainOption, String dataOption) {
        if (domainOption.equals(SOURCE_DOMAIN)) {
            loadData();
        } else if (domainOption.equals(TARGET_DOMAIN)) {
            if (dataOption.equals(TEST)) {
                transferCurDate = new Date(transferCurDate.getTime() + 1000*24*60*60);
                this.targetFileName = this.dateFormat.format(transferCurDate) + ".arff";
                targetTestData = load(targetTestPath + targetFileName, cindex);
            } else if (dataOption.equals(TRAIN)) {
                targetTrainingData = load(targetTrainPath + targetFileName, cindex);
            } else if (dataOption.equals(TEST_TRAIN)) {
                transferCurDate = new Date(transferCurDate.getTime() + 1000*24*60*60);
                this.targetFileName = this.dateFormat.format(transferCurDate) + ".arff";
                targetTestData = load(targetTestPath + targetFileName, cindex);
                targetTrainingData = load(targetTrainPath + targetFileName, cindex);
            }
        }

    }

    public InspectionData delayEvaluate(AbstractClassifier scheme) {
        InspectionData result = new InspectionData();
        boolean report = false;
        Iterator it = keepDelayInstances.entrySet().iterator();
        List<String> popSN = new ArrayList<>();
        int index = 0;
        while(it.hasNext()) {
            HashMap.Entry pair = (HashMap.Entry) it.next();
            if (index == (keepDelayInstances.keySet().size() - 1)) report = true;
            List<Instance> instances = (List<Instance>) pair.getValue();
            assert (instances.size() <= this.validationWindow);
            if (blRegression) {
                ((BasicRegressionPerformanceEvaluator)globalEvaluator).addResultDelay(instances);
                ((BasicRegressionPerformanceEvaluator)localEvaluator).addResultDelay(instances);
            } else {
                ((BasicClassificationPerformanceEvaluator)globalEvaluator).addResultDelay(instances);
                ((BasicClassificationPerformanceEvaluator)localEvaluator).addResultDelay(instances);
            }
            if (report) {
                result = evaluateDelay(scheme, globalEvaluator, localEvaluator, index, this.blModelMeasurement);
            }
            ((LinkedList)instances).removeFirst();
            if (instances.size() == 0) {
                popSN.add((String) pair.getKey());
            }
            keepDelayInstances.put((String) pair.getKey(), instances);
            index ++;
        }
        for (String sn : popSN) {
            keepDelayInstances.remove(sn);
        }
        return result;
    }

    public InspectionData run() {
        InspectionData result = inspect(testData, globalEvaluator, localEvaluator, scheme, blDelay,
                validationWindow);
        updateModel(trainingData, scheme, downSampleRatio);
        return result;
    }

    // for transfer learning
    public InspectionData run(String domainOption, String operationOption) {
        InspectionData result = null;
        if (domainOption.equals(SOURCE_DOMAIN)) {
            updateModel(trainingData, scheme, downSampleRatio);
        } else if (domainOption.equals(TARGET_DOMAIN)) {
           if (operationOption.equals(TEST)) {
               // Since the target model is still not trained,
               // we use source model for prediction during the first positive window.
               result = inspect(targetTestData, globalEvaluator, localEvaluator, scheme, blDelay,
                       validationWindow);
           } else if (operationOption.equals(TRAIN)) {
               updateModel(targetTrainingData, transferScheme, transferDownSampleRatio);
           } else if (operationOption.equals(TEST_TRAIN)) {
               result = inspect(targetTestData, globalEvaluator, localEvaluator, transferScheme, blDelay,
                       validationWindow);
               updateModel(targetTrainingData, transferScheme, transferDownSampleRatio);
           }
        }
        return result;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        return;
    }

    @Override
    public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository){
        return;
    }

    public static void main (String args[]) throws Exception {
        String s = "Simulate ";
        for (int i = 0; i < args.length; i ++) {
            if (i < args.length - 1)
                s += (args[i] + " ");
            else s += args[i];
        }
        Simulate sim = (Simulate)ClassOption.cliStringToObject(s, Simulate.class, null);
        sim.init();
        if (sim.blTransfer) {
            // check overlap between the periods of source and target domain
            // Assumption: no target domain in warm-up period of source domain
            for (int i = 0; i < sim.validationWindow; i++) {
                sim.loadData();
                sim.run(sim.SOURCE_DOMAIN, null);
            }
            int idxSourceIterations = 0;
            while (idxSourceIterations < sim.iterations && sim.curDate.before(sim.transferCurDate)) {
                sim.loadData();
                sim.run(sim.SOURCE_DOMAIN, null);
                idxSourceIterations++;
            }
            sim.cleanKeepDelayInstances();
            sim.globalEvaluator = sim.resetEvaluator();
            InspectionData result;
            for (int i = 0; i < sim.positiveWindowSize - 1; i++) {
                sim.loadData(sim.TARGET_DOMAIN, sim.TEST);
                sim.run(sim.TARGET_DOMAIN, sim.TEST);
            }
            // initialize the target model
            sim.cleanKeepDelayInstances();
            sim.globalEvaluator = sim.resetEvaluator();
            ((HomOTL)sim.transferScheme).setOldClassifier(sim.scheme);
            sim.loadData(sim.TARGET_DOMAIN, sim.TRAIN);
            sim.run(sim.TARGET_DOMAIN, sim.TRAIN);

            // update the target model
            for (int i = 0; i < sim.validationWindow; i++) {
                if (idxSourceIterations < sim.iterations) {
                    // update the source model if it doesn't finish iterations
                    sim.loadData();
                    sim.run(sim.SOURCE_DOMAIN, null);
                    ((HomOTL)sim.transferScheme).setOldClassifier(sim.scheme);
                    idxSourceIterations++;
                }
                sim.loadData(sim.TARGET_DOMAIN, sim.TEST_TRAIN);
                sim.run(sim.TARGET_DOMAIN, sim.TEST_TRAIN);
            }

            sim.globalEvaluator = sim.resetEvaluator();
            for (int i = 0; i < sim.transferIterations; i++) {
                if (idxSourceIterations < sim.iterations) {
                    // update the source model if it doesn't finish iterations
                    sim.loadData();
                    sim.run();
                    ((HomOTL)sim.transferScheme).setOldClassifier(sim.scheme);
                    idxSourceIterations++;
                }
                sim.localEvaluator = sim.resetEvaluator();
                System.out.println(sim.dateFormat.format(sim.transferCurDate));
                sim.loadData(sim.TARGET_DOMAIN, sim.TEST_TRAIN);
                result = sim.delayEvaluate(sim.transferScheme);
                sim.run(sim.TARGET_DOMAIN, sim.TEST_TRAIN);
                System.out.println(result.toString());
                if (i == sim.transferIterations - 1) {
                    // let OS to do GC
                    System.exit(0);
                }
            }
        } else {
            // disable transfer learning
            if (sim.blDelay) {
                for (int i = 0; i < sim.validationWindow; i++) {
                    sim.loadData();
                    sim.run();
                }
            }
            InspectionData result;
            for (int i = 0; i < sim.iterations; i++) {
                sim.localEvaluator = sim.resetEvaluator();
                System.out.println(sim.dateFormat.format(sim.curDate));
                sim.loadData();
                result = sim.delayEvaluate(sim.scheme);
                sim.run();
                System.out.println(result.toString());
                if (i == sim.iterations - 1) {
                    // let OS to do GC
                    System.exit(0);
                }
            }

        }
    }
}
