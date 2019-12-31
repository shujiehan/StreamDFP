package simulate;

import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.MiscUtils;
import moa.core.ObjectRepository;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.BasicRegressionPerformanceEvaluator;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
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
            "valication window size.", 30);

    public FloatOption downSampleRatioOption = new FloatOption("downSampleRatio", 'D',
            "down sampling ratio", 5);

    public IntOption randomSeedOption = new IntOption("randomSeed", 'r', "random seed", 1);

    public FlagOption modelMeasurementOption = new FlagOption("modelMeasurements", 'm',
            "enable model measurements");
    public FlagOption regressionOption = new FlagOption("regressionTask", 'g', "regression task");

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

        Attribute serialNumber = trainingData.attribute("serial_number");
        List<String> attrSN = serialNumber.getAttributeValues();
        int idxSerialNumber = trainingData.indexOf(serialNumber);
        // record serializable serial numbers
        List<String> recordSN = new ArrayList<>();

        // compute imbalance ratio
        int cntNegatives = 0;
        int cntPositives = 0;
        Instance inst;
        for (int i = 0; i < trainingData.numInstances(); i++) {
            inst = trainingData.instance(i);
            if (blRegression) {
                if (inst.classValue() == 0) cntNegatives ++;
                else if (inst.classValue() > 0) cntPositives ++;
            } else {
                if (inst.classValue() == 0) cntNegatives ++;
                else if (inst.classValue() == 1) cntPositives ++;
            }
            recordSN.add(attrSN.get((int)inst.value(idxSerialNumber)));
        }

        // delete serial number in attributes
        trainingData.deleteAttributeAt(idxSerialNumber);

        double imbalanceRatio;
        if (cntPositives == 0) imbalanceRatio = 1000.0;
        else imbalanceRatio = cntNegatives * 1.0 / cntPositives;

        scheme.updateDownSampleRatio(this.downSampleRatio * imbalanceRatio);
        //System.out.println(cntNegatives + " " + cntPositives +" " + this.downSampleRatio * imbalanceRatio);

        for (int i = 0; i < trainingData.numInstances(); i++) {
            inst = trainingData.instance(i);
            inst.setSerialNumber(recordSN.get(i));
            switch (learnerOption.getChosenIndex()) {
                case AbstractPredict.ENSEMBLE:
                    scheme.trainOnInstance(inst);
                    break;
                case AbstractPredict.SINGLE_TREE:
                    scheme.trainOnInstance(inst);
                    break;
                case AbstractPredict.SINGLE_TREE_DOWN_SAMPLING:
                    int k;
                    if (inst.classValue() == 0) {
                        k = MiscUtils.poisson(1 / scheme.getDownSampleRatio(),
                                samplingRandom);
                    } else {
                        k = MiscUtils.poisson(this.lambdaOption.getValue(), samplingRandom);
                    }
                    if (k > 0) {
                        Instance weightedInst = (Instance) inst.copy();
                        weightedInst.setWeight(inst.weight() * k);
                        scheme.trainOnInstance(weightedInst);
                    }
                    break;
            }
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
        return evaluator;
    }

    public void loadData() {
        curDate = new Date(curDate.getTime() + 1000*24*60*60);
        this.fileName = this.dateFormat.format(curDate) + ".arff";
        testData = load(testPath + fileName, cindex);
        trainingData = load(trainPath + fileName, cindex);
    }

    public InspectionData delayEvaluate() {
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
        InspectionData result = inspect(trainingData, testData, globalEvaluator, localEvaluator, scheme, blDelay,
                validationWindow);
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
        if (sim.blDelay) {
            for (int i = 0; i < sim.validationWindow; i++) {
                sim.loadData();
                //System.out.println(sim.dateFormat.format(sim.curDate));
                sim.run();
            }
        }
        InspectionData result;
        for (int i = 0; i < sim.iterations; i++) {
            sim.localEvaluator = sim.resetEvaluator();
            System.out.println(sim.dateFormat.format(sim.curDate));
            sim.loadData();
            result = sim.delayEvaluate();
            sim.run();
            System.out.println(result.toString());
            if (i == sim.iterations - 1) {
                // let OS to do GC
                System.exit(0);
            }
        }
    }
}
