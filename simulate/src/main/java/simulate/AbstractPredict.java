package simulate;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.Range;
import moa.classifiers.AbstractClassifier;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.options.AbstractOptionHandler;

import java.io.*;
import java.util.*;

public abstract class AbstractPredict extends AbstractOptionHandler {

    public static class InspectionData {

        /** the inspection point. */
        public int index = -1;

        /** the votes. */
        public double[] votes = new double[0];

        /** the global measurements. */
        public Measurement[] globalMeasurements = new Measurement[0];

        /** the local measurements. */
        public Measurement[] localMeasurements = new Measurement[0];

        /** the model measurements. */
        public Measurement[] modelMeasurements = new Measurement[0];

        /**
         * Returns the container data as string.
         *
         * @return		the string representation
         */
        @Override
        public String toString() {
            StringBuilder	result;
            int		i;

            result = new StringBuilder();

            result.append("Index " + index + "\n");

            result.append("Votes\n");
            for (i = 0; i < votes.length; i++)
                result.append("  " + i + ": " + doubleToString(votes[i], 8) + "\n");

            result.append("Global Measurements\n");
            for (Measurement m: globalMeasurements)
                result.append("  " + m.getName() + ": " + doubleToString(m.getValue(), 8) + "\n");

            result.append("Local Measurements\n");
            for (Measurement m: localMeasurements)
                result.append("  " + m.getName() + ": " + doubleToString(m.getValue(), 8) + "\n");

            result.append("Model measurements\n");
            for (Measurement m: modelMeasurements) {
                if (m.getName().indexOf("serialized") > -1)
                    continue;
                result.append("  " + m.getName() + ": " + doubleToString(m.getValue(), 8) + "\n");
            }

            return result.toString();
        }
    }

    public static String doubleToString(double value, int afterDecimalPoint) {
        StringBuilder 	builder;
        double 		temp;
        int 		dotPosition;
        int 		currentPos;
        long 		precisionValue;
        char		separator;

        temp = value * Math.pow(10.0, afterDecimalPoint);
        if (Math.abs(temp) < Long.MAX_VALUE) {
            precisionValue = 	(temp > 0) ? (long)(temp + 0.5)
                    : -(long)(Math.abs(temp) + 0.5);
            if (precisionValue == 0)
                builder = new StringBuilder(String.valueOf(0));
            else
                builder = new StringBuilder(String.valueOf(precisionValue));

            if (afterDecimalPoint == 0)
                return builder.toString();

            separator   = '.';
            dotPosition = builder.length() - afterDecimalPoint;
            while (((precisionValue < 0) && (dotPosition < 1)) || (dotPosition < 0)) {
                if (precisionValue < 0)
                    builder.insert(1, '0');
                else
                    builder.insert(0, '0');
                dotPosition++;
            }

            builder.insert(dotPosition, separator);

            if ((precisionValue < 0) && (builder.charAt(1) == separator))
                builder.insert(1, '0');
            else if (builder.charAt(0) == separator)
                builder.insert(0, '0');

            currentPos = builder.length() - 1;
            while ((currentPos > dotPosition) && (builder.charAt(currentPos) == '0'))
                builder.setCharAt(currentPos--, ' ');

            if (builder.charAt(currentPos) == separator)
                builder.setCharAt(currentPos, ' ');

            return builder.toString().trim();
        }
        return new String("" + value);
    }

    protected static final int ENSEMBLE = 0;
    protected static final int SINGLE_TREE = 1;
    protected static final int SINGLE_TREE_DOWN_SAMPLING = 2;
    // For transfer learning
    protected static final String SOURCE_DOMAIN = "source";
    protected static final String TARGET_DOMAIN = "target";
    protected static final String TEST = "test";
    protected static final String TRAIN = "train";
    protected static final String TEST_TRAIN = "test and train";

    // key: serial number; values: instances of the disk
    protected HashMap<String, List<Instance>> keepDelayInstances;
    public int cindex;
    protected Random samplingRandom;
    protected int randomSeed = 1;
    public boolean blRegression;
    public boolean blTransfer;

    public int validationWindow;
    public int labelWindow;

    public MultiChoiceOption learnerOption = new MultiChoiceOption("learner", 'l',
            "Ensembles or single tree?", new String[]{"ensemble", "single tree", "single tree with downsampling"},
            new String[]{"ensemble", "singleTree", "singleTreeDownSampling"}, 0);

    public FloatOption lambdaOption = new FloatOption("lambda", 'b', "lambda for oversampling positive class in single tree",
            6);

    public void cleanKeepDelayInstances() {
        keepDelayInstances.clear();
    }

    protected void keep(Instance inst, int queueSize) {
        String sn = inst.getSerialNumber();
        List<Instance> insts = keepDelayInstances.get(sn);
        if (insts == null) insts = new LinkedList<Instance>();
        insts.add(inst);
        keepDelayInstances.put(sn, insts);
        assert (insts.size() <= queueSize);
    }

    protected Instances load(String filename, int classIndex) {
        Instances	result = null;

        try {
            File tmp=new File(filename);
            FileInputStream fileStream = new FileInputStream(tmp.getAbsolutePath());
            Reader reader=new BufferedReader(new InputStreamReader(fileStream));
            Range range = new Range("-1");
            result = new Instances(reader,range);
            result.setClassIndex(classIndex);
            while (result.readInstance(null));
        }
        catch (Exception e) {
            System.err.println("Failed to load dataset: " + filename);
            e.printStackTrace();
            result = null;
        }

        return result;
    }

    public InspectionData evaluateDelay(AbstractClassifier scheme, LearningPerformanceEvaluator globalEvaluator,
                                        LearningPerformanceEvaluator localEvaluator,
                                        int index, boolean blModel) {
        InspectionData result = new InspectionData();
        result.index = index;
        result.globalMeasurements = globalEvaluator.getPerformanceMeasurements();
        result.localMeasurements = localEvaluator.getPerformanceMeasurements();
        if (blModel)
            result.modelMeasurements = scheme.getModelMeasurements();
        return result;
    }

    public void backtracking(HashSet<String> failedSN) {
        Iterator it = keepDelayInstances.entrySet().iterator();
        while (it.hasNext()) {
            HashMap.Entry pair = (HashMap.Entry) it.next();
            String sn = (String)pair.getKey();
            if (failedSN.contains(sn)) {
                List<Instance> instances = (List<Instance>) pair.getValue();
                if (blRegression) {
                   int listSize = instances.size();
                   ListIterator reverseIt = instances.listIterator(listSize);
                   int idx=1;
                   while (reverseIt.hasPrevious()) {
                       Instance inst = (Instance)reverseIt.previous();
                       if (idx <= labelWindow) {
                           inst.setClassValue(1.0 - 1.0 / (labelWindow + 1.0) * idx);
                           //inst.setClassValue(1.0 / (1.0 + idx));
                       } else {
                           break;
                       }
                       assert(inst.classValue() >= 0 && inst.classValue() <= 1);
                       idx ++;
                   }
                } else {
                    for (Instance inst : instances) {
                        inst.setClassValue(1);
                        assert(inst.classValue() >= 0 && inst.classValue() <= 1);
                    }
                }
                keepDelayInstances.put(sn, instances);
            }
        }
    }

    public InspectionData inspect(Instances testData,
                                  LearningPerformanceEvaluator globalEvaluator,
                                  LearningPerformanceEvaluator localEvaluator, AbstractClassifier scheme,
                                  Boolean blDelay, int validationWindow) {
        InspectionData result = new InspectionData();
        Instance inst;
        double[] votes;
        // test
        // serial number
        Attribute serialNumber = testData.attribute("serial_number");
        List<String> attrSN = serialNumber.getAttributeValues();
        int idxSerialNumber = testData.indexOf(serialNumber);
        // record serializable serial numbers
        List<String> recordSN = new ArrayList<>();
        // serial numbers of failed disks for backtracking
        HashSet<String> failedSN = new HashSet<>();

        for (int i = 0; i < testData.numInstances(); i++) {
            inst = testData.instance(i);
            String sn = attrSN.get((int)inst.value(idxSerialNumber));
            recordSN.add(sn);
            if (inst.classValue() == 1) {
                failedSN.add(sn);
            }
        }
        testData.deleteAttributeAt(idxSerialNumber);

        // backtracking
        if (blDelay) {
            backtracking(failedSN);
        }

        for (int i = 0; i < testData.numInstances(); i++) {
            inst = testData.instance(i);
            inst.setSerialNumber(recordSN.get(i));
            votes = scheme.getVotesForInstance(inst);
            assert(inst.classValue() >= 0 && inst.classValue() <= 1);
            if (blDelay) {
                inst.keepPredictedVotes(votes);
                keep(inst, validationWindow);
            } else {
                globalEvaluator.addResult((Example<Instance>) new InstanceExample(inst), votes);
                localEvaluator.addResult((Example<Instance>) new InstanceExample(inst), votes);
            }
        }
        return result;
    }

    public void updateModel(Instances trainingData, AbstractClassifier scheme, double downSampleRatio) {
        // train
        // compute imbalance ratio
        Instance inst;
        int cntNegatives = 0;
        int cntPositives = 0;
        Attribute serialNumber = trainingData.attribute("serial_number");
        List<String> attrSN = serialNumber.getAttributeValues();
        int idxSerialNumber = trainingData.indexOf(serialNumber);
        // record serializable serial numbers
        List<String> recordSN = new ArrayList<>();
        for (int i = 0; i < trainingData.numInstances(); i++) {
            inst = trainingData.instance(i);
            assert(inst.classValue() >= 0 && inst.classValue() <= 1);
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

        scheme.updateDownSampleRatio(downSampleRatio * imbalanceRatio);
        for (int i = 0; i < trainingData.numInstances(); i++) {
            inst = trainingData.instance(i);
            inst.setSerialNumber(recordSN.get(i));
            if (inst.classValue() > 1) {
                System.out.println("training class value > 1, " + inst.getSerialNumber());
            }
            switch (learnerOption.getChosenIndex()) {
                case AbstractPredict.ENSEMBLE:
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
}
