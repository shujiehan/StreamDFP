/*
 *    BasicClassificationPerformanceEvaluator.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.evaluation;

import com.github.javacliparser.FlagOption;
import moa.AbstractMOAObject;
import moa.core.*;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceData;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Classification evaluator that performs basic incremental evaluation.
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * 
 * Updates in September 15th 2017 to include precision, recall and F1 scores.
 * @author Jean Karax (karaxjr@gmail.com)
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @author Wilson Sasaki Jr (sasaki.wilson.jr@gmail.com)
 * @version $Revision: 8 $
 */
public class BasicClassificationPerformanceEvaluator extends AbstractOptionHandler
        implements ClassificationPerformanceEvaluator {

    private static final long serialVersionUID = 1L;

    protected Estimator weightCorrect;

    protected Estimator[] columnKappa;

    protected Estimator[] rowKappa;

    protected Estimator[] precision;

    protected Estimator[] recall;

    public int numClasses;

    private Estimator weightCorrectNoChangeClassifier;

    private Estimator weightMajorityClassifier;

    private int lastSeenClass;

    private double totalWeightObserved;

    public FlagOption precisionRecallOutputOption = new FlagOption("precisionRecallOutput",
            'o',
            "Outputs average precision, recall and F1 scores.");
    
    public FlagOption precisionPerClassOption = new FlagOption("precisionPerClass",
            'p',
            "Report precision per class.");

    public FlagOption recallPerClassOption = new FlagOption("recallPerClass",
            'r',
            "Report recall per class.");

    public FlagOption f1PerClassOption = new FlagOption("f1PerClass", 'f',
            "Report F1 per class.");

    public FlagOption confusionMatrixOption = new FlagOption("confusionMatrix", 'm',
            "Confusion Matrix.");

    public FlagOption falseAlarmOption = new FlagOption("falseAlarmRate", 'a', "False Alarm Rate");

    public double totalNumDaysBeforeFailure = 0;

    public double threshold = 0.5;

    @Override
    public void reset() {
        reset(this.numClasses);
    }

    public void reset(int numClasses) {
        this.numClasses = numClasses;
        this.rowKappa = new Estimator[numClasses];
        this.columnKappa = new Estimator[numClasses];
        this.precision = new Estimator[numClasses];
        this.recall = new Estimator[numClasses];
        for (int i = 0; i < this.numClasses; i++) {
            this.rowKappa[i] = newEstimator();
            this.columnKappa[i] = newEstimator();
            this.precision[i] = newEstimator();
            this.recall[i] = newEstimator();
        }
        this.weightCorrect = newEstimator();
        this.weightCorrectNoChangeClassifier = newEstimator();
        this.weightMajorityClassifier = newEstimator();
        this.lastSeenClass = 0;
        this.totalWeightObserved = 0;
        this.totalNumDaysBeforeFailure = 0;
    }

    public void setThreshold(double inputThreshold) {
        this.threshold = inputThreshold;
    }

    @Override
    public void addResult(Example<Instance> example, double[] classVotes) {
        Instance inst = example.getData();
        double weight = inst.weight();
        if (inst.classIsMissing() == false) {
            int trueClass = (int) inst.classValue();
            int predictedClass;
            if (threshold == 0.5) {
                predictedClass = Utils.maxIndex(classVotes);
            } else {
                predictedClass = Utils.maxIndex(classVotes, threshold);
            }
            if (weight > 0.0) {
                if (this.totalWeightObserved == 0) {
                    if (inst.dataset().numClasses() == 1) {
                        System.out.println("num classes = 1");
                        reset(2);
                    }
                    reset(inst.dataset().numClasses());
                }
                this.totalWeightObserved += weight;
                this.weightCorrect.add(predictedClass == trueClass ? weight : 0);
                for (int i = 0; i < this.numClasses; i++) {
                    this.rowKappa[i].add(predictedClass == i ? weight : 0);
                    this.columnKappa[i].add(trueClass == i ? weight : 0);
                    // for both precision and recall, NaN values are used to 'balance' the number
                    // of instances seen across classes
                    if (predictedClass == i) {
                        precision[i].add(predictedClass == trueClass ? weight : 0.0);
                    } else precision[i].add(Double.NaN);
                    if (trueClass == i) {
                        recall[i].add(predictedClass == trueClass ? weight : 0.0);
                    } else recall[i].add(Double.NaN);
                }
            }
            this.weightCorrectNoChangeClassifier.add(this.lastSeenClass == trueClass ? weight : 0);
            this.weightMajorityClassifier.add(getMajorityClass() == trueClass ? weight : 0);
            this.lastSeenClass = trueClass;
        }
    }

    // Shujie Add:
    public void addResultDelay(List<Instance> instances) {
        boolean blFailed = false;
        int idxLastFailed = 0;
        int listSize = instances.size();
        for (int i = 0; i < listSize; i++) {
            Instance inst = instances.get(i);
            double[] votes = inst.getPredictedVotes();
            double trueClass = inst.classValue();
            if (trueClass == 1) {
                blFailed = true;
                idxLastFailed = i;
            }
            // handle corner case: a disk first failed, then restart to use in several days
            if (blFailed && idxLastFailed < i) {
                Instance lastFailedInst = instances.get(idxLastFailed);
                double[] lastFailedVotes = lastFailedInst.getPredictedVotes();
                this.totalNumDaysBeforeFailure += 0;
                addResult((Example<Instance>) new InstanceExample(lastFailedInst), lastFailedVotes);
                return;
            }
            int predictedClass;
            if (threshold == 0.5) {
                predictedClass = Utils.maxIndex(votes);
            } else {
                predictedClass = Utils.maxIndex(votes, threshold);
            }
            if (predictedClass == (int)trueClass) {
                if ((int)trueClass == 1) {
                    this.totalNumDaysBeforeFailure += (listSize - i);
                }
                addResult((Example<Instance>) new InstanceExample(inst), votes);
                return;
            }
        }
        Instance inst = instances.get(instances.size() - 1);
        double[] votes = inst.getPredictedVotes();
        addResult((Example<Instance>) new InstanceExample(inst), votes);
    }

    private int getMajorityClass() {
        int majorityClass = 0;
        double maxProbClass = 0.0;
        for (int i = 0; i < this.numClasses; i++) {
            if (this.columnKappa[i].estimation() > maxProbClass) {
                majorityClass = i;
                maxProbClass = this.columnKappa[i].estimation();
            }
        }
        return majorityClass;
    }

    @Override
    public Measurement[] getPerformanceMeasurements() {
        ArrayList<Measurement> measurements = new ArrayList<Measurement>();
        measurements.add(new Measurement("classified instances", this.getTotalWeightObserved()));
        measurements.add(new Measurement("classifications correct (percent)", this.getFractionCorrectlyClassified() * 100.0));
        measurements.add(new Measurement("Kappa Statistic (percent)", this.getKappaStatistic() * 100.0));
        measurements.add(new Measurement("Kappa Temporal Statistic (percent)", this.getKappaTemporalStatistic() * 100.0));
        measurements.add(new Measurement("Kappa M Statistic (percent)", this.getKappaMStatistic() * 100.0));

        if (confusionMatrixOption.isSet()) {
            double[] confusionMatrix = getConfusionMatrix();
            measurements.add(new Measurement("TP", confusionMatrix[0]));
            measurements.add(new Measurement("FP", confusionMatrix[1]));
            measurements.add(new Measurement("TN", confusionMatrix[2]));
            measurements.add(new Measurement("FN", confusionMatrix[3]));
            if (falseAlarmOption.isSet()) {
                double cntNegs = confusionMatrix[1] + confusionMatrix[2];
                double falseAlarmRate = 0;
                if (cntNegs == 0) {
                    falseAlarmRate = Float.NaN;
                } else {
                    falseAlarmRate = 100* confusionMatrix[1] / cntNegs;
                }
                measurements.add(new Measurement("False Alarm Rate (percent)", falseAlarmRate));
                measurements.add(new Measurement("Average Days before Failure",
                        this.totalNumDaysBeforeFailure/confusionMatrix[0]));
            }
        }
        if (precisionRecallOutputOption.isSet()) 
            measurements.add(new Measurement("F1 Score (percent)", 
                    this.getF1Statistic() * 100.0));
        if (f1PerClassOption.isSet()) {
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("F1 Score for class " + i + 
                        " (percent)", 100.0 * this.getF1Statistic(i)));
            }
        }
        if (precisionRecallOutputOption.isSet())
            measurements.add(new Measurement("Precision (percent)", 
                this.getPrecisionStatistic() * 100.0));               
        if (precisionPerClassOption.isSet()) {
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Precision for class " + i + 
                        " (percent)", 100.0 * this.getPrecisionStatistic(i)));
            }
        }
        if (precisionRecallOutputOption.isSet())
            measurements.add(new Measurement("Recall (percent)", 
                this.getRecallStatistic() * 100.0));
        if (recallPerClassOption.isSet()) {
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Recall for class " + i + 
                        " (percent)", 100.0 * this.getRecallStatistic(i)));
            }
        }

        Measurement[] result = new Measurement[measurements.size()];

        return measurements.toArray(result);

    }

    public double getTotalWeightObserved() {
        return this.totalWeightObserved;
    }

    public double getFractionCorrectlyClassified() {
        return this.weightCorrect.estimation();
    }

    public double getFractionIncorrectlyClassified() {
        return 1.0 - getFractionCorrectlyClassified();
    }

    public double getKappaStatistic() {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getFractionCorrectlyClassified();
            double pc = 0.0;
            for (int i = 0; i < this.numClasses; i++) {
                pc += this.rowKappa[i].estimation()
                        * this.columnKappa[i].estimation();
            }
            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }

    public double getKappaTemporalStatistic() {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getFractionCorrectlyClassified();
            double pc = this.weightCorrectNoChangeClassifier.estimation();

            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }

    private double getKappaMStatistic() {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getFractionCorrectlyClassified();
            double pc = this.weightMajorityClassifier.estimation();

            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }

    public double getPrecisionStatistic() {
        double total = 0;
        for (Estimator ck : this.precision) {
            total += ck.estimation();
        }
        return total / this.precision.length;
    }

    public double getPrecisionStatistic(int numClass) {
        return this.precision[numClass].estimation();
    }

    public double getRecallStatistic() {
        double total = 0;
        for (Estimator ck : this.recall) {
            total += ck.estimation();
        }
        return total / this.recall.length;
    }

    public double getRecallStatistic(int numClass) {
        return this.recall[numClass].estimation();
    }

    public double getF1Statistic() {
        return 2 * ((this.getPrecisionStatistic() * this.getRecallStatistic())
                / (this.getPrecisionStatistic() + this.getRecallStatistic()));
    }

    public double getF1Statistic(int numClass) {
        return 2 * ((this.getPrecisionStatistic(numClass) * this.getRecallStatistic(numClass))
                / (this.getPrecisionStatistic(numClass) + this.getRecallStatistic(numClass)));
    }

    //Shujie Add:
    public double[] getConfusionMatrix() {
        // [tp, fp, tn, fn]
        double[] confusionMatrix = new double[4];
        if (this.precision.length > 1) {
            confusionMatrix[0] = ((BasicEstimator)this.precision[1]).sum;
        }
        confusionMatrix[2] = ((BasicEstimator)this.recall[0]).sum; // tn
        confusionMatrix[1] = ((BasicEstimator)this.recall[0]).len - confusionMatrix[2];
        confusionMatrix[3] = ((BasicEstimator)this.precision[0]).len - confusionMatrix[2];
        return confusionMatrix;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        Measurement.getMeasurementsDescription(getPerformanceMeasurements(),
                sb, indent);
    }

    @Override
    public void addResult(Example<Instance> testInst, Prediction prediction) {
        // TODO Auto-generated method stub

    }

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

    }

    public interface Estimator extends Serializable {

        void add(double value);

        double estimation();
    }

    public class BasicEstimator implements Estimator {

        protected double len;

        protected double sum;

        @Override
        public void add(double value) {
            if(!Double.isNaN(value)) {
                sum += value;
                len++;
            }
        }

        @Override
        public double estimation() {
            return sum / len;
        }

    }

    protected Estimator newEstimator() {
        return new BasicEstimator();
    }
}
