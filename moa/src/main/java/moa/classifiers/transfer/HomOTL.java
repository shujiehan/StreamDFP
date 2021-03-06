package moa.classifiers.transfer;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Homogeneous online transfer learning
 *
 * <p>Peilin Zhao et al. Online transfer learning.
 * In Artificial Intelligence 2014, pages 76-102. </p>
 *
 */

/**
 * Implemented by Shujie Han.
 */
public class HomOTL extends AbstractClassifier implements MultiClassClassifier {

    @Override
    public String getPurposeString() { return "Homogeneous online transfer learning."; }

    private static final long serialVersionUID = 1L;

    public IntOption tradeOffOption = new IntOption("tradeOff", 'c',
            "The trade-off parameter for classifiers.", 5, 0, Integer.MAX_VALUE);

    public FloatOption discountWeightOption = new FloatOption("discountWeight", 'w',
            "The discount weight parameter.", 0.5, 0, 1);

    public ClassOption newClassifierOption = new ClassOption("newClassifier", 'n',
            "The new classifier trained in target domain.", Classifier.class,
            "meta.OzaBagAdwin"); // classifier trained in target domain

    protected Classifier oldClassifier;
    protected Classifier newClassifier;
    protected double[] theta = {1, 1};


    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
        DoubleVector voteOldClassifier = new DoubleVector(this.oldClassifier.getVotesForInstance(inst));
        voteOldClassifier.scaleValues(theta[0]);
        DoubleVector voteNewClassifier = new DoubleVector(this.newClassifier.getVotesForInstance(inst));
        combinedVote.addValues(voteOldClassifier);
        combinedVote.addValues(voteNewClassifier);
        return combinedVote.getArrayRef();
    }

    @Override
    public void resetLearningImpl() {
        this.newClassifier = (Classifier) getPreparedClassOption(newClassifierOption);
        this.newClassifier.setRandomSeed(randomSeedOption.getValue());
        this.newClassifier.resetLearning();
        theta[0] = 1;
        theta[1] = 1;
    }

    public void setOldClassifier(Classifier classifier) {
        this.oldClassifier = classifier;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        // inst belongs to the target domain
        this.oldClassifier.trainOnInstance(inst);
        this.newClassifier.trainOnInstance(inst);
        if (! this.oldClassifier.correctlyClassifies(inst)) {
            theta[0] = theta[0] * this.discountWeightOption.getValue();
        }
    }

    @Override
    public void updateDownSampleRatio(double downSampleRatio) {
        ((AbstractClassifier)this.newClassifier).updateDownSampleRatio(downSampleRatio);
    }

    @Override
    public void setModelContext(InstancesHeader ih) {
        this.newClassifier.setModelContext(ih);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new LinkedList<>();
        measurementList.addAll(Arrays.asList(this.newClassifier.getModelMeasurements()));
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}
