/*...*/
package moa.classifiers.meta;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.jidesoft.plaf.eclipse.EclipseMenuItemUI;
import com.yahoo.labs.samoa.instances.Instance;
import moa.AbstractMOAObject;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Recurrent Neural Networks
 * @Author Shujie Han
 */
public class RNN extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 2L;

    @Override
    public String getPurposeString() {return "RNN: Recurrent Neural Networks"; }

    public FloatOption learningRateOption = new FloatOption("learningRate", 'r', "Learning rate", 0.1);
    //public IntOption numLayerOption = new IntOption("numLayer", 'l', "Number of layers", 2);
    public IntOption numUnitOption = new IntOption("numUnit", 'u', "Number of units per hidden layer", 3);
    public FloatOption lambdaNegativeOption = new FloatOption("lambdaNegatives", 'n', "The lambda parameter of downsampling for negative samples", 1.0, 1.0, Float.MAX_VALUE);
    public FloatOption lambdaPositiveOption = new FloatOption("lambdaPositives", 'p', "The lambda parameter of downsampling for positive samples", 1.0, 1.0, Float.MAX_VALUE);
    public IntOption numSamplesResetOption = new IntOption("numSampleReset", 's', "Reset RNN periodically", 10000, 100, Integer.MAX_VALUE);
    public FloatOption clippingGradientOption = new FloatOption("clippingGradient", 'c', "To clip the gradients", 5.0);
    public FlagOption clipOption = new FlagOption("blClipping", 'o', "Whether clipping the gradients");

    protected boolean reset;
    protected double learningRate;
    protected int numLayers = 2; // We only consider one hidden layer and one output layer here
    protected int numUnits; // numUnits per hidden layer
    protected List<Layer> layers; // store each layer
    protected long instancesSeen;
    protected int numSamplesReset;
    protected boolean blClip;
    protected double clippingGradient;

    @Override
    public void resetLearningImpl() {
        this.reset = true;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.reset) {
            this.reset = false;
            learningRate = learningRateOption.getValue();
            numUnits = numUnitOption.getValue();
            layers = new ArrayList<Layer>();
            numSamplesReset = numSamplesResetOption.getValue();
            blClip = clipOption.isSet();
            if (blClip) {
                clippingGradient = clippingGradientOption.getValue();
            }

            int numAttributes = inst.numAttributes() - 1; // exclude the class label
            // init all layers
            for (int idxLayer = 0; idxLayer < numLayers; idxLayer++) {
                if (idxLayer == numLayers - 1) {// output layer
                    layers.add(new Layer(numUnits, inst.numClasses(), Layer.OUTPUT, this.classifierRandom));
                } else {// hidden layers
                    int numPreUnits;
                    // determine the number of units in the previous layer
                    if (idxLayer == 0) {// the first hidden layer
                        numPreUnits = numAttributes;
                    } else {
                        numPreUnits = numUnits;
                    }
                    layers.add(new Layer(numPreUnits, numUnits, Layer.HIDDEN, this.classifierRandom));
                }
            }
        }
        instancesSeen++;

        // downsampling
        int k;
        if (inst.classValue() == 0) {
            k = MiscUtils.poisson(this.lambdaNegativeOption.getValue() / this.downSampleRatio,
                    this.classifierRandom);
        } else {
            k = MiscUtils.poisson(this.lambdaPositiveOption.getValue(), this.classifierRandom);
        }
        if (k > 0) {
            Instance weightedInst = (Instance) inst.copy();
            weightedInst.setWeight(inst.weight() * k);
            double[] forwardInput = {0};
            double[] recurrentInput = {0};
            for (int i = 0; i < numLayers; i ++) {
                if (i == 0) {
                    // from the input layer to the first hidden layer
                    forwardInput = layers.get(i).forwardPropagate(weightedInst);
                } else {
                    // from the hidden layer to output layer
                    forwardInput = layers.get(i).forwardPropagate(forwardInput);
                }
            }

            // propagate the errors backward
            double[] gradientBias = {0};
            for (int i = numLayers - 1; i >= 0 ; i--) {
                if (i == numLayers - 1) {
                    // compute the gradient of bias and weights for the output layer
                    // delta(b_y)
                    gradientBias = layers.get(i).computeGradientsForOutputLayer(weightedInst, layers.get(i-1).getRecurrentOutputs());
                } else {
                    // from the output layer to the last hidden layer
                    // delta(b_a)
                    gradientBias = layers.get(i).computeGradientsForHiddenLayer(gradientBias, layers.get(i + 1), weightedInst);
                    layers.get(i).computeGradientsForRecurrentLayer(gradientBias);
                }
            }

            // update all inWeights and bias;
            for (int i = 0; i < numLayers; i++) {
                if (i == 0) {
                    layers.get(i).updateParameters(learningRate);
                    layers.get(i).updateParametersForRecurrentLayer(learningRate);
                } else {
                    layers.get(i).updateParameters(learningRate);
                }
            }
        }
        if (instancesSeen % numSamplesReset == 0) {
            this.reset = true;
        }
    }

    public class Layer extends AbstractMOAObject {
        private static final long serialVersionUID = 1L;

        protected List<Neurode> units;
        protected List<RecurrentNeurode> recurrentUnits; // recurrent units
        protected int numUnits; // number of units in this layer
        protected int numPreUnits; // number of units in the previous layer
        protected String layerType;
        public static final String HIDDEN = "Hidden";
        public static final String OUTPUT = "Output";

        public Layer(int numPreUnits, int numUnits, String layerType, Random randomGenerator) {
            this.numPreUnits = numPreUnits;
            this.numUnits = numUnits;
            this.layerType = layerType;
            units = new ArrayList<>();
            for (int i = 0; i < numUnits; i++) {
                units.add(new Neurode(numPreUnits, randomGenerator));
            }
            if (layerType.equals(HIDDEN)) {
                recurrentUnits = new ArrayList<>();
                for (int i = 0; i < numUnits; i++) {
                    recurrentUnits.add(new RecurrentNeurode(numUnits, randomGenerator));
                }
            }
        }

        public double[] forwardPropagate(Instance inst) {
            // forward propagate for the first hidden layer
            double[] inputToNextLayer = new double[numUnits];
            for (int i = 0; i < numUnits; i++) {
                double output = units.get(i).computeOutput(inst); // X_t * W_xh + b_h
                // combine the weights of recurrent layers
                double recurrentOutput = 0; // H_t-1 * W_hh
                for (int j = 0; j < numUnits; j++) {
                    recurrentOutput += recurrentUnits.get(j).getPrevOutput() * recurrentUnits.get(j).getWeight(i);
                }
                inputToNextLayer[i] = Math.tanh(output + recurrentOutput);
                recurrentUnits.get(i).storeOutput(inputToNextLayer[i]);
            }
            return inputToNextLayer;
        }

        public double[] forwardPropagate(double[] input) {
            // forward propagate from the hidden layer to the output layer
            double[] inputToNextLayer = new double[numUnits];
            for (int i = 0; i < numUnits; i++) {
                double output = units.get(i).computeOutput(input);
                inputToNextLayer[i] = output;
            }
            return inputToNextLayer;
        }

        public double[] computeGradientsForOutputLayer(Instance inst, double[] recurrentOutputs) {
            double[] gradientBias = new double[numUnits]; // length = 2
            for (int i = 0; i < numUnits; i++) {
                 gradientBias[i] = units.get(i).computeGradientsForOutputUnits(inst, recurrentOutputs);
            }
            return gradientBias;
        }

        public void computeGradientsForRecurrentLayer(double[] gradientBiasInHiddenLayer) {
            for (int i = 0; i < numUnits; i++) {
                recurrentUnits.get(i).computeGradientsForRecurrentUnits(gradientBiasInHiddenLayer);
            }
        }

        public double[] computeGradientsForHiddenLayer(double[] gradientBiasInOutputLayer, Layer outputLayer, Instance inst) {
            // compute gradient bias and weights for hidden layer:
            double[] gradientBias = new double[numUnits]; // length = 3
            for (int i = 0; i < numUnits; i++) {
                double[] inWeightsFromThisUnit = new double[outputLayer.numUnits]; // length = 2
                for (int j = 0; j < outputLayer.numUnits; j++) {
                    inWeightsFromThisUnit[j] = outputLayer.getInWeightByIndex(i, j);
                }
                gradientBias[i] = units.get(i).computeGradientsForHiddenUnits(
                        gradientBiasInOutputLayer, inWeightsFromThisUnit, recurrentUnits.get(i).getOutput(), inst, recurrentUnits.get(i).getGradientNext());
            }
            return gradientBias;
        }

        public double getInWeightByIndex(int fromUnit, int thisUnit) {
            return units.get(thisUnit).getInWeight(fromUnit);
        }

        public double getUnitOutput(int idxUnit) {
            return units.get(idxUnit).getOutput();
        }

        public double[] getOutput() {
            double[] output = new double[numUnits];
            for (int i = 0; i < numUnits; i++) {
                output[i] = units.get(i).getOutput();
            }
            return output;
        }

        public double[] getRecurrentOutputs() {
           double[] output = new double[numUnits];
           for (int i = 0; i < numUnits; i++) {
               output[i] = recurrentUnits.get(i).getOutput();
           }
           return output;
        }

        public void updateParameters(double learningRate) {
            // update parameters for the hidden layer
            for (int i = 0; i < numUnits; i++) {
                units.get(i).updateBias(learningRate);
                units.get(i).updateInWeights(learningRate);
            }
        }

        public void updateParametersForRecurrentLayer(double learningRate) {
            // update parameters for the recurrent layer
            for (int i = 0; i < numUnits; i++) {
                recurrentUnits.get(i).updateOutWeights(learningRate);
            }
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {}
    }

    public class Neurode extends AbstractMOAObject {
        /** The units in the hidden layers and output layer. */
        private static final long serialVersionUID = 1L;
        // For the hidden layers and output layer, the weights are from all neurodes in the previous layer to this neurode
        protected DoubleVector inWeights; // W_ax for (input layer -> hidden layer) or W_ya for (hidden layer -> output layer)
        protected double bias; // b_a or b_y
        protected double output; // forward output: W_ax * X + b_a or sigmoid(W_ya * a + b_y)
        protected double gradientBias; // backward gradients: delta(b_a) or delta(b_y)
        protected DoubleVector gradientWeights; // backward gradients of weights: delta(W_ax) or delta(W_ya)

        public Neurode(int numPreUnits, Random randomGen) {
            this.inWeights = new DoubleVector();
            this.gradientWeights = new DoubleVector();
            for (int i = 0; i < numPreUnits; i++) {
                this.inWeights.setValue(i, 0.2 * randomGen.nextDouble() - 0.1);
                gradientWeights.setValue(i, 0);
            }
            bias = 0.2 * randomGen.nextDouble() - 0.1;
            output = 0;
            gradientBias = 0;
        }

        // compute the output of forward propagation for the first hidden layer
        public double computeOutput(Instance inst) {
            output = 0;
            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                int idxAtt = modelAttIndexToInstanceAttIndex(i, inst);
                output += inst.value(idxAtt) * inWeights.getValue(i);
            }
            output += bias;
            //output = 1.0 / (1.0 + Math.exp(-output)); // sigmoid function
            return output;
        }

        // compute the output of forward propagation from the hidden layer to the output layer
        public double computeOutput(double[] input) {
            output = 0;
            for (int i = 0; i < input.length; i++) {
                output += input[i] * inWeights.getValue(i);
            }
            output += bias;
            output = 1.0 / (1.0 + Math.exp(-output)); // sigmoid function
            return output;
        }

        // compute the error for backward propagation for the output layer
        public double computeGradientsForOutputUnits(Instance inst, double[] recurrentOutputs) {
            gradientBias = (output - inst.classValue());
            for (int i = 0; i < recurrentOutputs.length; i++) {
                gradientWeights.addToValue(i, gradientBias * recurrentOutputs[i]);
            }
            return gradientBias;
        }

        public double computeGradientsForHiddenUnits(double[] gradientInOutputLayer,
                                                     double[] inWeightsFromThisUnit,
                                                     double recurrentOutput,
                                                     Instance inst,
                                                     double gradientNext) {
            // delta(b_a) = ((\hat{y} - y) * W_ya + delta(a_t+1)) * (1 - (a_t)^2)
            //            = (gradient bias in output) * (inWeights in output layer) * (1 - output^2 in recurrent layer)
            double errorSum = 0;
            for (int i = 0; i < gradientInOutputLayer.length; i++) {
               errorSum += gradientInOutputLayer[i] * inWeightsFromThisUnit[i];
            }
            errorSum += gradientNext;
            errorSum *= (1 - recurrentOutput * recurrentOutput);
            gradientBias = errorSum;
            // compute gradient weights
            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                int idxAtt = modelAttIndexToInstanceAttIndex(i, inst);
                gradientWeights.addToValue(i, gradientBias * inst.value(idxAtt));
            }
            return gradientBias;
        }

        public double getInWeight(int index) {
            return inWeights.getValue(index);
        }

        public double getError() {
           return gradientBias;
        }

        public double getOutput() {
            return output;
        }

        public void updateInWeights(double learningRate) {
            if (blClip) {
               for (int i = 0; i < gradientWeights.numValues(); i++) {
                   if (gradientWeights.getValue(i) > clippingGradient) {
                       gradientWeights.setValue(i, clippingGradient);
                   } else if (gradientWeights.getValue(i) < -1 * clippingGradient) {
                       gradientWeights.setValue(i, -1 * clippingGradient);
                   }
               }
            }
            gradientWeights.scaleValues(-1 * learningRate);
            inWeights.addValues(gradientWeights);
        }

        public void updateBias(double learningRate) {
            if (blClip) {
                if (gradientBias > clippingGradient) {
                    gradientBias = clippingGradient;
                } else if (gradientBias < -1 * clippingGradient) {
                    gradientBias = -1 * clippingGradient;
                }
            }
            bias += -1 * learningRate * gradientBias;
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {}
    }

    public class RecurrentNeurode extends AbstractMOAObject {
        // The units in the recurrent layers
        private static final long serialVersionUID = 1L;
        // For the recurrent layers, the weights are from this neurode in the recurrent layer (t-1) to all neurodes in the hidden layer (t)
        protected DoubleVector outWeights; // W_aa
        protected double prevOutput; // previous output, a_(t-1)
        protected double output; // forward output, tanh(W_ax * X_t + W_aa * a_(t-1) + b_a)
        protected DoubleVector gradientWeights; // backward error, W_aa
        protected double gradientNext; // delta(a_t+1)

        public RecurrentNeurode(int numUnits, Random randomGen) {
            this.outWeights = new DoubleVector();
            this.gradientWeights = new DoubleVector();
            for (int i = 0; i < numUnits; i++) {
                this.outWeights.setValue(i, 0.2 * randomGen.nextDouble() - 0.1);
                this.gradientWeights.setValue(i, 0);
            }
            prevOutput = 0;
            output = 0;
            gradientNext = 0;
        }

        public void computeGradientsForRecurrentUnits(double[] gradientBiasInHiddenUnits) {
            for (int i = 0; i < gradientWeights.numValues(); i++) {
                gradientWeights.addToValue(i, gradientBiasInHiddenUnits[i] * prevOutput);
            }
            // update gradientNext
            double sum = 0;
            for (int i = 0; i < outWeights.numValues(); i++) {
                sum += outWeights.getValue(i) * gradientBiasInHiddenUnits[i];
            }
            gradientNext = sum;
        }

        public void updateOutWeights(double learningRate) {
            if (blClip) {
                for (int i = 0; i < gradientWeights.numValues(); i++) {
                    if (gradientWeights.getValue(i) > clippingGradient) {
                        gradientWeights.setValue(i, clippingGradient);
                    } else if (gradientWeights.getValue(i) < -1 * clippingGradient) {
                        gradientWeights.setValue(i, -1 * clippingGradient);
                    }
                }
            }
           gradientWeights.scaleValues(-1 * learningRate);
           outWeights.addValues(gradientWeights);
        }

        public double getGradientNext() {
            return gradientNext;
        }

        public double getPrevOutput() {
            return prevOutput;
        }

        public double getOutput() {
            return output;
        }

        public void storeOutput(double newOutput) {
            output = newOutput;
        }

        public double getWeight(int index) {
            // get the weight from this recurrent neurode (t-1) to the neurode index in the hidden layer (t)
           return outWeights.getValue(index);
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {}
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] forwardInput = {0};
        for (int i = 0; i < numLayers; i ++) {
            if (i == 0) {
                // from the input layer to the first hidden layer
                forwardInput = layers.get(i).forwardPropagate(inst);
            } else {
                forwardInput = layers.get(i).forwardPropagate(forwardInput);
            }
        }
        return forwardInput;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}
