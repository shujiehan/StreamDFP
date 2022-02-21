/*...*/
package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
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
 * Multi-layer Perceptron
 * @Author Shujie Han
 */
public class MultiLayerPerceptron extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 2L;

    @Override
    public String getPurposeString() {return "MLP classifier: Multi-layer classifier"; }

    public FloatOption learningRateOption = new FloatOption("learningRate", 'r', "Learning rate", 0.1);
    public IntOption numLayerOption = new IntOption("numLayer", 'l', "Number of layers", 2);
    public IntOption numUnitOption = new IntOption("numUnit", 'u', "Number of units per hidden layer", 3);
    public FloatOption lambdaNegativeOption = new FloatOption("lambdaNegatives", 'n', "The lambda parameter of downsampling for negative samples", 1.0, 1.0, Float.MAX_VALUE);
    public FloatOption lambdaPositiveOption = new FloatOption("lambdaPositives", 'p', "The lambda parameter of downsampling for positive samples", 1.0, 1.0, Float.MAX_VALUE);
    public IntOption numSamplesResetOption = new IntOption("numSampleReset", 's', "Reset MLP periodically", 10000, 100, Integer.MAX_VALUE);

    protected boolean reset;
    protected double learningRate;
    protected int numLayers;
    protected int numUnits; // numUnits per hidden layer
    protected List<Layer> layers; // store each layer
    protected long instancesSeen;
    protected int numSamplesReset;

    @Override
    public void resetLearningImpl() {
        this.reset = true;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.reset) {
            this.reset = false;
            learningRate = learningRateOption.getValue();
            numLayers = numLayerOption.getValue();
            numUnits = numUnitOption.getValue();
            layers = new ArrayList<Layer>();
            numSamplesReset = numSamplesResetOption.getValue();

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
            for (int i = 0; i < numLayers; i ++) {
                if (i == 0) {
                   // from the input layer to the first hidden layer
                    forwardInput = layers.get(i).forwardPropagate(weightedInst);
                } else {
                    forwardInput = layers.get(i).forwardPropagate(forwardInput);
                }
            }

            // propagate the errors backward
            double[] errors = {0};
            for (int i = numLayers - 1; i >= 0 ; i--) {
                if (i == numLayers - 1) {
                    // compute the error for the output layer
                    errors = layers.get(i).computeErrorInOutputLayer(weightedInst);
                } else {
                    // from the output layer to the last hidden layer
                    errors = layers.get(i).computeErrorInHiddenLayer(errors, layers.get(i + 1));
                }
            }

            // update all inWeights and bias;
            for (int i = 0; i < numLayers; i++) {
                if (i == 0) {
                    layers.get(i).updateParameters(weightedInst, learningRate);
                } else {
                    layers.get(i).updateParameters(layers.get(i - 1), learningRate);
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
        }

        public double[] forwardPropagate(Instance inst) {
            // forward propagate for the first hidden layer
            double[] inputToNextLayer = new double[numUnits];
            for (int i = 0; i < numUnits; i++) {
                double output = units.get(i).computeOutput(inst);
                inputToNextLayer[i] = output;
            }
            return inputToNextLayer;
        }

        public double[] forwardPropagate(double[] input) {
            // forward propagate for the remaining hidden layers
            double[] inputToNextLayer = new double[numUnits];
            for (int i = 0; i < numUnits; i++) {
                double output = units.get(i).computeOutput(input);
                inputToNextLayer[i] = output;
            }
            return inputToNextLayer;
        }

        public double[] computeErrorInOutputLayer(Instance inst) {
            double[] errors = new double[numUnits];
            for (int i = 0; i < numUnits; i++) {
                errors[i] = units.get(i).computeError(inst);
            }
            return errors;
        }

        public double[] computeErrorInHiddenLayer(double[] errorsInNextLayer, Layer nextLayer) {
            double[] errors = new double[numUnits];
            for (int i = 0; i < numUnits; i++) {
                double[] inWeightsFromThisUnit = new double[nextLayer.numUnits];
                // get inWeights from this unit to all units in the next layer;
                for (int j = 0; j < nextLayer.numUnits; j++) {
                    inWeightsFromThisUnit[j] = nextLayer.getInWeightByIndex(i, j);
                }
                errors[i] = units.get(i).computeError(errorsInNextLayer, inWeightsFromThisUnit);
            }
            return errors;
        }

        public double getInWeightByIndex(int fromUnit, int thisUnit) {
            return units.get(thisUnit).getInWeight(fromUnit);
        }

        public double getUnitOutput(int idxUnit) {
            return units.get(idxUnit).getOutput();
        }

        public void updateParameters(Instance inst, double learningRate) {
            // update parameters for the first hidden layer
            for (int i = 0; i < numUnits; i++) {
                units.get(i).updateBias(learningRate);
                // outputs from the previous layer
                double[] outputs = new double[numPreUnits];
                for (int j = 0; j < inst.numAttributes() - 1; j++) {
                    int idxAtt = modelAttIndexToInstanceAttIndex(j, inst);
                    outputs[j] = inst.value(idxAtt);
                }
                units.get(i).updateInWeights(outputs, learningRate);
            }
        }

        public void updateParameters(Layer preLayer, double learningRate) {
            // update parameters for the remaining hidden layer
            for (int i = 0; i < numUnits; i++) {
                units.get(i).updateBias(learningRate);
                // outputs from the previous layer
                double[] outputs = new double[numPreUnits];
                for (int j = 0; j < numPreUnits; j++) {
                    outputs[j] = preLayer.getUnitOutput(j);
                }
                units.get(i).updateInWeights(outputs, learningRate);
            }
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {}
    }


    public class Neurode extends AbstractMOAObject {
        /** The units in the hidden layers and output layer. */
        private static final long serialVersionUID = 1L;
        protected DoubleVector inWeights; // the weights from all neurodes in the previous layer to this neurode
        protected double bias;
        protected double output; // forward output
        protected double error; // backward error

        public Neurode(int numPreUnits, Random randomGen) {
            this.inWeights = new DoubleVector();
            for (int i = 0; i < numPreUnits; i++) {
                this.inWeights.setValue(i, 0.2 * randomGen.nextDouble() - 0.1);
            }
            bias = 0.2 * randomGen.nextDouble() - 0.1;
            output = 0;
            error = 0;
        }

        // compute the output of forward propagation for the first hidden layer
        public double computeOutput(Instance inst) {
            output = 0;
            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                int idxAtt = modelAttIndexToInstanceAttIndex(i, inst);
                output += inst.value(idxAtt) * inWeights.getValue(i);
            }
            output += bias;
            output = 1.0 / (1.0 + Math.exp(-output)); // sigmoid function
            return output;
        }

        // compute the output of forward propagation for the remaining hidden layers
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
        public double computeError(Instance inst) {
            error = output * (1 - output) * (inst.classValue() - output);
            return error;
        }

        public double computeError(double[] errorsInNextLayer, double[] inWeightsFromThisUnit) {
            double errorSum = 0;
            for (int i = 0; i < errorsInNextLayer.length; i++) {
               errorSum += errorsInNextLayer[i] * inWeightsFromThisUnit[i];
            }
            error = output * (1 - output) * errorSum;
            return error;
        }

        public double getInWeight(int index) {
            return inWeights.getValue(index);
        }

        public double getError() {
           return error;
        }

        public double getOutput() {
            return output;
        }

        public void updateInWeights(double[] outputsFromPreLayer, double learningRate) {
            DoubleVector deltaWeights = new DoubleVector(outputsFromPreLayer);
            deltaWeights.scaleValues(learningRate * error);
            inWeights.addValues(deltaWeights);
        }

        public void updateBias(double learningRate) {
            bias += learningRate * error;
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
