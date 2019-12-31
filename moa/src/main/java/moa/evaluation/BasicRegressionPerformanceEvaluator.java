/*
 *    BasicRegressionPerformanceEvaluator.java
 *    Copyright (C) 2011 University of Waikato, Hamilton, New Zealand
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

import moa.AbstractMOAObject;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.Measurement;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceData;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.core.Utils;

import java.util.HashMap;
import java.util.List;

/**
 * Regression evaluator that performs basic incremental evaluation.
 *
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
public class BasicRegressionPerformanceEvaluator extends AbstractMOAObject
        implements RegressionPerformanceEvaluator {

    private static final long serialVersionUID = 1L;

    protected double weightObserved;

    protected double squareError;

    protected double averageError;

    protected double sumTarget;
    
    protected double squareTargetError;
    
    protected double averageTargetError;

    // Shujie add:
    protected double averageErrorDaysBeforeFailure;

    protected double averageErrorForFailures;

    protected double squareErrorForFailures;

    protected int numFailures;

    protected int labelWindowSize;

    @Override
    public void reset() {
        this.weightObserved = 0.0;
        this.squareError = 0.0;
        this.averageError = 0.0;
        this.sumTarget = 0.0;
        this.averageTargetError = 0.0;
        this.squareTargetError = 0.0;
        this.numFailures = 0;
        this.averageErrorDaysBeforeFailure = 0.0;
    }

    public void setLabelWindowSize(int labelWindowSize) {
        this.labelWindowSize = labelWindowSize;
    }

    @Override
    public void addResult(Example<Instance> example, double[] prediction) {
	Instance inst = example.getData();
        if (inst.weight() > 0.0) {
            if (prediction.length > 0) {
                double meanTarget = this.weightObserved != 0 ? 
                            this.sumTarget / this.weightObserved : 0.0;
                this.squareError += (inst.classValue() - prediction[0]) * (inst.classValue() - prediction[0]);
                this.averageError += Math.abs(inst.classValue() - prediction[0]);
                this.squareTargetError += (inst.classValue() - meanTarget) * (inst.classValue() - meanTarget);
                this.averageTargetError += Math.abs(inst.classValue() - meanTarget);
                this.sumTarget += inst.classValue();
                this.weightObserved += inst.weight();
                assert(inst.classValue() >= 0 && inst.classValue() <= 1);
                if (inst.classValue() > 0) {
                    this.numFailures ++;
                    this.averageErrorForFailures += Math.abs(inst.classValue() - prediction[0]);
                    this.squareErrorForFailures += (inst.classValue() - prediction[0]) * (inst.classValue() - prediction[0]);
                    // averageErrorDaysBeforeFailure > 0 --> pessimistic; averageErrorDaysBeforeFailure < 0 --> optimistic
                    this.averageErrorDaysBeforeFailure += (prediction[0] - inst.classValue()) * (this.labelWindowSize + 1.0);
                    //System.out.println("class value = " + inst.classValue() + ", prediction = " + prediction[0]);
                } 
            }
           //System.out.println(inst.classValue()+", "+prediction[0]);
        }
    }

    // Shujie Add:
    public void addResultDelay(List<Instance> instances) {
        Instance inst = instances.get(0);
        assert(inst.classValue() >= 0 && inst.classValue() <= 1);
        double[] votes = inst.getPredictedVotes();
        addResult((Example<Instance>) new InstanceExample(inst), votes);
    }

    @Override
    public Measurement[] getPerformanceMeasurements() {
        return new Measurement[]{
                    new Measurement("classified instances",
                    getTotalWeightObserved()),
                    new Measurement("mean absolute error",
                    getMeanError()),
                    new Measurement("root mean squared error",
                    getSquareError()),
                    new Measurement("relative mean absolute error",
                    getRelativeMeanError()),
                    new Measurement("relative root mean squared error",
                    getRelativeSquareError()),
                    new Measurement("mean absolute error for failures",
                    getMeanErrorForFailures()),
                    new Measurement("root mean squared error for failures",
                    getSquareErrorForFailures()),
                    new Measurement("average error days before failures",
                    getAverageErrorDaysBeforeFailure()),
                    new Measurement("num failures", getNumFailures())
        };
    }

    // Shujie Add:
    public int getNumFailures() {
        return this.numFailures;
    }

    public double getAverageErrorDaysBeforeFailure() {
        return this.averageErrorDaysBeforeFailure / (1.0 * numFailures);
    }

    public double getMeanErrorForFailures() {
        return this.averageErrorForFailures / this.numFailures;
    }

    public double getSquareErrorForFailures() {
        return this.squareErrorForFailures / this.numFailures;
    }

    public double getTotalWeightObserved() {
        return this.weightObserved;
    }

    public double getMeanError() {
        return this.weightObserved > 0.0 ? this.averageError
                / this.weightObserved : 0.0;
    }

    public double getSquareError() {
        return Math.sqrt(this.weightObserved > 0.0 ? this.squareError
                / this.weightObserved : 0.0);
    }

    public double getTargetMeanError() {
        return this.weightObserved > 0.0 ? this.averageTargetError
                / this.weightObserved : 0.0;
    }

    public double getTargetSquareError() {
        return Math.sqrt(this.weightObserved > 0.0 ? this.squareTargetError
                / this.weightObserved : 0.0);
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        Measurement.getMeasurementsDescription(getPerformanceMeasurements(),
                sb, indent);
    }

    private double getRelativeMeanError() {
        //double targetMeanError = getTargetMeanError();
        //return targetMeanError > 0 ? getMeanError()/targetMeanError : 0.0;
        return this.averageTargetError> 0 ?
                this.averageError/this.averageTargetError : 0.0;
}

    private double getRelativeSquareError() {
        //double targetSquareError = getTargetSquareError();
        //return targetSquareError > 0 ? getSquareError()/targetSquareError : 0.0;
    return Math.sqrt(this.squareTargetError> 0 ?
                this.squareError/this.squareTargetError : 0.0);
    }
    
    @Override
    public void addResult(Example<Instance> example, Prediction prediction) {
    	if(prediction!=null)
    		addResult(example,prediction.getVotes(0));
    }
}
