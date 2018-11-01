package flinker
import org.apache.flink.api.scala._
import org.apache.flink.ml.MLUtils
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.recommendation._
import org.apache.flink.ml.common.ParameterMap


object flinkml {
  def main(args: Array[String]) {



    val env = ExecutionEnvironment.getExecutionEnvironment;
    val astroTrain: DataSet[LabeledVector] = MLUtils.readLibSVM(env, "/home/system/FLINK/scala-flink-ml/svmguide1");
    val astroTest: DataSet[(org.apache.flink.ml.math.Vector, Double)] = MLUtils.readLibSVM(env, "/home/system/FLINK/scala-flink-ml/svmguide1.t").map(x => (x.vector, x.label));
    val astroTest2: DataSet[org.apache.flink.ml.math.Vector] = MLUtils.readLibSVM(env, "/home/system/FLINK/scala-flink-ml/svmguide1.t").map(x => x.vector);

import org.apache.flink.ml.classification.SVM
import org.apache.flink.ml.regression.MultipleLinearRegression


val mlr = MultipleLinearRegression()
.setIterations(10)
.setStepsize(0.5)
.setConvergenceThreshold(0.001)

val svm = SVM()
  .setBlocks(2)
  .setIterations(20)
  .setRegularization(0.001)
  .setStepsize(0.1)
  .setSeed(42);


svm.fit(astroTrain);
mlr.fit(astroTrain);

val evaluationPairs: DataSet[(Double, Double)] = svm.evaluate(astroTest)


val predictions = mlr.predict(astroTest2)

predictions.print()
evaluationPairs.writeAsText("/home/system/FLINK/scala-flink-ml/liblu.csv")
env.execute();

  }
}
