package flinker
import org.apache.flink.api.scala._
import org.apache.flink.ml.MLUtils
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.recommendation._
import org.apache.flink.ml.common.ParameterMap
import org.apache.flink.ml.common._
import org.apache.flink.streaming.api.scala._

object flinkregression {
  def main(args: Array[String]) {

    val env = ExecutionEnvironment.getExecutionEnvironment;
    val astroTrain: DataSet[LabeledVector] = MLUtils.readLibSVM(env, "/home/system/FLINK/scala-flink-ml/svmguide1");

    val astroTest2: DataSet[org.apache.flink.ml.math.Vector] = MLUtils.readLibSVM(env, "/home/system/FLINK/scala-flink-ml/svmguide1.t").map(x => x.vector);

import org.apache.flink.ml.classification.SVM
import org.apache.flink.ml.regression.MultipleLinearRegression

val mlr = MultipleLinearRegression()
.setIterations(10)
.setStepsize(0.5)
.setConvergenceThreshold(0.001)

mlr.fit(astroTrain);


val model = mlr.weightsOption.get

val weightVectorTypeInfo = TypeInformation.of(classOf[WeightVector])
val weightVectorSerializer = weightVectorTypeInfo.createSerializer(new ExecutionConfig())
val outputFormat = new TypeSerializerOutputFormat[WeightVector]
outputFormat.setSerializer(weightVectorSerializer)

model.write(outputFormat, "/home/system/FLINK/scala-flink-ml/weights")

val predictions = mlr.predict(astroTest2)

predictions.writeAsText("/home/system/FLINK/scala-flink-ml/output001")

env.execute();

  }
}
