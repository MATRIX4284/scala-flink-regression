import org.apache.flink.ml.MLUtils

val astroTrain: DataSet[LabeledVector] = MLUtils.readLibSVM(env, "/home/system/FLINK/scala-flink-ml/svmguide1")
val astroTest: DataSet[(Vector, Double)] = MLUtils.readLibSVM(env, "/home/system/FLINK/scala-flink-ml/svmguide1.t")
      .map(x => (x.vector, x.label))



import org.apache.flink.ml.classification.SVM

val svm = SVM()
  .setBlocks(env.getParallelism)
  .setIterations(100)
  .setRegularization(0.001)
  .setStepsize(0.1)
  .setSeed(42)

svm.fit(astroTrain)

val evaluationPairs: DataSet[(Double, Double)] = svm.evaluate(astroTest)

