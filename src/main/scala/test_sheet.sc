import breeze.linalg._
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import java.io.File



val data = csvread(new File("/Users/valeriytashchilin/IdeaProjects/unt/data/data.csv"))
println(data.cols, data.rows)
val target = csvread(new File("/Users/valeriytashchilin/IdeaProjects/unt/data/target.csv"))
println(target.cols, target.rows)


val trainSplitRatio = 0.8 // 80% for training
val kFoldParam = 5 // Cross validation in 5 folds

for (i <- 0 until 5) {
  println(i)
}


def LinearRegression(features: DenseMatrix[Double],
                     target:DenseMatrix[Double],
                     LearningRate: Double,
                     MaxStep: Int): DenseMatrix[Double] = {
  var model = DenseMatrix.rand[Double](features.cols, 1)
  val tol = 0.0001
  val row: Double = features.rows
  for (step <- 0 until MaxStep) {
    val res =  target - features * model
    val grad = features.t * res / row
    val derModel = LearningRate * grad
    val normGrad : Double = max(grad.t * grad)
    model = model + derModel
    if (normGrad < tol) {
      return model
    }
  }
  return model
} : DenseMatrix[Double]


def LinearRegressionMSE(features: DenseMatrix[Double], target: DenseMatrix[Double], model:DenseMatrix[Double]): DenseMatrix[Double] = {
  val res = features * model - target
  val row: Double = features.rows
  val err = res.t * res / row
  return err
}: DenseMatrix[Double]

def SplitTrainTest(features: DenseMatrix[Double], target: DenseMatrix[Double], trainTestRatio: Double):
(DenseMatrix[Double], DenseMatrix[Double],DenseMatrix[Double],DenseMatrix[Double]) = {
  val trainLength: Int = (features.rows * trainTestRatio).toInt
  var trainFeature: DenseMatrix[Double] = DenseMatrix.zeros[Double](trainLength, features.cols)
  var testFeature: DenseMatrix[Double] = DenseMatrix.zeros[Double](features.rows - trainLength, features.cols)
  var trainTarget: DenseMatrix[Double] = DenseMatrix.zeros[Double](trainLength, target.cols)
  var testTarget: DenseMatrix[Double] = DenseMatrix.zeros[Double](features.rows - trainLength, target.cols)
  val rowsCount: Int = features.rows
  val colsCount: Int = features.cols

  val indexVec = shuffle(DenseVector.tabulate(features.rows){i => i})
  for (index <- 0 until rowsCount){
    val feature_index: Int = indexVec(index)
    if (index < trainLength){
      for (row <- 0 until colsCount){
        //          println(index, row, features.rows, indexVec(index))
        trainFeature(index, row) = features(feature_index, row)
      }
      trainTarget(index, 0) = target(feature_index, 0)
    }
    else{
      for (row <- 0 until colsCount){
        testFeature(index - trainLength, row) = features(feature_index, row)
      }
      testTarget(index - trainLength, 0) = target(feature_index, 0)
    }
  }
  return (trainFeature, testFeature, trainTarget, testTarget)
}: (DenseMatrix[Double], DenseMatrix[Double],DenseMatrix[Double],DenseMatrix[Double])

//def SplitTrainTest(features: DenseMatrix[Double], target: DenseMatrix[Double], trainTestRatio: Double):
//(DenseMatrix[Double], DenseMatrix[Double],DenseMatrix[Double],DenseMatrix[Double]) = {
//  val trainLength: Int = (features.rows * trainTestRatio).toInt
//  val trainFeature: DenseMatrix[Double] = features(0 until trainLength, 0 until features.cols)
//  val testFeature: DenseMatrix[Double] =  features(trainLength until features.rows, 0 until features.cols)
//  val trainTarget: DenseMatrix[Double] =  target(0 until trainLength, 0 until target.cols)
//  val testTarget: DenseMatrix[Double] = target(trainLength until target.rows, 0 until target.cols)
//  return (trainFeature, testFeature, trainTarget, testTarget)
//}: (DenseMatrix[Double], DenseMatrix[Double],DenseMatrix[Double],DenseMatrix[Double])


val (trainFeat, testFeat, trainVal, testVal) = SplitTrainTest(data, target, 0.8)
println(trainFeat.rows, trainFeat.cols)
println(trainVal.rows, trainVal.cols)
val model = LinearRegression(trainFeat, trainVal, 0.1, 100)

val trainError = LinearRegressionMSE(trainFeat, trainVal, model)
println(trainError)

println(testFeat.rows, testFeat.cols)
println(testVal.rows, testVal.cols)
val trainError = LinearRegressionMSE(testFeat, testVal, model)
println(trainError)

//println(testFeat.rows, testFeat.cols)
//println(testVal.rows, testVal.cols)
//println(model.rows, model.cols)
//val resq = trainFeat * model
//val res = testFeat * model
//val row: Double = testFeat.rows
//val err = res.t * res / row

//val testError = LinearRegressionMSE(testFeat, testVal, model)
//println(testError)


//val model = data \ target
//val err_vec = data * model - target
//val error = err_vec.t * err_vec
//val batch_size = DenseMatrix(target.rows).mapValues(_.toDouble)
//val mse_error = error /:/ batch_size
//println(mse_error)
//println(error(0) / batch_size)

