package scala_linreg
import breeze.linalg._
import breeze.stats.distributions.Rand.VariableSeed.randBasis

import java.io.File
import java.io.PrintWriter
import scopt.OParser



case class Config(
                   train: Boolean = false,
                   test: Boolean = false,
                   dataset_path: File = new File(""),
                   target_path: File = new File(""),
                   model_path: File = new File("")
                 )



object Main {
  val usage =
    """
      Usage:
      --train         Activate train mode
      --test          Activate test mode
      -d, --dataset   Path to dataset, required option
      -t, -- target   Path to target values for train dataset
      -m, --model     Path to pretrained model
      """.stripMargin
  val builder = OParser.builder[Config]
  val parser1 = {
    import builder._
    OParser.sequence(
      programName("scala_linreg"),
      opt[Unit]("train")
        .action((_, c) => c.copy(train=true))
        .text("Train mode activated!"),
      opt[Unit]("test")
        .action((_, c) => c.copy(test=true))
        .text("Test mode activted!"),
      opt[File]('d', "dataset")
        .required()
        .valueName("<file>")
        .action((x, c) => c.copy(dataset_path = x))
        .text("out is a required file property"),
      opt[File]('t', "target")
        .valueName("<file>")
        .action((x, c) => c.copy(target_path = x))
        .text("out is a required file property"),
      opt[File]('m', "model")
        .valueName("<file>")
        .action((x, c) => c.copy(model_path = x))
        .text("out is a required file property"),
    )
  }

  def LinearRegression(features: DenseMatrix[Double],
                       target:DenseMatrix[Double],
                       LearningRate: Double,
                       MaxStep: Int): DenseMatrix[Double] = {
    var model = DenseMatrix.rand[Double](features.cols, 1)
    val tol = 0.00001
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
    val trainFeature: DenseMatrix[Double] = DenseMatrix.zeros[Double](trainLength, features.cols)
    val testFeature: DenseMatrix[Double] = DenseMatrix.zeros[Double](features.rows - trainLength, features.cols)
    val trainTarget: DenseMatrix[Double] = DenseMatrix.zeros[Double](trainLength, target.cols)
    val testTarget: DenseMatrix[Double] = DenseMatrix.zeros[Double](features.rows - trainLength, target.cols)
    val rowsCount: Int = features.rows
    val colsCount: Int = features.cols

    val indexVec = shuffle(DenseVector.tabulate(features.rows){i => i.toFloat})
    for (index <- 0 until rowsCount){
      val feature_index: Int = indexVec(index).toInt
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

 def main(args: Array[String]): Unit ={
   OParser.parse(parser1, args, Config()) match {
     case Some(config) =>
       {
         val data = csvread(config.dataset_path)
         if (config.train) {
           val target = csvread(config.target_path)
           val learningRate = 0.1
           val maxStep = 1000
           val trainValSplit = 0.9
           val (trainFeature, valFeature, trainTarget,  valTarget) = SplitTrainTest(data, target, trainValSplit)
           val model = LinearRegression(trainFeature, trainTarget, learningRate, maxStep)
           csvwrite(new File("linreg_model.csv"), model)
           val errorTrain = LinearRegressionMSE(trainFeature, trainTarget, model)
           val errorValidation = LinearRegressionMSE(valFeature, valTarget, model)
           new PrintWriter("train.log") {write("Train log \n");
             write("Train error: "); write(errorTrain.toString);
             write("\nValidation error: "); write(errorValidation.toString);
           close }
           println("Model is trained and saved in linreg_model.csv! \nSee train.log for more details")
         }
         if (config.test){
           val model = csvread(config.model_path)
           val inference = data * model
           csvwrite(new File("inference.csv"), inference)
           println("Inference saved in inference,csv!")
         }
       }
     case _ => println(usage)
   }
 }
}

