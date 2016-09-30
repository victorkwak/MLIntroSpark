import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Victor Kwak, 9/10/16
  */
object NaiveBayesClassifierScala extends App {
  Logger.getLogger("org").setLevel(Level.OFF)
  // configuration set for local running on 4 cores.
  val spark = SparkSession
    .builder()
    .master("local[4]")
    .appName("Naive Bayes Spam filter")
    .getOrCreate()

  import spark.implicits._

  val data = {
    val dataDirectory: String = "./Data/Debate/debateClean"

    val input = spark.read.textFile(dataDirectory).map(_.split(": ")).map(
      array => array(0) match {
        case "HOLT" => (0, array(1))
        case "CLINTON" => (1, array(1))
        case "TRUMP" => (2, array(1))
      }
    ).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("filteredWords")
    val countVectorizer = new CountVectorizer().setInputCol(remover.getOutputCol).setOutputCol("nonNormalizedFeatures")
    val normalizer = new Normalizer().setInputCol(countVectorizer.getOutputCol).setOutputCol("features").setP(Double.PositiveInfinity)

    val wordsData = tokenizer.transform(input)
    val filteredWordsData = remover.transform(wordsData)
    val countVectorizerModel = countVectorizer.fit(filteredWordsData)
    val featureizedData = countVectorizerModel.transform(filteredWordsData)
    val normalizedData = normalizer.transform(featureizedData)

    normalizedData.select("label", "sentence", "features")
  }

  def evaluateModel(predictions: DataFrame) = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    predictions.show()
    println("Accuracy: " + accuracy)
  }

  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  //Naive Bayes
  val naiveBayesModel = new NaiveBayes().fit(trainingData)
  val naiveBayesPredictions = naiveBayesModel.transform(testData)
  evaluateModel(naiveBayesPredictions)

  //Logistic Regression
  val logisticRegressionModel = new LogisticRegression().fit(trainingData)
  val logisticRegressionPredictions = logisticRegressionModel.transform(testData)
  evaluateModel(logisticRegressionPredictions)
}