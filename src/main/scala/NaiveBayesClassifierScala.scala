import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

/**
  * Victor Kwak, 9/10/16
  */
object NaiveBayesClassifierScala extends App {
  // configuration set for local running on 4 cores.
  val spark = SparkSession
    .builder()
    .master("local[4]")
    .appName("Naive Bayes Spam filter")
    .getOrCreate()

  import spark.implicits._

  val data = {
    val dataDirectory: String = "./Data/RedditData/"
    val subreddits: Seq[String] = Seq(
      "AMA",
//      "AskEngineers",
//      "BuyItForLife",
//      "DnD",
//      "Economics",
//      "Fitness",
      "Frugal",      "JamesBond",
      "LifeProTips",
      "Showerthoughts"
    )
    val dataDirectories: Seq[String] = subreddits.map(subreddit => dataDirectory + subreddit + ".TITLE")

    val input = dataDirectories
      .map(directory => spark.read.text(directory).as[String])
      .zipWithIndex.map { case (stringDataset, i) => stringDataset.map(string => (i, string)) }
      .map(stringDataset => stringDataset.toDF("label", "sentence"))
      .reduce(_ union _)

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("filteredWords")
    //    val tf = new HashingTF().setInputCol(remover.getOutputCol).setOutputCol("nonNormalFeatures")
    val countVectorizer = new CountVectorizer().setInputCol(remover.getOutputCol).setOutputCol("nonNormalFeatures")
    val normalizer = new Normalizer().setInputCol(countVectorizer.getOutputCol).setOutputCol("features")
      .setP(2)
    //    val normalizer = new Normalizer().setInputCol(tf.getOutputCol).setOutputCol("features")
    //      .setP(2)


    val wordsData = tokenizer.transform(input)
    val filteredWordsData = remover.transform(wordsData)
    val countVectorizerModel = countVectorizer.fit(filteredWordsData)
    val featureizedData = countVectorizerModel.transform(filteredWordsData)
    val normalizedData = normalizer.transform(featureizedData)

    normalizedData
  }

  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  val model = new NaiveBayes().fit(trainingData)

  val predictions = model.transform(testData)
  predictions.show(1000)

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println("Accuracy: " + accuracy)
}