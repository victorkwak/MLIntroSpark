import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.annotation.tailrec

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
    val dataDirectory: String = "./Data/test/RedditData/"
    val subreddits: Seq[String] = Seq(
      "AMA",
      "AskEngineers",
      "Economics",
      "Fitness",
      "Showerthoughts"
    )
    val dataDirectories: Seq[String] = subreddits.map(subreddit => dataDirectory + subreddit + ".TITLE")

    val input = dataDirectories
      .map(directory => spark.read.text(directory).as[String])
      .zipWithIndex.map { case (stringDataset, i) => stringDataset.map(string => (i, PorterStemmer.stem(string))) }
      .map(stringDataset => stringDataset.toDF("label", "sentence"))

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords")
    val wordsData = input.map(stringDataset => tokenizer.transform(stringDataset))
    val filteredWordsData = wordsData.map(words => remover.transform(words))

    val tf = new HashingTF().setInputCol("filteredWords").setOutputCol("features")
    val featureizedData = filteredWordsData.map(stringDataFrame => tf.transform(stringDataFrame))

    def mergeData(dataSequence: Seq[DataFrame]) = {
      @tailrec def helper(head: DataFrame, tail: Seq[DataFrame]): DataFrame = {
        if (tail.isEmpty) head
        else helper(head union tail.head, tail.tail)
      }
      helper(dataSequence.head, dataSequence.tail)
    }
    mergeData(featureizedData)
  }

  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  val model = new NaiveBayes().fit(trainingData)

  val predictions = model.transform(testData)
  predictions.show()

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println("Accuracy: " + accuracy)
}