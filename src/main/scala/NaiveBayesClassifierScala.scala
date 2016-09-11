import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Victor Kwak, 9/10/16
  */
object NaiveBayesClassifierScala extends App {
  // configuration set for local running on 4 cores.
  val configuration = new SparkConf()
    .setAppName("Naive Bayes Spam Filter")
    .setMaster("local[4]")
  val sc = new SparkContext(configuration)

  val trainingData: RDD[LabeledPoint] = {
    val dataDirectory: String = "./Data/test/RedditData/"
    val subreddits: Seq[String] = Seq("AMA", "AskEngineers", "Economics", "Fitness", "Showerthoughts")
    val dataDirectories: Seq[String] = subreddits.map(subreddit => dataDirectory + subreddit + ".TITLE")
    val stopWords = sc.textFile("./stopwords")

    val tf = new HashingTF()
    val idf = new IDF()
    val data: Seq[RDD[LabeledPoint]] = {
      //sequence of big strings (titles)
      val input: Seq[RDD[String]] = dataDirectories
        .map(directory => sc.textFile(directory))
        .map(titles => titles.map(title => title.toLowerCase))
        .map(titles => titles.subtract(stopWords))

      input.foreach(titles => titles.foreach(println))
      val labeledPoints = input
        .zipWithIndex.map { case (stringRDD, i) => stringRDD
        .map(line => tf.transform(line.split("\\s")))
        .map(features => LabeledPoint(i, features))
      }
      labeledPoints
    }
    sc.union(data)
  }
  trainingData.cache()

  val randomSplit = trainingData.randomSplit(Array(0.6, 0.4))
  val training = randomSplit(0)
  val test = randomSplit(1)
  val model = NaiveBayes.train(training, lambda = 1.0)

  val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
  val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

  println(accuracy)
}
