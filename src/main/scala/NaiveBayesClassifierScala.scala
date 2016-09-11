import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{HashingTF, IDF, LabeledPoint, Tokenizer}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

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

  val trainingData = {
    val dataDirectory: String = "./Data/test/RedditData/"
    val subreddits: Seq[String] = Seq("AMA", "AskEngineers", "Economics", "Fitness", "Showerthoughts")
    val dataDirectories: Seq[String] = subreddits.map(subreddit => dataDirectory + subreddit + ".TITLE")

    val data = {
      val input: Seq[DataFrame[String]] = dataDirectories
        .map(directory => spark.read.text(directory).as[String])
          .map(stringDataset => )
//        .map(stringDataset => stringDataset.toDF("label", "sentence"))

      val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
      val wordsData = input.map(stringDataset => tokenizer.transform(stringDataset))

      val tf = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
      val featureizedData = wordsData.map(stringDataFrame => tf.transform(stringDataFrame))
      val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
      val idfModels = featureizedData.map(tfDataFrame => idf.fit(tfDataFrame))
      val rescaledData = (idfModels, featureizedData).zipped
        .map { (idfData, stringDataFrame) => idfData.transform(stringDataFrame) }
      rescaledData
    }
    data.foreach(dataframe => dataframe.select("features", "label"))
  }
}
