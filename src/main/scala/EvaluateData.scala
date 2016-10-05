import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

/**
  * Victor Kwak, 10/1/16
  */
object EvaluateData extends App {
  Logger.getLogger("org").setLevel(Level.OFF)
  System.setProperty("hadoop.home.dir", ".\\winutils\\")
  // configuration
  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Naive Bayes Spam filter")
    .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
    .getOrCreate()

  import spark.implicits._ //So that I can convert Dataset to DataFrame

  val presDebateData = {
    val dataDirectory: String = "./Data/Debate/debateClean"

    spark.read.textFile(dataDirectory).map(_.split(": ")).map(
      array => array(0) match {
        case "HOLT" => (0, array(1))
        case "CLINTON" => (1, array(1))
        case "TRUMP" => (2, array(1))
      }
    ).toDF("label", "sentence")
  }

  val redditData = {
    val dataDirectory: String = "./Data/RedditData/"
    val subreddits: Seq[String] = Seq(
      "AMA",
      "AskEngineers",
      "BuyItForLife",
      "DnD",
      "Economics",
      "Fitness",
      "Frugal",
      "JamesBond",
      "LifeProTips",
      "Showerthoughts"
    )
    val dataDirectories: Seq[String] = subreddits.map(subreddit => dataDirectory + subreddit + ".TITLE")

    dataDirectories
      .map(directory => spark.read.text(directory).as[String])
      .zipWithIndex.map { case (dataset, i) => dataset.map(string => (i, string)) }
      .map(dataset => dataset.toDF("label", "sentence"))
      .reduce(_ union _)
  }

  val smsSpamData = {
    val dataDirectory: String = "./Data/spam/SMSSpam"

    spark.read.textFile(dataDirectory).map(_.split("\t")).map(
      array => array(0) match {
        case "ham" => (0, array(1))
        case "spam" => (1, array(1))
      }
    ).toDF("label", "sentence")
  }

  Classifier.processModelEvaluate(presDebateData, "Presidential Debate Data")
  Classifier.processModelEvaluate(redditData, "Reddit Title Data")
  Classifier.processModelEvaluate(smsSpamData, "SMS Spam Data")
}
