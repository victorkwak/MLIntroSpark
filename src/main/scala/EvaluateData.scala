import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

/**
  * Victor Kwak, 10/1/16
  */
object EvaluateData extends App{
  Logger.getLogger("org").setLevel(Level.OFF)
  // configuration set for local running on 4 cores.
  val spark = SparkSession
    .builder()
    .master("local[4]")
    .appName("Naive Bayes Spam filter")
    .getOrCreate()

  import spark.implicits._

  val presDebateData = {
    val dataDirectory: String = "./Data/Debate/debateClean"

    val data = spark.read.textFile(dataDirectory).map(_.split(": ")).map(
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

}
