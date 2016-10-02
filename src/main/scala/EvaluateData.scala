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

}
