import org.apache.spark.sql.SparkSession

/**
  * Victor Kwak, 9/11/16
  */
object JSONTest extends App {
  // configuration set for local running on 4 cores.
  val spark = SparkSession
    .builder()
    .master("local[4]")
    .appName("Naive Bayes Spam filter")
    .getOrCreate()

  val path = "./Data/subAndTitle/*.json"
  val data = spark.read.json(path)

  import spark.implicits._

  val ids = data.select("subreddit").distinct().rdd.zipWithIndex().map {
    case (row, i) => (i, row.mkString)}.toDF("ID", "subreddit")

  val dataWithID = data.join(ids, "subreddit")

  ids.show(100)
  data.show(100)
  dataWithID.show(100)
}
