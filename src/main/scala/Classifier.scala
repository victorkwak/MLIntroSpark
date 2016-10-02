import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame

/**
  * Victor Kwak, 9/10/16
  */
object Classifier {
  // A tokenizer splits sentences into their words, e.g., "ML is cool" to ["ML", "is", "cool"]
  // https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)
  private val tokenizer = new Tokenizer()
    .setInputCol("sentence")
    .setOutputCol("words")

  // Stop words are words that are filtered out.
  // In this case, words that don't impart a lot of meaning, e.g., "the", "and", "a", etc.
  // https://en.wikipedia.org/wiki/Stop_words
  private val remover = new StopWordsRemover()
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("filteredWords")

  // https://en.wikipedia.org/wiki/Vector_space_model
  private val countVectorizer = new CountVectorizer()
    .setInputCol(remover.getOutputCol)
    .setOutputCol("nonNormalizedFeatures")

  // https://en.wikipedia.org/wiki/Norm_(mathematics)
  private val normalizer = new Normalizer()
    .setInputCol(countVectorizer.getOutputCol)
    .setOutputCol("features")
    .setP(Double.PositiveInfinity)


  /**
    * Process the data using common Natural Language Processing techniques.
    * @param data
    * @return
    */
  private def process(data: DataFrame) = {
    val wordsData = tokenizer.transform(data)
    val filteredWordsData = remover.transform(wordsData)
    val countVectorizerModel = countVectorizer.fit(filteredWordsData)
    val featureizedData = countVectorizerModel.transform(filteredWordsData)
    val normalizedData = normalizer.transform(featureizedData)

    normalizedData.select("label", "sentence", "features")
  }

  /**
    * Shows a small portion of the evaluated data as well as the accuracy achieved
    * @param predictions
    */
  private def evaluateModel(predictions: DataFrame) = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    predictions.show()
    println("Accuracy: " + accuracy)
  }

  /**
    * Processes the data via commonly used Natural Language Processing techniques,
    * models the data using Naive Bayes and One vs Rest (via logistic regression),
    * then evaluates how well that model does.
    * @param data
    */
  def processModelEvaluate(data: DataFrame) = {
    val processedData = process(data)
    val Array(trainingData, testData) = processedData.randomSplit(Array(0.7, 0.3))

    val naiveBayesModel = new NaiveBayes().fit(trainingData)
    val naiveBayesPredictions = naiveBayesModel.transform(testData)
    evaluateModel(naiveBayesPredictions)

    val logisticRegression = new LogisticRegression()
    val oneVsRest = new OneVsRest().setClassifier(logisticRegression)
    val oneVsRestModel = oneVsRest.fit(trainingData)
    val oneVsRestPredictions = oneVsRestModel.transform(testData)
    evaluateModel(oneVsRestPredictions)
  }
}