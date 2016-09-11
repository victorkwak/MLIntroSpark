name := "SpamNaiveBayes"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.0.0",
  "org.scala-lang" %% "scala-compiler" % "2.11.8",
  "org.scala-lang" %% "scala-reflect" % "2.11.8"
)
