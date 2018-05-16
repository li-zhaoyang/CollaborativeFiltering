name := "do"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.2.1",
  "org.apache.spark" %% "spark-mllib" % "2.2.1",
  "com.soundcloud" % "cosine-lsh-join-spark_2.11" % "1.0.1"
)
// https://mvnrepository.com/artifact/ml.dmlc/xgboost4j
libraryDependencies += "ml.dmlc" % "xgboost4j" % "0.8-wmf-1"
