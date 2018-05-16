import java.io.{File, PrintWriter}


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD

import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.XGBoost


import scala.io.Source
object ModelBasedCF {
  def main(args: Array[String]): Unit = {
    val startTime = System.nanoTime()
    val inputPath1: String = args(0)
    val inputPath2: String = args(1)
    val inputPath3: String = args(2)

    val inputFilePath = inputPath1
    val truthFilePath =   inputPath2
    val outPutFilePath = inputPath3
//    val inputFilePath = "Data\\video_small_num.csv"
//    val truthFilePath = "Data\\video_small_testing_num.csv"
//    val outPutFilePath = "outputModelCF.txt"
    val conf = new SparkConf()
    conf.setAppName("hw3")
    conf.setMaster("local[4]")
    val sc = new SparkContext(conf)

    val header = Source
      .fromFile(inputFilePath)
      .getLines()
      .toList
      .head

    val testingHeader =  Source
      .fromFile(truthFilePath)
      .getLines()
      .toList
      .head

    val testingTriple = sc.textFile(truthFilePath).filter(_ != testingHeader).map(_.split(',')).map(a => (a(0), a(1), a(2))).cache()
    val testingTripleSet = testingTriple.collect().toSet


    val data = sc.textFile(inputFilePath)
        .filter(_ != header)
        .map(_.split(','))
      .map(a => (a(0), a(1), a(2)))
      .filter(!testingTripleSet.contains(_))

    val ratings = data.map(a => Rating(a._1.toInt, a._2.toInt, a._3.toDouble))

    val rank = 3
    val numIterations = 17
    val model = ALS.train(ratings, rank, numIterations, 1.6)

    //3, 0.2 is a good choice
    //3, 0.2, 1, 4 is a seed
    //1, 0.04 is also good
    //1, 0.04, 1, 3 is a seed

    val usersProducts = testingTriple.map(a => (a._1.toInt, a._2.toInt))


//    val max: Double
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }.cache()
    val max = predictions.map(_._2).max()
    val min = predictions.map(_._2).min()

    def scaleBack(max:Double, min: Double):Double => Double = {
      val range = max - min
      def fun(value: Double): Double = {
        (value - min) * 4 / range + 1
      }
      fun
    }

    val ratesAndPreds = testingTriple.map { case (user, product, rate) =>
      ((user.toInt, product.toInt), rate.toDouble)
    }.join(predictions
      .mapValues(scaleBack(max, min))
    )
      .map(a => (a._1, a._2, Math.abs(a._2._1 - a._2._2))).cache()

    val endTime = System.nanoTime()

    //output
    val writer = new PrintWriter(new File(outPutFilePath))
    ratesAndPreds.collect().sortBy(_._1._2).sortBy(_._1._1).foreach(a => writer.write(a._1._1 + "," + a._1._2 + "," + a._2._2 + "\n"))
    writer.close()



    def printRangeCount(predictionRangeCount: Array[(Int, Int)], range: Int): Unit = {
      if (predictionRangeCount.exists(_._1 == range) ) {
        println(predictionRangeCount.filter(_._1 == range).head._2)
      } else {
        println(0)
      }
    }


    val predictionRangeCount = ratesAndPreds
      .map(a => Math.floor(a._3).toInt)
      .groupBy(a => a)
      .mapValues(_.size)
      .sortByKey()
      .collect()

    print(">=0 and <1:")
    printRangeCount(predictionRangeCount, 0)
    print(">=1 and <2:")
    printRangeCount(predictionRangeCount, 1)
    print(">=2 and <3:")
    printRangeCount(predictionRangeCount, 2)
    print(">=3 and <4:")
    printRangeCount(predictionRangeCount, 3)
    print(">=4:")
    printRangeCount(predictionRangeCount, 4)

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2), diff) =>
      diff * diff
    }.mean()
    val RMSE = Math.pow(MSE, 0.5)
    println(s"RMSE = $RMSE")
    println("Time:" + (endTime - startTime)/1000000000 + " sec")


//    sc
//      .textFile("course.txt")
//      .map(_.split(","))
//      .map(a => (a(0), a(1)))
//      .filter(!_._2.equals("Spring 2018"))
//      .count()
//    sc
//      .textFile("student.txt")
//      .map(_.split(","))
//      .map(a => (a(0), a(1)))
//      .groupBy(_._2)
//      .mapValues(a => a.size)
//
//    def mean(intList: Iterable[Int]): Double = {
//      intList.sum / intList.size.toDouble
//    }
//
//    sc
//      .textFile("quiz.txt")
//      .map(_.split(","))
//      .filter(a => a(0).equals("323"))
//      .map(a => (a(1), a(3).toInt))
//      .groupBy(_._1)
//      .mapValues(a => a.map(_._2))
//      .mapValues(mean)

  }

}
