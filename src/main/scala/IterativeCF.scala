import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

object IterativeCF {
  def main(args: Array[String]): Unit = {
    val startTime = System.nanoTime()
    val inputPath1: String = args(0)
    val inputPath2: String = args(1)
    val inputPath3: String = args(2)

    val inputFilePath = inputPath1
    val truthFilePath =   inputPath2
    val outPutFilePath = inputPath3

    val conf = new SparkConf()
    conf.setAppName("hw3")
    conf.setMaster("local[4]")
    val sc = new SparkContext(conf)


    // don't trust if only one mutrual

    val headerheader = Source
      .fromFile(inputFilePath)
      .getLines()
      .toList
      .head

    val testingHeader =  Source
      .fromFile(truthFilePath)
      .getLines()
      .toList
      .head

    val testingTriple = sc.textFile(truthFilePath)
      .filter(_ != testingHeader)
      .map(_.split(','))
      .map(a => (a(0).toInt, a(1).toInt, a(2))).cache()
    val testingTripleSet = testingTriple.collect().toSet
    val testingUsers = testingTriple.map(_._1).distinct()
    val testingUserList = testingTripleSet.map(_._1).toList
    val testingItemList = testingTripleSet.map(_._2).toList

    val data = sc.textFile(inputFilePath)
      .filter(_ != headerheader )
      .map(_.split(','))
      .map(a => (a(0).toInt, a(1).toInt, a(2).toDouble))
      //        .union(testingTriple)
      .filter(b => !testingTripleSet.map(a => (a._1, a._2)).contains((b._1, b._2)))
      .cache()

    //    print("user:" + data.map(_._1).collect().toSet.size )
    //    print("item:" + data.map(_._2).collect().toSet.size )
    //    val writer1 = new PrintWriter(new File("traing.txt"))
    //    data.collect().foreach(a => writer1.write(a._1 + "," + a._2 + "," + a._3 + "\n"))
    //    writer1.close()

    val userMeanRDD = data
      .groupBy(_._1)
      .mapValues(_.toList)
      .mapValues(a => a.map(_._3.toDouble).sum / a.size.toDouble)
      .cache()

    val userMeanMap = userMeanRDD
      .collect().toMap

    val ratingMap = data
      .map(a => (a._1, a._2, a._3.toDouble))   //RDD[(User, Item, Rating)]
      .map(a => (a._1, a))                     //RDD[(User, (User, Item, Rating))]
      .groupByKey()
      .mapValues(a => a.toList.map(b => (b._2, b._3)).toMap)
      .collect()
      .toMap

    val userPairList =
      testingUsers
        .cartesian(testingUsers)
        .filter(a => a._1 != a._2)
      .collect()
      .toList
    val trainingTriple = data.collect().toList
    val m = trainingTriple.map(_._1).toSet.size
    val n = trainingTriple.map(_._2).toSet.size


//    val prediction = testingTriple.map(a => (a._1, a._2, getPredictForPair(a._1, a._2, trainingTriple)))

    //    ratingMap.foreach(a => a._2.foreach(b => println(a._1 + b.toString())))

//    val normalizedRatingMap = data
//      .map(a => (a._1, a._2, a._3.toDouble))   //RDD[(User, Item, Rating)]
//      .map(a => (a._1, a))                     //RDD[(User, (User, Item, Rating))]
//      .join[Double](userMeanRDD)               //RDD[(User, ((User, Item, Rating), meanRating))]
//      .values                                  //RDD[((User, Item, Rating), meanRating)]
//      .map(a => (a._1._1, a._1._2, a._1._3 - a._2))     //RDD[(user, item, normalizedRating)]
//      .groupBy(a => a._1)                       //RDD[(User, Iterable[(User, Item, normalizedRating)])]
//      .mapValues(a => a.map(b => (b._2, b._3)).toMap)     //RDD[(User, Map[Item, normalizedRating])]
//      .collect()
//      .toMap                                       //Map[User, Map[Item, normalizedRating]]

    val elseEmptyMap: Map[Int, Double] = List(0).map(a => (a, a.toDouble)).toMap
    val allUser = data.map(_._1).distinct().cache()
//    val allUserLocal = allUser.collect().toList
    val userNum = allUser.count()
    //    val allItem = data.map(_._2).distinct().cache()

//    val pearsonSize: Int = userNum.toInt
    //    val pearsonSize: Int = 90
    val muturalThreshold = 3
//    val pearsonThreshold = 0.00

    def pearson(a: Int, b:Int, map:Map[Int, Map[Int, Double]]): Double = {
      val ratingMapAAll = map.getOrElse(a, elseEmptyMap)
      val ratingMapBAll = map.getOrElse(b, elseEmptyMap)
      val muturalItems = ratingMapAAll.keys.toSet.intersect(ratingMapBAll.keys.toSet)
      if (muturalItems.size < muturalThreshold) return 0.0    //trust
      //      val ratingsA = ratingMapAAll
      //        .toList
      //        .filter(aa => muturalItems.contains(aa._1))
      //        .map(_._2)
      //
      //      val ratingsB = ratingMapBAll
      //        .toList
      //        .filter(aa => muturalItems.contains(aa._1))
      //        .map(_._2)
      //
      //
      //      val meanA = ratingsA.sum / muturalItems.size.toDouble
      //      val meanB = ratingsB.sum / muturalItems.size.toDouble
      ////      val down = Math.pow(
      ////          ratingsA
      ////          .map(aa => (aa - meanA) * (aa - meanA))
      ////          .sum,
      ////        0.5)* Math.pow(
      ////        ratingsB
      ////          .map(aa => (aa - meanB) * (aa - meanB))
      ////          .sum,
      ////        0.5)
      //      val ratingForCompute = muturalItems
      //        .map(aa => (ratingMapAAll.getOrElse(aa, 3.0) - meanA, ratingMapBAll.getOrElse(aa, 3.0) - meanB))
      //        .filter(aa => aa._1 > 0.1 && aa._2 > 0.1)
      //      val down = Math.pow(
      //        ratingForCompute.map(aa => Math.pow(aa._1, 2)).sum * ratingForCompute.map(aa => Math.pow(aa._2, 2)).sum,
      //        0.5
      //        )
      //      val up = ratingForCompute.map(aa => aa._1 * aa._2).sum
      //      //      println(up)
      //      if (up == 0.0 || down == 0.0) 0.0
      //      else up / down
      // Referenced From
      // https://alvinalexander.com/scala/scala-pearson-correlation-score-algorithm-programming-collective-intelligence
      val p1Ratings = map.getOrElse(a, elseEmptyMap)
      val p2Ratings = map.getOrElse(b, elseEmptyMap)
      val listOfCommon = ratingMapAAll.keys.toSet.intersect(ratingMapBAll.keys.toSet)

      val n = listOfCommon.size
      if (n == 0) return 0.0

      // reduce the maps to only the movies both reviewers have seen
      val p1CommonRatings = p1Ratings.filterKeys(aaa => listOfCommon.contains(aaa))
      val p2CommonRatings = p2Ratings.filterKeys(aaa => listOfCommon.contains(aaa))

      // add up all the preferences
      val sum1 = p1CommonRatings.values.sum
      val sum2 = p2CommonRatings.values.sum

      // sum up the squares
      val sum1Sq = p1CommonRatings.values.foldLeft(0.0)(_ + Math.pow(_, 2))
      val sum2Sq = p2CommonRatings.values.foldLeft(0.0)(_ + Math.pow(_, 2))

      // sum up the products
      val pSum = listOfCommon.foldLeft(0.0)((accum, element) => accum + p1CommonRatings.getOrElse(element, userMeanMap.getOrElse(a, 3.0)) * p2CommonRatings.getOrElse(element, userMeanMap.getOrElse(b, 3.0)))

      // calculate the pearson score
      val numerator = pSum - (sum1*sum2/n)
      val denominator = Math.sqrt( (sum1Sq-Math.pow(sum1,2)/n) * (sum2Sq-Math.pow(sum2,2)/n))
      if (denominator == 0) 0.0 else numerator/denominator
    }
//    val testingUserCorrelations = testingUsers
//      .map(a => (a, allUserLocal))
//
//      .map(a => (a._1, a._2
//        .filter(_ != a._1)
//        //        .filter(bb => userPairSimilaritySet.contains(a._1, bb) || userPairSimilaritySet.contains(bb, a._1))
//        .map(b => (b, pearson(a._1, b, ratingMap)))))
//      .mapValues(_
//        //        .filter(a => Math.abs(a._2) > 0.05)
//        .sortBy(a =>
//        Math.abs(
//          a._2
//        )
//          * -1)
//        //        .take(pearsonSize)
//        .toList)

    //    testingUserCorrelations.filter(_._2.count(_._2 != 0.0) > 10).mapValues(_.take(100)).takeSample(false, 10)
    //      .foreach(a => println(a._1 + "\n" + a._2 + "\n\n\n"))


//    def predictWithItemCorrelationsFun(mapMap: Map[Int, Map[Int, Double]]): ((Int, List[(Int, Double)])) => Double = {
//      def fun(a: (Int, List[(Int, Double)])): Double = {
//        def upInSigma(bb: (Int, Double)): Double = {
//          val thisUserMap = mapMap.getOrElse(bb._1, elseEmptyMap)
//          if (!thisUserMap.keySet.contains(a._1)) return 0.0
//          val rui = thisUserMap.getOrElse(a._1, 3.0)
//          val listOfRatingWithoutThisItem = thisUserMap
//            .toList
//            //            .filter(_._1 != a._1)
//            .map(_._2)
//          val ruMean = listOfRatingWithoutThisItem.sum / listOfRatingWithoutThisItem.size.toDouble
//          (rui
//            - ruMean
//            ) * bb._2
//        }
//        val ratedUserPearson = a._2.filter(b => mapMap.getOrElse(b._1, elseEmptyMap).keySet.contains(a._1)).take(pearsonSize).filter(b => Math.abs(b._2) > pearsonThreshold)
//        val up = ratedUserPearson
//          .map(upInSigma)
//          .sum
//        //        val up = a._2.map(b => normalizedRatingMap.getOrElse(b._1, elseEmptyMap).getOrElse(a._1, 0.0) * b._2).sum
//        val down = ratedUserPearson
//          .map(b =>
//            Math.abs(
//              b._2
//            )
//          )
//          .sum
//        if (up == 0.0 || down == 0.0) 0.0
//        else up / down
//      }
//      fun
//    }
    def getPredictForPair(u0:Int, i0: Int, trainingTriple: List[(Int, Int, Double)], testingUserList: List[Int],  userPairList: List[(Int, Int)], testingItemList: List[Int], ratingMap:Map[Int, Map[Int, Double]]): Double = {
      var oldRating = 3.0
      var newRating = 4.0
      var thisRatingMap = ratingMap
      while (Math.abs(newRating - oldRating) > 0.01) {
        oldRating = newRating
        val userMeanMap =
          thisRatingMap.mapValues(a => a.values.sum / a.size.toDouble)
        val testingUserCorrelationMap =
          userPairList
            .map(a => (a, pearson(a._1, a._2, thisRatingMap)))
            .groupBy(a => a._1._1)
            .mapValues(b =>
              b.groupBy(_._1._2).mapValues(_.map(_._2).head)
            )
        val dMap:Map[Int, Int] = thisRatingMap.mapValues(_.size)
        val cMap:Map[Int, Int] = trainingTriple.groupBy(_._2).mapValues(a => a.size)
        val gama:Double = 3 * (m + n - 1) * dMap.values.sum / (m * n)
//        val testingUserSet = testingUserList.toSet
        for (u <- testingUserList) {
          for (i <- testingItemList) {
            if ( dMap.getOrElse(u,0) + cMap.getOrElse(i,0) > gama
              && thisRatingMap.getOrElse(u, elseEmptyMap).getOrElse(i, -100.0) == -100.0) {
              val suvList = testingUserCorrelationMap.getOrElse(u, elseEmptyMap).toList
              val value = userMeanMap.getOrElse(u, 3.0) + suvList.map(a => a._2 * (thisRatingMap.getOrElse(a._1, elseEmptyMap).getOrElse(i, 3.0) - userMeanMap.getOrElse(a._1, 3.0))).sum / suvList.map(_._2).sum
              thisRatingMap =
                thisRatingMap
                  .updated(
                    u,
                    thisRatingMap
                      .getOrElse(u, elseEmptyMap)
                      .updated(i, value)
                  )
            }
          }
        }
        val suvList = testingUserCorrelationMap.getOrElse(u0, elseEmptyMap).toList
        newRating = userMeanMap.getOrElse(u0, 3.0) + suvList.map(a => a._2 * (thisRatingMap.getOrElse(a._1, elseEmptyMap).getOrElse(i0, 3.0) - userMeanMap.getOrElse(a._1, 3.0))).sum / suvList.map(_._2).sum
      }
      newRating
    }
    val testingUserItemsRating = testingTriple
      .map(a => ((a._1, a._2), getPredictForPair(a._1, a._2,trainingTriple, testingUserList, userPairList, testingItemList, ratingMap)))                 //RDD[(Int, Int)]
//      .join(testingUserCorrelations)          //Rdd[(Int, (Int, List[(Int, Double)])]
//      .map(a => ((a._1, a._2._1), a._2))      //RDD[((User, Item), (Item, List[(Int, Double)]))]
//      .mapValues(predictWithItemCorrelationsFun(ratingMap))  //RDD[((User, Item), normalizedRating)]
      //        .foreach(a => println(a._2))

//    val max = testingUserItemsRating.map(_._2).max()
//    val min = testingUserItemsRating.map(_._2).min()
//
//    def scaleBack(max:Double, min: Double):Double => Double = {
//      val range = max - min
//      def fun(value: Double): Double = {
//        if (value > 5)  5.0
//        else if (value < 1)  1.0
//        else value
//      }
//      fun
//    }
    val testingTupleDouble = testingTriple.map(a => ((a._1, a._2), a._3.toDouble)).cache()
    val ratesAndPreds  = testingTupleDouble
      .join(testingUserItemsRating
//        .mapValues(scaleBack(max, min))
      )
      .map( a => (a._1, a._2, Math.abs(a._2._1 - a._2._2)))
      .cache()




    //output
    val writer = new PrintWriter(new File(outPutFilePath))
    val collectedRatesAndPreds = ratesAndPreds.collect().sortBy(_._1._2).sortBy(_._1._1)
    collectedRatesAndPreds.foreach(a => writer.write(a._1._1 + "," + a._1._2 + "," + a._2._2 + "\n"))
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

    val MSE = collectedRatesAndPreds.map { a => a._3 * a._3
    }.sum / collectedRatesAndPreds.length
    val RMSE = Math.pow(MSE, 0.5)
    println("RMSE = " + RMSE)
    val endTime = System.nanoTime()
    println("Time:" + (endTime - startTime)/1000000000 + " sec")
  }
}
