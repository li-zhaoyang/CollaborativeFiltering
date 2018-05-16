import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._

import scala.util.Random
//import spark.implicits._
//https://stackoverflow.com/questions/46856014/the-proper-way-to-compute-correlation-between-two-seq-columns-into-a-third-colum
import scala.io.Source
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

object UserBasedCF {
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
//    val outPutFilePath = "outputUserCF.txt"
    val conf = new SparkConf()
    conf.setAppName("hw3")
    conf.setMaster("local[4]")
    val sc = new SparkContext(conf)


    //Jaccard

//    val bands = 33
//    val row = 5
//    val hashFunNum: Int = bands * row
//    val threshold = 0.8
//    // 33, 5, seed Num
//    def mapStrToKVPair(s: String): (Int, Int) = {
//      val sArray = s.split(",")
//      (sArray(1).toInt, sArray(0).toInt)
//    }
//
//    val fileRDD = sc.textFile(inputFilePath)   //By default one partiton per 128MB of file
//    val header = fileRDD.first
//    val rawKVPair =
//      fileRDD
//        .filter(row => row != header)
//        .map(mapStrToKVPair)
//
//    val indexedUserItemSet =
//      rawKVPair
//        .groupByKey()
//        .map(a => (a._1, a._2.toList))
//        .cache()
//
//    val localItemUserSetMap:Map[Int, List[Int]] = rawKVPair
//      .groupBy(_._2)
//      .map(a => (a._1, a._2.map(_._1).toList))
//      .cache()
//      .collect()
//      .toMap
//
//    val userSize: Long = indexedUserItemSet.count()
//    def numToFunFunction(num: Int): (Int, Long => Long) = {
//      val random: Random = new Random(num)
//      val a = random.nextInt() % 10000 + 10000
//      val b = random.nextInt() % 10000 + 10000
//      def hashFun(index: Long): Long = {
//        //        (a * index + b) % userSize
//        ((a * index + b) % 12251) % userSize
//      }
//      (num, hashFun)
//    }
//    val integers = List.range(0, hashFunNum, 1)
//    val hashFunsLocal = integers.map(numToFunFunction)
//    val hashFunctions = sc.parallelize( hashFunsLocal )
//    val indexItemListHashFunJoin = indexedUserItemSet.map(a => (1, a))
//      .join[(Int, Long => Long)](hashFunctions.map(b => (1, b)))
//      .map(a => (a._2._1._1, a._2._1._2, a._2._2._1, a._2._2._2))      //RDD[index, ItemList, hashFunIdx, hashFun]
//
//    def indexItemListHashFunJoinToListsOfItemHashVals(a: (Int, List[Int], Int, Long => Long)): List[(Int, Int, Long)] = {
//      a._2.map(s => (a._3, s, a._4(a._1)))
//    }
//
//    val indexItemHashValuePair = indexItemListHashFunJoin
//      .map(indexItemListHashFunJoinToListsOfItemHashVals)       //RDD[List[int, itemStr, hashValue]]
//      .flatMap(a => a)    //RDD[int, ItemStr, hashValue]
//      .groupBy(a => a._1) // RDD[int, Iterable[int, ItemStr, hashValue]]
//      .mapValues(_
//      .toList
//      .map(a => (a._2, a._3))
//      .groupBy(_._1)
//      .mapValues(_.map(_._2).min))           //RDD[int, List[ItemStr, minHashValue]]
//      .flatMapValues(a => a)        //RDD[(index, (ItemStr, minHashValue))]
//    //      .cache()
//
//
//    val itemHashValuesMap = indexItemHashValuePair
//      .groupBy(_._2._1)             //RDD[(ItemStr, iterable(index, (ItemStr, minHashValue)))]
//      .mapValues(_.map(b => (b._1, b._2._2))
//      .toList
//      .sortBy(_._1)
//      .map(_._2))
//      .cache()
//
//    def mapListToListOfBandedList(bandNum: Int): List[Long] => List[(Int, List[Long])] = {
//      def fun(list: List[Long]): List[(Int, List[Long])] = {
//        list.zipWithIndex.groupBy(_._2 / row).toList.map(a => (a._1, a._2.map(_._1)))
//      }
//      fun
//    }
//
//    def itemsToPairs(list: List[Int]): List[(Int, Int)] = {
//      for(a <- list; b <- list)
//        yield (a, b)
//    }
//    val candidatePairs = itemHashValuesMap         //RDD[Item, List[Long]]
//      .flatMapValues(a => mapListToListOfBandedList(bands)(a))    //RDD[Item, (Int, List[Long])]
//      .groupBy(_._2._1)    //RDD[Int, Iterable[(Item, (Int, List[Long]))]
//      .mapValues(a => a    // value: Iterable[(Item, (Int, List[Long]))
//      .map(b => (b._1, b._2._2))     //Iterable[(Item, List[Long])]
//      .groupBy(_._2)                 //Map[ List[Long], Iterable[(Item, List[Long])] ]
//      .filter(_._2.size > 1)         //Map[ List[Long], Iterable[(Item, List[Long])] ]
//      .mapValues(e => e.map(_._1))   //Map[ List[Long], Iterable[Item] ]
//      .values                        //Iterable[Iterable[Item]]
//      .toList                        //List[Iterable[Item]
//      .flatMap(c => itemsToPairs(c.toList)
//      .filter(d => d._1 < d._2)
//    )                               //List[(Item, Item)]
//      .toSet
//    )                                 //RDD[(Int, Set[(Item, Item)])]
//      .values
//      .flatMap(a => a)
//      .distinct
//
//
//    def listsToJaccard(list1: List[Int], list2: List[Int]): Double = {
//      val up = list1.intersect(list2).size.toDouble
//      val down: Double = list1.union(list2).toSet.size.toDouble
//      if (up == 0.0 || down == 0.0)  0.0
//      else up / down
//    }
//
//
//    val elseListInt: List[Int] = (0 until hashFunNum).toList
//
//    val itemPairSimilarity = candidatePairs
//      .map(a => (
//        a._1,
//        a._2,
//        listsToJaccard(
//          localItemUserSetMap.getOrElse(a._1, elseListInt),
//          localItemUserSetMap.getOrElse(a._2, elseListInt)
//        )
//      ))
//      .filter(_._3 >= threshold)
//    val userPairSimilaritySet = itemPairSimilarity
//      .map(a => (a._1, a._2))
//      .collect()
//      .toSet

    //End ofaccard

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


    val data = sc.textFile(inputFilePath)
      .filter(_ != headerheader )
      .map(_.split(','))
      .map(a => (a(0).toInt, a(1).toInt, a(2)))
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

//    ratingMap.foreach(a => a._2.foreach(b => println(a._1 + b.toString())))

    val normalizedRatingMap = data
      .map(a => (a._1, a._2, a._3.toDouble))   //RDD[(User, Item, Rating)]
      .map(a => (a._1, a))                     //RDD[(User, (User, Item, Rating))]
      .join[Double](userMeanRDD)               //RDD[(User, ((User, Item, Rating), meanRating))]
      .values                                  //RDD[((User, Item, Rating), meanRating)]
      .map(a => (a._1._1, a._1._2, a._1._3 - a._2))     //RDD[(user, item, normalizedRating)]
      .groupBy(a => a._1)                       //RDD[(User, Iterable[(User, Item, normalizedRating)])]
      .mapValues(a => a.map(b => (b._2, b._3)).toMap)     //RDD[(User, Map[Item, normalizedRating])]
      .collect()
      .toMap                                       //Map[User, Map[Item, normalizedRating]]

    val elseEmptyMap: Map[Int, Double] = List(0).map(a => (a, a.toDouble)).toMap
    val allUser = data.map(_._1).distinct().cache()
    val allUserLocal = allUser.collect().toList
    val userNum = allUser.count()
//    val allItem = data.map(_._2).distinct().cache()

    val pearsonSize: Int = userNum.toInt
//    val pearsonSize: Int = 90
    val mutrualThreshold = 1
    val pearsonThreshold = 0.00

    def pearson(a: Int, b:Int, map:Map[Int, Map[Int, Double]]): Double = {
      val ratingMapAAll = map.getOrElse(a, elseEmptyMap)
      val ratingMapBAll = map.getOrElse(b, elseEmptyMap)
      val muturalItems = ratingMapAAll.keys.toSet.intersect(ratingMapBAll.keys.toSet)
      if (muturalItems.size < mutrualThreshold) return 0.0    //trust
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
    val testingUserCorrelations = testingUsers
      .map(a => (a, allUserLocal))

      .map(a => (a._1, a._2
        .filter(_ != a._1)
//        .filter(bb => userPairSimilaritySet.contains(a._1, bb) || userPairSimilaritySet.contains(bb, a._1))
        .map(b => (b, pearson(a._1, b, ratingMap)))))
      .mapValues(_
//        .filter(a => Math.abs(a._2) > 0.05)
        .sortBy(a =>
        Math.abs(
          a._2
        )
          * -1)
//        .take(pearsonSize)
        .toList)

//    testingUserCorrelations.filter(_._2.count(_._2 != 0.0) > 10).mapValues(_.take(100)).takeSample(false, 10)
//      .foreach(a => println(a._1 + "\n" + a._2 + "\n\n\n"))


    def predictWithItemCorrelationsFun(mapMap: Map[Int, Map[Int, Double]]): ((Int, List[(Int, Double)])) => Double = {
      def fun(a: (Int, List[(Int, Double)])): Double = {
        def upInSigma(bb: (Int, Double)): Double = {
          val thisUserMap = mapMap.getOrElse(bb._1, elseEmptyMap)
          if (!thisUserMap.keySet.contains(a._1)) return 0.0
          val rui = thisUserMap.getOrElse(a._1, 3.0)
          val listOfRatingWithoutThisItem = thisUserMap
            .toList
//            .filter(_._1 != a._1)
            .map(_._2)
          val ruMean = listOfRatingWithoutThisItem.sum / listOfRatingWithoutThisItem.size.toDouble
          (rui
            - ruMean
            ) * bb._2
        }
        val ratedUserPearson = a._2.filter(b => mapMap.getOrElse(b._1, elseEmptyMap).keySet.contains(a._1)).take(pearsonSize).filter(b => Math.abs(b._2) > pearsonThreshold)
        val up = ratedUserPearson
          .map(upInSigma)
          .sum
//        val up = a._2.map(b => normalizedRatingMap.getOrElse(b._1, elseEmptyMap).getOrElse(a._1, 0.0) * b._2).sum
        val down = ratedUserPearson
          .map(b =>
          Math.abs(
            b._2
            )
          )
          .sum
        if (up == 0.0 || down == 0.0) 0.0
        else up / down
      }
      fun
    }
    val testingUserItemsRating = testingTriple
      .map(a => (a._1, a._2))                 //RDD[(Int, Int)]
      .join(testingUserCorrelations)          //Rdd[(Int, (Int, List[(Int, Double)])]
      .map(a => ((a._1, a._2._1), a._2))      //RDD[((User, Item), (Item, List[(Int, Double)]))]
      .mapValues(predictWithItemCorrelationsFun(ratingMap))  //RDD[((User, Item), normalizedRating)]
//        .foreach(a => println(a._2))
      .map( a=> (a._1, a._2
//          + 3.9
        + userMeanMap.getOrElse(a._1._1, 3.0)
      ))

    val max = testingUserItemsRating.map(_._2).max()
    val min = testingUserItemsRating.map(_._2).min()

    def scaleBack(max:Double, min: Double):Double => Double = {
      val range = max - min
      def fun(value: Double): Double = {
        if (value > 5)  5.0
        else if (value < 1)  1.0
        else value
      }
      fun
    }
    val testingTupleDouble = testingTriple.map(a => ((a._1, a._2), a._3.toDouble)).cache()
    val ratesAndPreds  = testingTupleDouble
      .join(testingUserItemsRating
        .mapValues(scaleBack(max, min))
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
