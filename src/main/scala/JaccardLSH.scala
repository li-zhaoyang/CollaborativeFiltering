import org.apache.spark.{SparkConf, SparkContext}
import java.io.{File, PrintWriter}
import scala.io.Source
import scala.util.Random
object JaccardLSH {
  def mapStrToKVPair(s: String): (Int, Int) = {
    val sArray = s.split(",")
    (sArray(0).toInt, sArray(1).toInt)
  }

  def main(args: Array[String]): Unit = {
    val inputPath: String = args(0)
    val outputFilePath: String = args(1)
    val inputFilePath = inputPath
//    val truthFilePath = "Data\\video_small_ground_truth_jaccard.csv"
//    val outputFilePath = "outputJaccard.txt"
    val conf = new SparkConf()
    conf.setAppName("hw3")
    conf.setMaster("local[4]")
    val sc = new SparkContext(conf)


    val bands = 33
    val row = 5
    val hashFunNum: Int = bands * row
    val threshold = 0.5
    // 33, 5, seed Num

    val fileRDD = sc.textFile(inputFilePath)   //By default one partiton per 128MB of file
    val header = fileRDD.first
    val rawKVPair =
      fileRDD
      .filter(row => row != header)
      .map(mapStrToKVPair)

    val indexedUserItemSet =
      rawKVPair
      .groupByKey()
      .map(a => (a._1, a._2.toList))
      .cache()

    val localItemUserSetMap:Map[Int, List[Int]] = rawKVPair
      .groupBy(_._2)
      .map(a => (a._1, a._2.map(_._1).toList))
      .cache()
      .collect()
      .toMap

    val userSize: Long = indexedUserItemSet.count()
    def numToFunFunction(num: Int): (Int, Long => Long) = {
      val random: Random = new Random(num)
      val a = random.nextInt() % 10000 + 10000
      val b = random.nextInt() % 10000 + 10000
      def hashFun(index: Long): Long = {
//        (a * index + b) % userSize
        ((a * index + b) % 12251) % userSize
      }
      (num, hashFun)
    }
    val integers = List.range(0, hashFunNum, 1)
    val hashFunsLocal = integers.map(numToFunFunction)
    val hashFunctions = sc.parallelize( hashFunsLocal )
    val indexItemListHashFunJoin = indexedUserItemSet.map(a => (1, a))
      .join[(Int, Long => Long)](hashFunctions.map(b => (1, b)))
      .map(a => (a._2._1._1, a._2._1._2, a._2._2._1, a._2._2._2))      //RDD[index, ItemList, hashFunIdx, hashFun]

    def indexItemListHashFunJoinToListsOfItemHashVals(a: (Int, List[Int], Int, Long => Long)): List[(Int, Int, Long)] = {
      a._2.map(s => (a._3, s, a._4(a._1)))
    }

    val indexItemHashValuePair = indexItemListHashFunJoin
      .map(indexItemListHashFunJoinToListsOfItemHashVals)       //RDD[List[int, itemStr, hashValue]]
      .flatMap(a => a)    //RDD[int, ItemStr, hashValue]
      .groupBy(a => a._1) // RDD[int, Iterable[int, ItemStr, hashValue]]
      .mapValues(_
        .toList
        .map(a => (a._2, a._3))
        .groupBy(_._1)
        .mapValues(_.map(_._2).min))           //RDD[int, List[ItemStr, minHashValue]]
      .flatMapValues(a => a)        //RDD[(index, (ItemStr, minHashValue))]
//      .cache()


    val itemHashValuesMap = indexItemHashValuePair
      .groupBy(_._2._1)             //RDD[(ItemStr, iterable(index, (ItemStr, minHashValue)))]
      .mapValues(_.map(b => (b._1, b._2._2))
        .toList
        .sortBy(_._1)
        .map(_._2))
      .cache()

    def mapListToListOfBandedList(bandNum: Int): List[Long] => List[(Int, List[Long])] = {
      def fun(list: List[Long]): List[(Int, List[Long])] = {
        list.zipWithIndex.groupBy(_._2 / row).toList.map(a => (a._1, a._2.map(_._1)))
      }
      fun
    }

    def itemsToPairs(list: List[Int]): List[(Int, Int)] = {
      for(a <- list; b <- list)
        yield (a, b)
    }
    val candidatePairs = itemHashValuesMap         //RDD[Item, List[Long]]
      .flatMapValues(a => mapListToListOfBandedList(bands)(a))    //RDD[Item, (Int, List[Long])]
      .groupBy(_._2._1)    //RDD[Int, Iterable[(Item, (Int, List[Long]))]
      .mapValues(a => a    // value: Iterable[(Item, (Int, List[Long]))
        .map(b => (b._1, b._2._2))     //Iterable[(Item, List[Long])]
        .groupBy(_._2)                 //Map[ List[Long], Iterable[(Item, List[Long])] ]
        .filter(_._2.size > 1)         //Map[ List[Long], Iterable[(Item, List[Long])] ]
        .mapValues(e => e.map(_._1))   //Map[ List[Long], Iterable[Item] ]
        .values                        //Iterable[Iterable[Item]]
        .toList                        //List[Iterable[Item]
        .flatMap(c => itemsToPairs(c.toList)
          .filter(d => d._1 < d._2)
        )                               //List[(Item, Item)]
        .toSet
      )                                 //RDD[(Int, Set[(Item, Item)])]
      .values
      .flatMap(a => a)
      .distinct


    def listsToJaccard(list1: List[Int], list2: List[Int]): Double = {
      val up = list1.intersect(list2).size.toDouble
      val down: Double = list1.union(list2).toSet.size.toDouble
      if (up == 0.0 || down == 0.0)  0.0
      else up / down
    }


    val elseListInt: List[Int] = (0 until hashFunNum).toList

    val itemPairSimilarity = candidatePairs
      .map(a => (
        a._1,
        a._2,
        listsToJaccard(
          localItemUserSetMap.getOrElse(a._1, elseListInt),
          localItemUserSetMap.getOrElse(a._2, elseListInt)
        )
      ))
      .filter(_._3 >= threshold)



    val writer = new PrintWriter(new File(outputFilePath))
    itemPairSimilarity.collect().toList.sortBy(_._2).sortBy(_._1).foreach(a => writer.write(a._1 + "," + a._2 + "," + a._3 + "\n"))
    writer.close()

//    def mapStringToSortedTuple(a: Array[String]): (String, String) = {
//      if (a(0) < a(1))
//        (a(0), a(1))
//      else
//        (a(1), a(0))
//    }
//    val truth = Source.fromFile(truthFilePath).getLines.map(_.split(",")).filter(_.length > 0).map(mapStringToSortedTuple).toSet
//    val output = Source.fromFile(outputFilePath).getLines().map(_.split(",")).filter(_.length > 0).map(mapStringToSortedTuple).toSet
////    val tp = truth.count(output.contains)
//    val tp = truth.intersect(output).size
//    val fp = output.size - tp
//    val fn = truth.size - tp
//    println("precision: " + tp.toDouble / (tp + fp).toDouble)
//    println("recall: " + tp.toDouble / (tp + fn).toDouble)
  }
}
