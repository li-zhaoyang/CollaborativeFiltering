import java.io.{File, PrintWriter}


import com.soundcloud.lsh.Lsh
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

import scala.io.Source

object CosineLSH {
  def main(args: Array[String]): Unit = {
//    val inputPath1: String = args(0)
//    val inputPath2: String = args(1)
//    val inputPath3: String = args(2)
    val inputFilePath = "Data\\video_small_num.csv"
    val truthFilePath = "Data\\video_small_ground_truth_cosine.csv"
    val outPutFilePath = "Data\\Zhaoyang_Li_SimilarProducts_Cosine.txt"


    val conf = new SparkConf()
    conf.setAppName("hw3")
    conf.setMaster("local[2]")
    val sc = new SparkContext(conf)
    val storageLevel = StorageLevel.MEMORY_AND_DISK
    val numPartitions = 2

    val header = Source
      .fromFile(inputFilePath)
      .getLines()
      .toList
      .head

    val set = Source
      .fromFile(inputFilePath)
      .getLines()
      .filter(_ != header)
      .map(_.split(","))
      .map(a => (a(0), a(1)))
      .toSet

    val allUser = sc.textFile(inputFilePath)
      .filter(_ != header)
      .map(_.split(","))
      .map(a => a(0))
      .distinct()
      .persist()

    val allItem = sc.textFile(inputFilePath)
      .filter(_ != header)
      .map(_.split(","))
      .map(a => a(1))
      .distinct()
      .persist()
    val itemNum = allItem.count()


    def itemToRatingFunFromUserMap(item: String, set: Set[(String, String)]): String => Double = {
      def fun(user: String): Double = {
        var ans = 0.0
        if (set.contains((user, item))) ans = 1.0
        ans
      }
      fun
    }

    val data = allItem.map((1, _))
      .join[String](allUser.map((1, _)))
      .map(_._2)
      .groupByKey()
      .map(a => (a._1, a._2.toArray.sorted.map(itemToRatingFunFromUserMap(a._1, set))))
    val indexed = data.zipWithIndex.persist(storageLevel)
    val rows = indexed.map {
        a => IndexedRow(a._2, Vectors.dense(a._1._2))
    }
    val index = indexed.map {
      a => (a._2, a._1._1)
    }.persist(storageLevel)

    val matrix = new IndexedRowMatrix(rows)
    val lsh = new Lsh(
      minCosineSimilarity = 0.47,
      dimensions = 150,
      numNeighbours = 150,
      numPermutations = 10,
      partitions = numPartitions,
      storageLevel = storageLevel
    )
    val similarityMatrix = lsh.join(matrix)
    val remapFirst = similarityMatrix.entries.keyBy(_.i).join(index).values
    val remapSecond = remapFirst.keyBy { case (entry, word1) => entry.j }.join(index).values.map {
      case ((entry, word1), word2) =>
        (word1, word2, entry.value)
    }

    def mapStringToSortedTuple(a: Array[String]): (String, String) = {
      if (a(0) < a(1))
        (a(0), a(1))
      else
        (a(1), a(0))
    }

    val writer = new PrintWriter(new File(outPutFilePath))
    remapSecond
      .collect()
      .map{ a =>
          if (a._1 > a._2) (a._2, a._1, a._3)
          else (a._1, a._2, a._3)
      }
      .map(a => ((a._1, a._2), a._3))
      .groupBy(_._1)
      .map(a => (a._1._1, a._1._2, a._2(0)._2))
      .toArray
      .sortBy(_._2)
      .sortBy(_._1)
      .foreach(a => writer.write(a._1 + "," + a._2 + "," + a._3 + "\n"))
    writer.close()

    val truth = Source.fromFile(truthFilePath).getLines.map(_.split(",")).filter(_.length > 0).map(mapStringToSortedTuple).toSet
    val output = Source.fromFile(outPutFilePath).getLines().map(_.split(",")).filter(_.length > 0).map(mapStringToSortedTuple).toSet
    val tp = truth.count(output.contains)
    val fp = output.size - tp
    val fn = truth.size - tp
    println("precision: " + tp.toDouble / (tp + fp).toDouble)
    println("recall: " + tp.toDouble / (tp + fn).toDouble)

  }

}
