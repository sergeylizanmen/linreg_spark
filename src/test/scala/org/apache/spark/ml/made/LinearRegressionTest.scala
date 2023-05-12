package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.scalatest.flatspec._
import org.scalatest.matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  lazy val deltaPred: Double = 0.00000001
  lazy val deltaParams: Double = 0.0001

  lazy val nRows: Int = LinearRegressionTest._nRows
  lazy val weights: DenseVector[Double] = LinearRegressionTest._weights
  lazy val bias: DenseVector[Double]  = LinearRegressionTest._bias
  lazy val y: DenseVector[Double] = LinearRegressionTest._y
  lazy val data: DataFrame = LinearRegressionTest._dataFrame

  "Model" should "make predictions" in {

    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.fromBreeze(weights).toDense,
      bias = Vectors.fromBreeze(bias).toDense
    ).setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")

    validateModel(model.transform(data))
  }

  "Estimator" should "calculate weights and bias" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setMaxIter(1000)
      .setStepSize(1.0)

    val model = estimator.fit(data)

    validateEstimator(model)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("y")
        .setPredictionCol("prediction")
        .setMaxIter(1000)
        .setStepSize(1.0)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: Pipeline = Pipeline.load(tmpFolder.getAbsolutePath)

    val model: LinearRegressionModel = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateEstimator(model)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("y")
        .setPredictionCol("prediction")
        .setMaxIter(1000)
        .setStepSize(1.0)
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reRead.transform(data))

  }

  private def validateModel(data: DataFrame): Unit = {

    val predictions = data.collect().map(_.getAs[Double](1))

    predictions.length should be(nRows)
    for (i <- 0 until predictions.length - 1) {
      predictions(i) should be(y(i) +- deltaPred)
    }

  }

  private def validateEstimator(model: LinearRegressionModel) = {

    val parameters = model.weights

    parameters.size should be(weights.size)
    parameters(0) should be(weights(0) +- deltaParams)
    parameters(1) should be(weights(1) +- deltaParams)
    parameters(2) should be(weights(2) +- deltaParams)
    model.bias(0) should be(bias(0) +- deltaParams)

  }
}

object LinearRegressionTest extends WithSpark {
  lazy val _nRows: Int = 10000
  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand(_nRows, 3)
  lazy val _weights: DenseVector[Double] = DenseVector(0.5, -0.1, 0.2)
  lazy val _bias: DenseVector[Double] = DenseVector.fill(_nRows)(0.2)
  lazy val _y: DenseVector[Double] = _X * _weights + _bias + DenseVector.rand(_nRows) * 0.0001
  lazy val _dataFrame: DataFrame = toDataFrame(_X, _y)

  def toDataFrame(X: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {

    import sqlc.implicits._

    lazy val data: DenseMatrix[Double] = DenseMatrix.horzcat(X, y.asDenseMatrix.t)

    lazy val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq.toDF("x1", "x2", "x3", "y")

    lazy val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")

    assembler.transform(df).select("features", "y")
  }

}