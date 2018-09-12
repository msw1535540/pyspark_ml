# -*- coding:utf-8 -*-
import re
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import RegressionMetrics
import os
os.environ['JAVA_HOME'] = 'F:/tools/java'
os.environ.setdefault("HADOOP_CONF_DIR", "F:/tools/Spark/hadoop-2.7.5")
os.environ.setdefault("HADOOP_USER_NAME", "hdfs")
os.environ.setdefault("YARN_CONF_DIR", "F:/tools/Spark/hadoop_config")
os.environ.setdefault("SPARK_CLASSPATH", "F:/tools/Spark/spark-2.1.0-bin-hadoop2.7/jars")
os.environ['SPARK_HOME'] = "F:/tools/Spark/spark-2.1.0-bin-hadoop2.7"
os.environ['PYTHONPATH'] = "F:/tools/Spark/spark-2.1.0-bin-hadoop2.7/python/lib/py4j"
os.environ['PYTHONPATH'] = "F:/tools/Spark/spark-2.1.0-bin-hadoop2.7/python/lib/pyspark"

if __name__ == '__main__':
    # 读数据
    conf = SparkConf().setMaster("local[*]").setAppName("First_App")
    sparkSession = (SparkSession.builder
                    .config(conf=conf)
                    .getOrCreate())
    print "读取与载入数据..."
    trainInput = (sparkSession.read
                  .option("header", "true")
                  .option("inferSchema", "true")
                  .csv("train.csv")
                  .cache())
    testInput = (sparkSession.read
                 .option("header", "true")
                 .option("inferSchema", "true")
                 .csv("test.csv")
                 .cache())
    print "数据载入完毕..."

    # 训练数据拆分和持久化
    data = trainInput.withColumnRenamed("loss", "label")
    [trainingData, validationData] = data.randomSplit([0.7, 0.3])
    trainingData.persist()
    validationData.persist()

    # 测试数据持久化
    testData = testInput.persist()

    print("数据与特征处理...")
    print("对类别型的特征进行处理...")
    # 对类别型的列用StringIndexer或者OneHotEncoder
    isCateg = lambda c: c.startswith("cat")
    categNewCol = lambda c: "idx_{0}".format(c) if (isCateg(c)) else c

    stringIndexerStages = map(lambda c: StringIndexer(inputCol=c, outputCol=categNewCol(c))
                              .fit(trainInput.select(c).union(testInput.select(c))),
                              filter(isCateg, trainingData.columns))

    # 干掉特别特别多类别的列(类似ID列)
    removeTooManyCategs = lambda c: not re.match(r"cat(109$|110$|112$|113$|116$)", c)

    # 只保留特征列
    onlyFeatureCols = lambda c: not re.match(r"id|label", c)

    # 用上述函数进行过滤
    featureCols = map(categNewCol,
                      filter(onlyFeatureCols,
                             filter(removeTooManyCategs,
                                    trainingData.columns)))
    # 组装特征(将多列数据转化为单列的向量列)
    assembler = VectorAssembler(inputCols=featureCols, outputCol="features")
    print("特征生成和组装完毕...")

    print("构建随机森林进行回归预测...")
    # 使用随机森林进行回归
    algo = RandomForestRegressor(featuresCol="features", labelCol="label")

    stages = stringIndexerStages
    stages.append(assembler)
    stages.append(algo)

    # 构建pipeline
    pipeline = Pipeline(stages=stages)

    print("K折交叉验证...")
    numTrees = [5, 20]
    maxDepth = [4, 6]
    maxBins = [32]
    numFolds = 3

    paramGrid = (ParamGridBuilder()
                 .addGrid(algo.numTrees, numTrees)
                 .addGrid(algo.maxDepth, maxDepth)
                 .addGrid(algo.maxBins, maxBins)
                 .build())

    cv = CrossValidator(estimator=pipeline,
                        evaluator=RegressionEvaluator(),
                        estimatorParamMaps=paramGrid,
                        numFolds=numFolds)

    cvModel = cv.fit(trainingData)

    trainPredictionsAndLabels = cvModel.transform(trainingData).select("label", "prediction").rdd

    validPredictionsAndLabels = cvModel.transform(validationData).select("label", "prediction").rdd

    trainRegressionMetrics = RegressionMetrics(trainPredictionsAndLabels)
    validRegressionMetrics = RegressionMetrics(validPredictionsAndLabels)

    bestModel = cvModel.bestModel
    featureImportances = bestModel.stages[-1].featureImportances.toArray()

    print("TrainingData count: {0}".format(trainingData.count()))
    print("ValidationData count: {0}".format(validationData.count()))
    print("TestData count: {0}".format(testData.count()))
    print("=====================================================================")
    print("Param algoNumTrees = {0}".format(",".join(map(lambda x: str(x), numTrees))))
    print("Param algoMaxDepth = {0}".format(",".join(map(lambda x: str(x), maxDepth))))
    print("Param algoMaxBins = {0}".format(",".join(map(lambda x: str(x), maxBins))))
    print("Param numFolds = {0}".format(numFolds))
    print("=====================================================================\n")
    # print("Training data MSE = {0}".format(trainRegressionMetrics.meanSquaredError))
    # print("Training data RMSE = {0}".format(trainRegressionMetrics.rootMeanSquaredError))
    # print("Training data R-squared = {0}".format(trainRegressionMetrics.r2))
    # print("Training data MAE = {0}".format(trainRegressionMetrics.meanAbsoluteError))
    # print("Training data Explained variance = {0}".format(trainRegressionMetrics.explainedVariance))
    # print("=====================================================================\n")
    # print("Validation data MSE = {0}".format(validRegressionMetrics.meanSquaredError))
    # print("Validation data RMSE = {0}".format(validRegressionMetrics.rootMeanSquaredError))
    # print("Validation data R-squared = {0}".format(validRegressionMetrics.r2))
    # print("Validation data MAE = {0}".format(validRegressionMetrics.meanAbsoluteError))
    # print("Validation data Explained variance = {0}".format(validRegressionMetrics.explainedVariance))
    # print("=====================================================================\n")
    # print("特征重要度:\n{0}\n".format(
    #     "\n".join(map(lambda z: "{0} = {1}".format(str(z[0]), str(z[1])), zip(featureCols, featureImportances)))))
    #
    # cvModel.transform(testData) \
    #     .select("id", "prediction") \
    #     .withColumnRenamed("prediction", "loss") \
    #     .coalesce(1) \
    #     .write.format("csv") \
    #     .option("header", "true") \
    #     .save("rf_sub.csv")

    "====================================================="
    # 与GBDT拟合
    "====================================================="
    # from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
    #
    # algo2 = GBTRegressor(featuresCol="features", labelCol="label")
    #
    # stages2 = stringIndexerStages
    # stages2.append(assembler)
    # stages2.append(algo2)
    #
    # pipeline2 = Pipeline(stages=stages2)
    #
    # print("K折交叉验证...")
    # numTrees = [5, 20]
    # maxDepth = [4, 6]
    # maxBins = [32]
    # numFolds = 3
    #
    # paramGrid = (ParamGridBuilder()
    #              .addGrid(algo.numTrees, numTrees)
    #              .addGrid(algo.maxDepth, maxDepth)
    #              .addGrid(algo.maxBins, maxBins)
    #              .build())
    #
    # cv = CrossValidator(estimator=pipeline2,
    #                     evaluator=RegressionEvaluator(),
    #                     estimatorParamMaps=paramGrid,
    #                     numFolds=numFolds)
    #
    # cvModel = cv.fit(trainingData)
    #
    # trainPredictionsAndLabels = cvModel.transform(trainingData).select("label", "prediction").rdd
    #
    # validPredictionsAndLabels = cvModel.transform(validationData).select("label", "prediction").rdd
    #
    # trainRegressionMetrics = RegressionMetrics(trainPredictionsAndLabels)
    # validRegressionMetrics = RegressionMetrics(validPredictionsAndLabels)
    #
    # bestModel = cvModel.bestModel
    # featureImportances = bestModel.stages[-1].featureImportances.toArray()
    #
    # print("TrainingData count: {0}".format(trainingData.count()))
    # print("ValidationData count: {0}".format(validationData.count()))
    # print("TestData count: {0}".format(testData.count()))
    # print("=====================================================================")
    # print("Param algoNumTrees = {0}".format(",".join(map(lambda x: str(x), numTrees))))
    # print("Param algoMaxDepth = {0}".format(",".join(map(lambda x: str(x), maxDepth))))
    # print("Param algoMaxBins = {0}".format(",".join(map(lambda x: str(x), maxBins))))
    # print("Param numFolds = {0}".format(numFolds))
    # print("=====================================================================\n")
    # print("Training data MSE = {0}".format(trainRegressionMetrics.meanSquaredError))
    # print("Training data RMSE = {0}".format(trainRegressionMetrics.rootMeanSquaredError))
    # print("Training data R-squared = {0}".format(trainRegressionMetrics.r2))
    # print("Training data MAE = {0}".format(trainRegressionMetrics.meanAbsoluteError))
    # print("Training data Explained variance = {0}".format(trainRegressionMetrics.explainedVariance))
    # print("=====================================================================\n")
    # print("Validation data MSE = {0}".format(validRegressionMetrics.meanSquaredError))
    # print("Validation data RMSE = {0}".format(validRegressionMetrics.rootMeanSquaredError))
    # print("Validation data R-squared = {0}".format(validRegressionMetrics.r2))
    # print("Validation data MAE = {0}".format(validRegressionMetrics.meanAbsoluteError))
    # print("Validation data Explained variance = {0}".format(validRegressionMetrics.explainedVariance))
    # print("=====================================================================\n")
    # print("特征重要度:\n{0}\n".format(
    #     "\n".join(map(lambda z: "{0} = {1}".format(str(z[0]), str(z[1])), zip(featureCols, featureImportances)))))















