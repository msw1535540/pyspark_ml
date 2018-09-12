# -*- coding:utf-8 -*-

import os
import urllib
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

os.environ['JAVA_HOME'] = 'F:/tools/java'
os.environ.setdefault("HADOOP_CONF_DIR", "F:/tools/Spark/hadoop-2.7.5")
os.environ.setdefault("HADOOP_USER_NAME", "hdfs")
os.environ.setdefault("YARN_CONF_DIR", "F:/tools/Spark/hadoop_config")
os.environ.setdefault("SPARK_CLASSPATH", "F:/tools/Spark/spark-2.1.0-bin-hadoop2.7/jars")
os.environ['SPARK_HOME'] = "F:/tools/Spark/spark-2.1.0-bin-hadoop2.7"
os.environ['PYTHONPATH'] = "F:/tools/Spark/spark-2.1.0-bin-hadoop2.7/python/lib/py4j"
os.environ['PYTHONPATH'] = "F:/tools/Spark/spark-2.1.0-bin-hadoop2.7/python/lib/pyspark"


def download_dateset():
    """下载数据集"""
    complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
    small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    datasets_path = os.path.join('F:/tmp', 'datasets')

    complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
    small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')
    # 下载数据集
    small_f = urllib.urlretrieve(small_dataset_url, small_dataset_path)
    complete_f = urllib.urlretrieve(complete_dataset_url, complete_dataset_path)

    # 解压缩数据
    import zipfile

    with zipfile.ZipFile(small_dataset_path, "r") as z:
        z.extractall(datasets_path)

    with zipfile.ZipFile(complete_dataset_path, "r") as z:
        z.extractall(datasets_path)


def small_dataset_model():
    small_ratings_file = "/user/test/datasets/ml-latest-small/ratings.csv"  # file:///F:/tmp/datasets/ml-latest-small
    small_ratings_raw_data = sc.textFile(small_ratings_file, 2000)
    small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

    # 解析数据生成新的RDD
    small_ratings_data = small_ratings_raw_data.filter(lambda line: line != small_ratings_raw_data_header)\
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1], tokens[2])).persist()
    # print small_ratings_data.take(10)

    # 处理movies.csv数据
    small_movies_file = "/user/test/datasets/ml-latest-small/movies.csv"  # file:///F:/tmp/datasets/ml-latest-small
    small_movies_raw_data = sc.textFile(small_movies_file, 2000)
    small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

    small_movies_data = small_movies_raw_data.filter(lambda line: line != small_movies_raw_data_header) \
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1])).persist()
    # print small_movies_data.take(10)

    # 准备数据
    training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0L)
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
    # print "small_ratings_data_validation_for_predict_RDD:\n{}".format(validation_for_predict_RDD.take(10))
    # print "small_ratings_data_test_for_predict_RDD:\n{}".format(test_for_predict_RDD.take(10))
    # print "small_ratings_data_training_RDD:\n{}".format(small_ratings_data.take(10))

    # 训练模型(选择ALS的参数完成训练)
    from pyspark.mllib.recommendation import ALS
    import math

    seed = 5L
    iterations = 10
    regularization_parameter = 0.1
    ranks = [4, 8, 12]
    errors = [0, 0, 0]
    err = 0
    tolerance = 0.02

    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    for rank in ranks:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                          lambda_=regularization_parameter)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[err] = error
        err += 1
        print 'For rank %s the RMSE is %s' % (rank, error)
        if error < min_error:
            min_error = error
            best_rank = rank

    print 'The best model was trained with rank %s' % best_rank


def complete_dataset_model():
    datasets_path = "/user/test/datasets/"
    # Load the complete dataset file
    complete_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
    complete_ratings_raw_data = sc.textFile(complete_ratings_file, 50)
    complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

    # Parse
    complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line != complete_ratings_raw_data_header) \
        .map(lambda line: line.split(",")).map(
        lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))).persist()

    print "There are %s recommendations in the complete dataset-small" % (complete_ratings_data.count())

    # 拆分数据
    training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0L)
    # 训练模型
    from pyspark.mllib.recommendation import ALS
    import math

    seed = 5L
    best_rank = 8
    iterations = 10
    regularization_parameter = 0.1
    complete_model = ALS.train(training_RDD, best_rank, seed=seed,
                               iterations=iterations, lambda_=regularization_parameter)
    # 在测试数据上验证
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

    predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
    print 'For testing data the RMSE is %s' % (error)

    # 进行预测
    complete_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')
    complete_movies_raw_data = sc.textFile(complete_movies_file, 50)
    complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

    # Parse
    complete_movies_data = complete_movies_raw_data.filter(lambda line: line != complete_movies_raw_data_header) \
        .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), tokens[1], tokens[2])).persist()

    complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))

    print "\nThere are %s movies in the complete dataset" % (complete_movies_titles.count())

    def get_counts_and_averages(ID_and_ratings_tuple):
        nratings = len(ID_and_ratings_tuple[1])
        return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1])) / nratings)  # 每部电影的平均打分

    # 设定一个最小的预测得分的rating数量，这个数字算是一个超参数，我们这里用每部电影的平均打分数量去作为这个值。
    movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
    movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
    movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))  # (2050, 6)movie_id, 对此电影评价的人数
    # 添加新的用户得分
    new_user_ID = 0

    # 按照(userID, movieID, rating)的格式来
    new_user_ratings = [
        (0, 260, 9),  # Star Wars (1977)
        (0, 1, 8),  # Toy Story (1995)
        (0, 16, 7),  # Casino (1995)
        (0, 25, 8),  # Leaving Las Vegas (1995)
        (0, 32, 9),  # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
        (0, 335, 4),  # Flintstones, The (1994)
        (0, 379, 3),  # Timecop (1994)
        (0, 296, 7),  # Pulp Fiction (1994)
        (0, 858, 10),  # Godfather, The (1972)
        (0, 50, 8)  # Usual Suspects, The (1995)
    ]
    new_user_ratings_RDD = sc.parallelize(new_user_ratings)
    print 'New user ratings: %s' % new_user_ratings_RDD.take(10)
    # 通过Spark的 union() transformation把它加到complete_ratings_data中
    complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)
    # print "complete_data_with_new_ratings_RDD:{}".format(complete_data_with_new_ratings_RDD.take(10))
    # 用在小数据集上调得的参数去初始化ALS参数，进行训练
    from time import time

    t0 = time()
    new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed,
                                  iterations=iterations, lambda_=regularization_parameter)
    tt = time() - t0

    print "New model trained in %s seconds" % round(tt, 3)
    # 计算取得最优推荐结果
    # 我们对加入的新用户进行预测推荐
    new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)  # get just movie IDs
    print "new_user_ratings_ids:{}".format(new_user_ratings_ids)
    # keep just those not on the ID list
    new_user_unrated_movies_RDD = (
        complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))
    print "new_user_unrated_movies_RDD:", new_user_unrated_movies_RDD.take(10)  # 输出：(0, 2), (0, 3)

    # Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
    new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
    print "new_user_recommendations_RDD:{}".format(new_user_recommendations_RDD.take(10))
    # Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
    new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
    print "new_user_recommendations_rating_RDD:{}".format(new_user_recommendations_rating_RDD.take(10))
    print "complete_movies_titles:{}".format(complete_movies_titles.take(10))
    print "movie_rating_counts_RDD:{}".format(movie_rating_counts_RDD.take(10))
    new_user_recommendations_rating_title_and_count_RDD = \
        new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
    print "new_user_recommendations_rating_title_and_count_RDD.take(3):\n{}".format(new_user_recommendations_rating_title_and_count_RDD.take(3))
    # 电影ID我们是没有看到实际电影名的，整理一下得到 (Title, Rating, Ratings Count)形式的结果
    new_user_recommendations_rating_title_and_count_RDD = \
        new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
    print "new_user_recommendations_rating_title_and_count_RDD:{}".format(new_user_recommendations_rating_title_and_count_RDD.take(10))
    # 输出：（u'Au revoir les enfants (1987)'，7.539435262200577，4）
    # 给用户取出推荐度最高的电影，数量用25去做一个截取(选取评分最高的25个)
    top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2] >= 25).takeOrdered(25, key=lambda x: -x[1])
    print ('TOP recommended movies (with more than 25 reviews):\n%s' %
           '\n'.join(map(str, top_movies)))


if __name__ == '__main__':
    # yarn方式
    # os.environ.setdefault("PYSPARK_PYTHON", "/usr/bin/python")
    # 本地
    os.environ.setdefault("PYSPARK_PYTHON", "F:/tools/anaconda2/python")
    conf = SparkConf().setMaster("local[*]").setAppName("recommender_system_using_spark")
    sparkSession = (SparkSession.builder
                    .config(conf=conf)
                    .getOrCreate())
    sc = sparkSession.sparkContext
    sc.setLogLevel("ERROR")

    complete_dataset_model()

