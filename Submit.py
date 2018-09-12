# -*- coding:utf-8 -*-
"""
提交任务
"""

import os, subprocess, sys
reload(sys)
sys.setdefaultencoding('utf8')


import os, zipfile


def zip_dir(dir_list, zipfilename):
    """
    代码打包
    """
    filelist = []
    for dirname in dir_list:
        if os.path.isfile(dirname):
            filelist.append(dirname)
        else :
            for root, dirs, files in os.walk(dirname):
                for name in files:
                    filelist.append(os.path.join(root, name))
                    # print os.path.join(root, name)

    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar
        print arcname
        zf.write(tar, arcname)
    zf.close()


# 执行代码打包
dir_list = ["./recommender_system_using_spark"]
zip_dir(dir_list, "./recommender_system_using_spark.zip")


SPARK_PATH = "F:\\tools\\Spark\\spark-2.1.0-bin-hadoop2.7\\bin\\spark-submit"
HADOOP_CONFIG = "F:\\tools\\Spark\\hadoop_config"
HADOOP_USER_NAME = "hdfs"
SPARK_CLASSPATH = "F:\\tools\\Spark\\spark-2.1.0-bin-hadoop2.7\\jars"


os.environ.setdefault("HADOOP_CONF_DIR", HADOOP_CONFIG)
os.environ.setdefault("HADOOP_USER_NAME", HADOOP_USER_NAME)
os.environ.setdefault("YARN_CONF_DIR", HADOOP_CONFIG)
os.environ.setdefault("SPARK_CLASSPATH", SPARK_CLASSPATH)

command = [
        SPARK_PATH,
        "--master", "yarn",
        "--deploy-mode", "cluster",
        "--name", "RobotX-Test",
        "--file", "F:/tmp/datasets/ml-latest-small/movies.csv",
        "--file", "F:/tmp/datasets/ml-latest-small/ratings.csv",
        "--py-files", "recommender_system_using_spark.py",
        "--driver-memory", "1G",
        "--num-executors", "1",
        "--executor-memory", "1G",
        "F:/bigdata_learn/spark_ml/recommender_system_using_spark/recommender_system_using_spark.py"
    ]
print( " ".join( command ) )
print( os.path.dirname( os.path.realpath( sys.argv[0] ) ) )
try:
    p = subprocess.Popen(" ".join( command ),
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         cwd=os.path.dirname(os.path.realpath(sys.argv[0])))
    application_id = None
    tracking_url = None
    while p.poll() is None:
        # print p.poll()
        line = p.stderr.readline()#.decode('utf-8', 'ignore').strip()
        print line
        if len(line) > 0 and (application_id is None or tracking_url is None):
            assert isinstance(line, str)
            if line.startswith("tracking URL:"):
                tracking_url = line.replace("tracking URL:", "").strip()
                print(tracking_url)
            elif "Submitted application" in line:
                application_id = line.split("Submitted application")[1].strip()
                print(application_id)
except Exception as e:
    # print( str( e ).decode( 'cp936' ).encode( 'utf-8' ) )
    print( e )
