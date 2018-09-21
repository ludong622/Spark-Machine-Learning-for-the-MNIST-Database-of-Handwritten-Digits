
# Import all necessary libraries and setup the environment for matplotlib

from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession \
    .builder \
    .appName("Random Forest") \
    .config('spark.executor',4) \
    .getOrCreate()
    

test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
train_datafile= "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"

num_test_samples = 10000
num_train_samples= 60000

test_df = spark.read.csv(test_datafile,header=False,inferSchema="true")
train_df = spark.read.csv(train_datafile,header=False,inferSchema="true")

assembler = VectorAssembler(inputCols=train_df.columns[1:],outputCol="feature")
train_vector=assembler.transform(train_df).select("_c0","feature")
pca = PCA(k=99, inputCol="feature", outputCol="features")#PCA 784 to 99
model = pca.fit(train_vector)
train_pca_result = model.transform(train_vector).select('_c0','features')
new_train_pca_result=train_pca_result.withColumnRenamed("_c0", "label")

assembler = VectorAssembler(inputCols=test_df.columns[1:],outputCol="feature")
test_vector=assembler.transform(test_df).select("_c0","feature")
test_pca_result = model.transform(test_vector).select('_c0','features')
new_test_pca_result=test_pca_result.withColumnRenamed("_c0", "label")

# train the model
# create the trainer and set its parameters
layers = [50,100,10]# 3-layer perceptron with 50 inputs, 10 outputs, and one hidden layer with 100 nodes#
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=123)
trainmodel = trainer.fit(new_train_pca_result)#label/features

results = trainmodel.transform(new_test_pca_result)
predictionAndLabels = results.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(""Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
#evaluator.saveAsTextFile(output_path)
