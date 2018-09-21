# Import all necessary libraries and setup the environment for matplotlib

from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
import matplotlib.pyplot as plt


spark = SparkSession \
    .builder \
    .getOrCreate()
    
    
  

test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
train_datafile= "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"

num_test_samples = 10000
num_train_samples= 60000

test_df = spark.read.csv(test_datafile,header=False,inferSchema="true")
train_df = spark.read.csv(train_datafile,header=False,inferSchema="true")

K=10
D=100

assembler = VectorAssembler(inputCols=train_df.columns[1:],outputCol="feature")
train_vector=assembler.transform(train_df).select("_c0","feature")

assembler = VectorAssembler(inputCols=test_df.columns[1:],outputCol="feature")
test_vector=assembler.transform(test_df).select("_c0","feature")

pca = PCA(k=D, inputCol="feature", outputCol="features")
model = pca.fit(train_vector)

train_pca_result = model.transform(train_vector).select('_c0','features')
test_pca_result = model.transform(test_vector).select('_c0','features')


train_labels = train_pca_result.rdd.map(lambda t:t[0]).collect()

train_pca = np.asarray(train_pca_result.rdd.map(lambda t:np.asarray(t[1])).collect())

def newKNNs(t):
    x=np.asarray(t[1])
    dis=np.einsum('ij,ji->i',train_pca,train_pca.T)-2*x.dot(train_pca.T)
    mini=np.argsort(dis)
    mins=np.zeros(10)
    for i in range(K):
        mins[train_labels[mini[i]]]+=1
    minn=np.argmax(mins)
    return(minn,t[0])
result_rdd=test_pca_result.rdd.map(newKNNs)

final_result=result_rdd.collect()

n=0
for i in range(10000):
    if(final_result[i][0]==final_result[i][1]):
        n=n+1

TP=np.zeros(10)
FP=np.zeros(10)
TN=np.zeros(10)
FN=np.zeros(10)
Accuracy=np.zeros(10)
Precision=np.zeros(10)
Recall=np.zeros(10)
F1_Score=np.zeros(10)
for i in range(10):
    for j in range(10000):
        if(final_result[j][0]==i and final_result[j][1]==i ):
            TP[i]+=1
        if(final_result[j][0]==i and final_result[j][1]!=i):
            FP[i]+=1
        if(final_result[j][0]!=i and final_result[j][1]==i):
            FN[i]+=1
        if(final_result[j][0]!=i and final_result[j][1]!=i):
            TN[i]+=1
    Accuracy[i]=(TP[i]+TN[i])/(TP[i]+TN[i]+FN[i]+FP[i])
    Precision[i] = TP[i]/(TP[i]+FP[i])
    Recall[i] = TP[i]/(TP[i]+FN[i])
    F1_Score[i] = 2*(Recall[i] * Precision[i]) / (Recall[i] + Precision[i])

print(Accuracy)
print(Precision)
print(Recall)
print(F1_Score)
print(n)
