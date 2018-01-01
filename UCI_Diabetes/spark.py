# Spark Application - execute with spark-submit：
# spark-submit app.py
# Imports
from pyspark import SparkConf, SparkContext
# from pyspark import mllib.linalg.Vector
# from pyspark import mllib.stat.{MultivariateStatisticalSummary, Statistics}
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Vector
from pyspark.mllib.stat import MultivariateStatisticalSummary
from pyspark.mllib.stat import Statistics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics

# Module Constants
APP_NAME = "DiabeteDecisionTree"


# Closure Functions
def tokenize(item):
    vector = Vectors.dense(float(item[0]), float(item[1]), float(item[2]), float(item[3]),
                           float(item[4]), float(item[5]), float(item[6]), float(item[7]))
    label = 0.0 if item[8] == '0' else 1.0
    item = LabeledPoint(label, vector)
    return item


# Main functionality
def main(sc):
    diabete_lines = sc.textFile("diabete.data")
    #print(diabete_lines.collect())
    """
    UCI Pima Indians Diabetes Data Set。
    数据集包含768个数据集，分为2类，
    每个数据包含8个属性。
    可通过怀孕次数，血糖浓度，血压舒张压，肱三头肌皮褶厚度，胰岛素浓度，身体质量指数（体重/(身高^2）），家族发病率，8个属性
    预测样本属于（患糖尿病，不患糖尿病）两个种类中的哪一类
    整个文件读取成一个一维数组，每行都是一个RDD
    [
        6,148,72,35,0,33.6,0.627,50,1,
        1,85,66,29,0,26.6,0.351,31,0,
        8,183,64,0,0,23.3,0.672,32,1,
        1,89,66,23,94,28.1,0.167,21,0,
        0,137,40,35,168,43.1,2.288,33,1,
        5,116,74,0,0,25.6,0.201,30,0,
    ]
    """
    diabete_lines = diabete_lines.map(lambda item: item.split(","))
    # print(diabete_lines.collect())
    plasma = []
    for i in diabete_lines.collect():
        plasma.append(i[1])
    # print(plasma)
    """
    我们对把每一行都通过split函数形成数组.整个diabete_lines看上去是一个二维数组.
    """
    diabete_points = diabete_lines.map(lambda item:tokenize(item))
    # print(diabete_points.collect())
    # print(diabete_points)
    """
    通过map把rdd中每一项都转换成一个labeledpoint
    """
    training, testing = diabete_points.randomSplit([0.8, 0.2], 11)
    print(training.count())
    print(testing.count())
    """
    首先进行数据集的划分，这里划分80%的训练集和20%的测试集：
    然后，调用决策树的trainClassifier方法构建决策树模型，设置参数，比如分类数、信息增益的选择、树的最大深度等
    """
    # model = DecisionTree.trainClassifier(diabete_points, numClasses = 3,maxDepth = 5,maxBins = 32,{})
    model = DecisionTree.trainClassifier(training, numClasses=2, maxDepth=5, maxBins=32, categoricalFeaturesInfo={})
    print(model)
    print(model.toDebugString())
    """
    从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，
    由该特征的不同取值建立子结点，再对子结点递归地调用以上方法，构建决策树；
    直到所有特征的信息增均很小或没有特征可以选择为止，最后得到一个决策树。
    决策树需要有停止条件来终止其生长的过程。
    一般来说最低的条件是：当该节点下面的所有记录都属于同一类，或者当所有的记录属性都具有相同的值时。
    这两种条件是停止决策树的必要条件，也是最低的条件。
    在实际运用中一般希望决策树提前停止生长，限定叶节点包含的最低数据量，
    以防止由于过度生长造成的过拟合问题。

    这里，采用了test部分的数据每一行都分为标签label和特征features，
    然后利用map方法，对每一行的数据进行model.predict(features)操作，
    获得预测值。并把预测值和真正的标签放到predictionAndLabels中。
    我们可以打印出具体的结果数据来看一下：
    注意,预测出来的值必须转换成float,否则在MulticlassMetrics会出现int到double转换的错误
    TypeError: DoubleType can not accept object 1 in type <class 'int'>
    """
    predictionList = []
    for item in testing.collect():
        predictionList.append([model.predict(item.features),item.label])
    predictionAndLabels = sc.parallelize(predictionList)
    #print(predictionAndLabels.collect())
    wrong, miss, hit = 0, 0, 0
    for i in predictionAndLabels.collect():
        if i[0] - i[1] == 1:
            wrong += 1
        if i[0] - i[1] == -1:
            miss +=1
        if i[0] - i[1] == 0:
            hit +=1
    #print(wuzhen, louzhen, quezhen)
    """
    [0.0, 0.0], 
    [0,0, 0.0], 
    [0.0, 0.0], 
    [0.0, 0.0], 
    [0.0, 0.0], 
    [0.0, 0.0], 
    [0.0, 0.0], 
    """
    metrics = MulticlassMetrics(predictionAndLabels)
    print("精确度:"+str(metrics.precision()))
    #print("准确率:"+str(metrics.accuracy))
    #print("召回率:"+str(metrics.recall()))
    print("混淆矩阵")
    print(metrics.confusionMatrix())
    """
    最后，我们把模型预测的准确性打印出来：
    """
    sc.stop()

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)
    # Execute Main functionality
    main(sc)
