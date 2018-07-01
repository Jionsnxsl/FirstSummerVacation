"""
决策树代码示例
"""

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
import numpy as np

# 读取数据
allElectronicData = open("data2.csv", "rt")
reader = csv.reader(allElectronicData)
headers = next(reader)
print(headers)

# 现在要将数据转换成sklearn可以接受的格式
featureList = []
labelList = []

for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    # 将特征值组装成键值对的形式
    for i in range(1, len(row)-1):
       rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

# 向量化特征值
# 将属性如age转换成向量的形式：(1,0,0)
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX", str(dummyX))
print(vec.get_feature_names())

# 向量化标记(类别)
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY", dummyY)

# 训练模型
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(dummyX, dummyY)
print("clf", str(clf))

# 可视化模型
with open("allElectronicInformationCainOri.dot", "w") as f:
    # 不写out_file会自动生成一个dot文件，内容都是一样的。
    # dot 文件转换问图片的命令： dot input_file.dot -Tpng -o output_file.png
    tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRow = dummyX[0, :]
testRow = oneRow
testRow[0] = 1
testRow[2] = 0

# sklearn 接受的是一个二维向量，需要将测试数据进行转换
testRow = np.reshape(testRow, (1, -1))

print(testRow)

predictY = clf.predict(testRow)
print("predictY", str(predictY))








