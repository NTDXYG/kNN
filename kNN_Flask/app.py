from flask import Flask,request,jsonify
from PIL import Image
import os
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN
import time
import requests

basedir = os.path.abspath(os.path.dirname(__file__)) #定义一个根目录 用于保存图片用

app = Flask(__name__)


@app.route('/api/v1/Upload', methods=['GET', 'POST'])
def editorData():
    #计算开始时间
    time1 = time.time()
    # 获取图片文件 name = upload
    img = request.files.get('file')
    # 定义一个图片存放的位置 存放在static下面
    path = basedir + "/static/img/"
    # 图片名称
    imgName = img.filename
    # 图片path和名称组成图片的保存路径
    file_path = path + imgName
    # 保存图片
    img.save(file_path)
    #将图片保存为32*32像素的bmp图片
    im = Image.open(file_path)
    out = im.resize((32, 32), Image.ANTIALIAS)
    outFile = path + "_new_" + imgName[:len(imgName)-3] +"bmp"
    out.save(outFile)
    result = handwritingClassTest(BMP2TXT(imgName,outFile))
    print(result)
    time2 = time.time()
    data = {
        "result": result,
        "time": str(time2-time1)
    }
    return jsonify(data)
"""
函数说明:手写数字分类测试
"""
def handwritingClassTest(filename):
    #训练集的Labels
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,训练集
    trainingMat = np.zeros((m, 1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
    #构建kNN分类器
    neigh =KNN(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=4, p=1,
           weights='distance')
    #拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    #返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行 分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        #获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))
    vectorUnderTest = img2vector(filename)
    classifierResult = neigh.predict(vectorUnderTest)
    return str(classifierResult[0])



def img2vector(filename):
    #创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    #打开文件
    fr = open(filename)
    #按行读取
    for i in range(32):
        #读一行数据
        lineStr = fr.readline()
        #每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    #返回转换后的1x1024向量
    return returnVect


def BMP2TXT(imgName,fileName):
    # 打开文件
    image_file = Image.open(fileName)
    width, height = image_file.size
    image_file = image_file.convert('RGB')
    fh = open(basedir + "/static/test/" + imgName[:len(imgName) - 3] + "txt", 'w')
    name = basedir + "/static/test/" + imgName[:len(imgName) - 3] + "txt"
    for i in range(width):
        for j in range(height):
            # 获取像素点颜色
            str_strlist = image_file.load()
            data = str_strlist[j, i]
            colorsum = data[0] + data[1] + data[2]
            # 很简陋判断黑白
            if (colorsum <= 250*3):
                fh.write('1')
            else:
                fh.write('0')
        fh.write('\n')
    fh.close()
    return name;

@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
