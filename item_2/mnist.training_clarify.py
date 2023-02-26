import pickle
import gzip
import numpy as np
from sklearn import svm
import time
from sklearn.preprocessing import scale
import mglearn

# TIME函数库用于输出时间供比较训练结果
# 纳入全部函数库


def load_data():
    """
    返回包含训练数据、验证数据、测试数据的元组的模式识别数据
    训练数据包含50，000张图片，测试数据包含10,000张图片
    """
    f = gzip.open('D:/python/pythontest/mnist.pkl.gz', 'rb')    #在这里更改路径，要求pkl.gz压缩包文件！
    training_data, validation_data, test_data = pickle.load(
        f, encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)
 # 从‘pkl.gz'中读取训练和测试数据
    


def svm_baseline():  # 训练主函数，经测试约需要训练6~7分钟
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    # 输出开始时间
    training_data, validation_data, test_data = load_data()

    # (可选）图像数据集预处理training_data[0] = preprocessing.StandardScaler().fit_transform(training_data[0])

    # 三种可能svc函数供选择:事实证明非线性的svm.SVC训练效果更好（非线性，非线性，线性，事实证明，线性svc需要更大的C值来弥补精度）
    #clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03, verbose=1, max_iter=2000)
    #clf = svm.SVC(C=8.0, kernel='rbf',cache_size=8000,probability=False)
    clf = svm.LinearSVC(dual=True, tol=0.0001, C=100.0,
                        verbose=1, max_iter=10000)
    clf.fit(training_data[0], training_data[1])

    # 非线性模型训练，调大C加快速率
    '''
# 保存模型组件（以备下次使用，可以大幅提高预测特定图片的效率，10000张图只需要预测处理约2分钟，节约了训练的大量时间）

    file = open("D:/python/pythontest/model.pickle", "wb")   #以写二进制的方式打开文件
 
    pickle.dump(clf, file)   #把模型写入到文件中

    file.close()  # 关闭文件
    '''
    '''
#读取模型组件
    file = open("D:/python/pythontest/model.pickle", "rb")  #以读二进制的方式打开文件

    clf = pickle.load(file)  # 把模型从文件中读取出来

    file.close()  # 关闭文件
    '''

    
    
    '''
 plus:支持向量可视化组件
 # 重启R
 # load('D:/model.RData')
 #mglearn.plots.plot_2d_separator(svm, training_data[0], eps=.5)

 #mglearn.discrete_scatter(training_data[:, 0], training_data[:, 1], training_data)

 # 画出支持向量

 #sv = svm.support_vectors_

 # 支持向量的类别标签有dual_coef_的正负号给出

 #sv_labels = svm.dual_coef_.ravel() > 0

 # mglearn.discrete_scatter(
 # sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)

 #plt.xlabel("Feature 0")

 #plt.ylabel("feature 1")
    '''
 # data[0]是图形集，data[1]是标签集，实现训练
 # test
 # 测试组件，测试集测试预测结果
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("%s of %s test values correct." % (num_correct, len(test_data[1])))
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
# 输出结束时间


if __name__ == "__main__":
    svm_baseline()
# 主函数
