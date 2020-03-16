import numpy as np

def perceptron(trainPath,testPath):
    def readData(testData, trainData):
        test_data = np.loadtxt(testData, dtype=np.str, delimiter=',')
        train_data = np.loadtxt(trainData, dtype=np.str, delimiter=',')
        x_train = train_data[:, 0:4].astype(np.float)
        y_train = train_data[:, 4]

        x_test = test_data[:, 0:4].astype(np.float)
        y_test = test_data[:, 4]
        x_train = np.c_[x_train, np.ones(x_train.shape[0])]  # another is for bias; it's always to be one
        x_test = np.c_[x_test, np.ones(x_test.shape[0])]  # another is for bia; sit's always to be one
        # print(x_train)
        return (x_train, y_train, x_test, y_test)

    def GrouptheData(y_data):
        classOne = []
        classTwo = []
        classThree = []
        for i in range(len(y_data)):
            if y_data[i] == "class-1":
                classOne.append(i)
            elif y_data[i] == "class-2":
                classTwo.append(i)
            elif y_data[i] == "class-3":
                classThree.append(i)
        return (classOne, classTwo, classThree)

    def getGroupDataSet(class_set,x,y):
        x_set = np.vstack((x[class_set[0],], x[class_set[1],]))
        y_set = np.hstack((y[class_set[0],], y[class_set[1],]))
        return x_set,y_set

    def unique_label(y):
        return np.unique(y)

    def label_num(y):
        return len(unique_label(y))

    # initialize_weights randomly
    def initialize_weights(dim, num):
        w = np.zeros((dim, num))
        return w.T

    def perceptron_model(x, w, s=0):
        z = np.dot(w, x.T)
        z[z > 0] = 1
        z[z <= 0] = 0
        return z.T

    def train_model(x, y, s, num_echo):
        w = initialize_weights(dim=(x.shape[1]), num=label_num(y) - 1)

        for i in range(num_echo):
            for j in range(x.shape[0]):
                input = np.array([x[j,]])
                output = perceptron_model(input, w)
                label = getLabel(y, j)
                w = update(w, err(label, output), label, input, -0.1)
        return w

    def update(w, err, label, x, r):
        if err == 0:
            return w
        if label == 1:
            t = 1
        elif label == 0:
            t = -1
        return w - 2 * r * w + err * x
        # w+(r*np.dot(x.T,err)).T

    def err(label, output):
        return label - output

    def getLabel(y, index):
        label_list = unique_label(y)
        i = np.where(label_list == y[index])[0]
        if i == 1:
            label = 1
        elif i == 0:
            label = 0

        return label

    def predicate(w, x):
        output = perceptron_model(x, w)
        return output

    def confusion_matrix(output,y_label):
        tp=0
        fp=0
        fn=0
        tn=0
        if len(output)==len(y_label):
            for i in range(len(output)):
                if output[i]==getLabel(y_label,i):
                    if getLabel(y_label,i)==0:
                        tp=tp+1
                    else:
                        tn=tn+1
                else:
                    if getLabel(y_label,i)==0:
                        fp=fp+1
                    else:
                        fn=fn+1
        else:
            print("the length of output and y_label not same")
        return tp,fp,fn,tn

    def runModel(class_trainSet,class_testSet,trainAll,testAll):
        x_train, y_train = getGroupDataSet(class_trainSet, trainAll[0], trainAll[1])
        x_test, y_test =getGroupDataSet(class_testSet,testAll[0],testAll[1])
        w = train_model(x_train, y_train, 0, 100)
        out = predicate(w, x_test)
        cMatric=confusion_matrix(out,y_test)
        return w,out,cMatric

    #trainPath = 'data/train.data'
    #testPath = 'data/test.data'

    x_train, y_train, x_test, y_test = readData(trainData=trainPath, testData=testPath)
    trainAll=[x_train,y_train]
    testAll=[x_test,y_test]

    train_classOne, train_classTwo, train_classThree = GrouptheData(y_train)
    test_classOne, test_classTwo, test_classThree = GrouptheData(y_test)

    #x_t1,y_t1=getGroupDataSet([train_classOne,train_classTwo],x_train,y_train)
    #w = train_model(x_t1, y_t1, 0, 100)

    #x_t2 = np.vstack((x_test[test_classTwo,], x_test[test_classOne,]))
   # y_t2 = np.hstack((y_test[test_classTwo,], y_test[test_classOne,]))

    # the (a) group to test; classone and classtwo
    train_classA=[train_classOne,train_classTwo]
    test_classA=[test_classOne,test_classTwo]
    resultA=runModel(class_trainSet=train_classA,class_testSet=test_classA,trainAll=trainAll,testAll=testAll)
    print(resultA)

trainPath = 'data/train.data'
testPath = 'data/test.data'
perceptron(trainPath=trainPath,testPath=testPath)

