import numpy as np


def readData(testData, trainData):
    test_data = np.loadtxt(testData, dtype=np.str, delimiter=',')
    train_data = np.loadtxt(trainData, dtype=np.str, delimiter=',')
    x_train = train_data[:, 0:4].astype(np.float_)
    y_train = train_data[:, 4]

    x_test = test_data[:, 0:4].astype(np.float_)
    y_test = test_data[:, 4]
    x_train = np.c_[x_train, np.ones(x_train.shape[0])]  # another is for bias; it's always to be one
    x_test = np.c_[x_test, np.ones(x_test.shape[0])]  # another is for bia; sit's always to be one
    # print(x_train)
    return (x_train, y_train, x_test, y_test)

def unique_label(y):
        return np.unique(y)

def label_num(y):
        return len(unique_label(y))

    # initialize_weights randomly
def initialize_weights(dim, num):
        w = np.zeros((dim, num))
        return w.T

def perceptron_model(x, w):
        z = np.dot(w, x.T)
        z[z > 0] = 1
        z[z <= 0] = -1
        return z.T

def err(label, output):
        return label - output

# the ratio is the
def binary_perceptron(trainPath,testPath, ratio=0.1, num_echo=100):# we can set the ratio value and looping times

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

    def train_model(x, y, num_echo):
        w = initialize_weights(dim=(x.shape[1]), num=label_num(y) - 1)

        for i in range(num_echo):
            for j in range(x.shape[0]):
                input = np.array([x[j,]])
                output = perceptron_model(input, w)
                label = getLabel(y, j)
                w = update(w, label,output, input, ratio)
        return w

    def update(w, y,output, x, r):
        if err(y,output) == 0:
           return w
        return w - 2*np.multiply.reduce([r,w]) + y * x
        # w+(r*np.dot(x.T,err)).T

    def getLabel(y, index):
        label_list = unique_label(y)
        i = np.where(label_list == y[index])[0]
        if i == 1:
            label = 1
        elif i == 0:
            label = -1
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

    def get_accelerate(cMetrix):
        tp, fp, fn, tn = cMetrix
        return (tp + tn) / (tp + fp + fn + tn)

    def runModel(class_trainSet,class_testSet,trainAll,testAll):
        x_train, y_train = getGroupDataSet(class_trainSet, trainAll[0], trainAll[1])
        x_test, y_test =getGroupDataSet(class_testSet,testAll[0],testAll[1])
        w = train_model(x_train, y_train, num_echo)
        out1 = predicate(w, x_test)
        out2 = predicate(w,x_train)
        cMatric1=confusion_matrix(out1,y_test)
        cMatric2=confusion_matrix(out2,y_train)
        acc1=get_accelerate(cMatric1)
        acc2=get_accelerate(cMatric2)
        return w,out1,out2,acc1,acc2

    #trainPath = 'data/train.data'
    #testPath = 'data/test.data'

    x_train, y_train, x_test, y_test = readData(trainData=trainPath, testData=testPath)
    trainAll=[x_train,y_train]
    testAll=[x_test,y_test]

    train_classOne, train_classTwo, train_classThree = GrouptheData(y_train)
    test_classOne, test_classTwo, test_classThree = GrouptheData(y_test)

    # the (a) group to test; classone and classtwo
    train_classA=[train_classOne,train_classTwo]
    test_classA=[test_classOne,test_classTwo]
    resultA=runModel(class_trainSet=train_classA,class_testSet=test_classA,trainAll=trainAll,testAll=testAll)

    # the (b) group to test; classtwo and class three
    train_classB = [train_classTwo, train_classThree]
    test_classB = [test_classTwo, test_classThree]
    resultB = runModel(class_trainSet=train_classB, class_testSet=test_classB, trainAll=trainAll, testAll=testAll)

    # the (c) group to test; class one and class three
    train_classC = [train_classOne, train_classThree]
    test_classC = [test_classOne, test_classThree]
    resultC = runModel(class_trainSet=train_classC, class_testSet=test_classC, trainAll=trainAll, testAll=testAll)

    return resultA,resultB,resultC

def multi_perceptron(trainPath,testPath,ratio=0,num_echo=20):
    def train_model(x, y, num_echo):
        w = initialize_weights(dim=(x.shape[1]), num=label_num(y) - 1)

        for i in range(num_echo):
            for j in range(x.shape[0]):
                input = np.array([x[j,]])
                output = perceptron_model(input, w)
                label = getLabel(y, j)
                w = update(w, label, output, input, ratio)
        return w

    def update(w, y, output, x, r):
        error = err(y, output)
        return w - 2 * np.multiply.reduce([r, w]) + error.reshape(1, -1).T * x

    def err(label, output):  # xxx
        return label - output

    def getLabel(y, index):
        label_list = unique_label(y)
        i = np.where(label_list == y[index])[0]
        label = np.zeros(label_num(y) - 1)
        if i == 0:
            label[1] = 1
            label[0] = -1
        elif i == 1:
            label[0] = 1
            label[1] = -1
        else:
            label[1] = 1
            label[0] = 1
        return label

    def predicate(w, x, y):
        output = perceptron_model(x, w)
        acc = get_accelerate(output, y)
        return acc, output

    def get_accelerate(output, y_label):
        acc = 0
        if len(output) == len(y_label):
            for i in range(len(output)):
                if (output[i] == getLabel(y_label, i)).all():
                    acc = acc + 1
            acc = acc / len(output)
        else:
            print("the length of output and y_label not same")
        return acc

    x_train, y_train, x_test, y_test = readData(trainData=trainPath, testData=testPath)
    w = train_model(x_train, y_train, num_echo)
    out = predicate(w, x_test, y_test)
    out2 = predicate(w, x_train, y_train)
    return w,out,out2

trainPath = 'data/train.data'# the training set path
testPath = 'data/test.data'# the testing set path
ratio_list=[0,0.01,0.1,1,10,100]

#binary perceptron
print("________the binary perceptron_____")

for i in range(len(ratio_list)):
    resultA, resultB,resultC=binary_perceptron(trainPath=trainPath,testPath=testPath,ratio=ratio_list[i],num_echo=100)
   # if ratio_list[i]==0:
    #   print(resultA[0])
     #  print(resultB[0])
      # print(resultC[0])

    print("When the regularisation coefficient to be: ", ratio_list[i])
    print("testing set")
    print("(a) group accelerate",(resultA[3]))
    print("(b) group accelerate",(resultB[3]))
    print("(c) group accelerate",(resultC[3]))
    #resultA, resultB, resultC = binary_perceptron(trainPath=trainPath, testPath=trainPath, ratio=ratio_list[i],num_echo=100)
    print("training set")
    print("(a) group accelerate",(resultA[4]))
    print("(b) group accelerate",(resultB[4]))
    print("(c) group accelerate",(resultC[4]))

#multi perceptron
w,output_testing,output_training=(multi_perceptron(trainPath=trainPath, testPath=trainPath))
print("________the multi perceptron_____")
print(w)
print("the testing accelerate: ",output_testing[0])
print("the training accelerate: ",output_training[0])