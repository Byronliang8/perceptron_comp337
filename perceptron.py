import numpy as np
import pandas as pd

def readData(testData, trainData):
    test_data=np.loadtxt(testData, dtype=np.str, delimiter=',')
    train_data=np.loadtxt(trainData, dtype=np.str, delimiter=',')
    #train_data = pd.read_table(trainData, delimiter=",")
    #test_data = pd.read_table(testData, delimiter=",")
    x_train = train_data[:,0:4].astype(np.float)
    y_train = train_data[:, 4]

    x_test = test_data[:,0:4].astype(np.float)
    y_test = test_data[:, 4]
    x_train=np.c_[x_train,np.ones(x_train.shape[0])]# another is for bias
    x_test=np.c_[x_test,np.ones(x_test.shape[0])]# another is for bias
    #print(x_train)
    return (x_train,y_train,x_test,y_test)

def unique_label(y):
    return np.unique(y)


def label_num(y):
    return len(unique_label(y))

# initialize_weights randomly
def initialize_weights(dim,num):
    w = np.zeros((dim,num))
    return w.T

def perceptron_model(x,w):
    z=np.dot(w, x.T)
    z[z>0]=1
    z[z<=0]=0
    return z.T

def train_model(x,y,s,num_echo):
    w=initialize_weights(dim=(x.shape[1]),num=label_num(y)-1)

    for i in range(num_echo):
        for j in range(x.shape[0]):
            input=np.array([x[j,]])
            output=perceptron_model(input,w)
            label=getLabel(y,j)
            w=update(w,label,output,input,0)
    return w

def update(w,y,output,x,r):
    error=err(y,output)
    if (error==0).all():
        return w
    return w-2*np.multiply.reduce([r,w]) + error.reshape(1,-1).T * x

def err(label,output):#xxx
    return label-output

def getLabel(y,index):
    label_list=unique_label(y)
    i = np.where(label_list==y[index])[0]
    label=np.zeros(label_num(y)-1)
    if i==0:
        label[1]=1
    elif i==1:
        label[0]=1
    else:
        label[1]=1
        label[0]=1
    return label

def predicate(w,x,y):
    output=perceptron_model(x,w)
    #print(output)
    #for i in range(output.shape[0]):
    #    out=output[i]
     #   if (out==np.array([0,1])).all():
      #      print("class-1")
       # elif (out==np.array([1,0])).all():
        #    print("class-2")
        #elif (out==np.array([1,1])).all():
         #   print("class-3")
        #else:
         #   print("err")
    acc=get_accelerate(output,y)
    return acc,output

def get_accelerate(output,y_label):
    acc=0
    if len(output)==len(y_label):
        for i in range(len(output)):
            if (output[i]==getLabel(y_label,i)).all():
                acc=acc+1
        acc=acc/len(output)
    else:
        print("the length of output and y_label not same")
    return acc

trainPath='data/train.data'
testPath='data/test.data'
x_train,y_train,x_test,y_test=readData(trainData=trainPath,testData=testPath)

w=train_model(x_train,y_train,0,20)
print(w)
out=predicate(w,x_test,y_test)
out2=predicate(w,x_train,y_train)
print(out)
print(out2)