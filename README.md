# perceptron_comp337

This is the simply python code about the perceptron including binary classification and multi classification. The goal is to classificate the data set having 3 different class. 

The binary classification has only one output and the math rule is y=wx+b, and we can make b=w_bias*bias (bias=1); therefore, the final function is y=w'*x'.

The binary classification has two output. The fist one is marked as 01, second one to be 10 and the third one to be 11. Those labels are in binary, which means the every output can print only 0 and 1.

## Read Data
The input of the data can only be read by reading the path.

## Input Parameters
Except to reading the training and testing data set, it also can set the regularisation coefficient(r) and echo times(echo_num). And r default 1, echo_num default 100.

## Output 
#### Binary percptron
The result is result of the 3 different groups data set (resultA,resultB,resultC) and each of the result(the result of the 3 groups) is the weight of perceptron, perceptron output of the testing and training and the accelerate of the testing and training.

#### Multi perceptron
The result is the weight of perceptron, perceptron output of the testing and training and the accelerate of the testing and training.
