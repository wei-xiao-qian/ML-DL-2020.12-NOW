```python
import numpy as np
import scipy.special
```

这些代码可用于创建、训练和查询3层神经网络，进行几乎任何任务，这么看来，代码不算太多。

#query()函数接受神经网络的输入，返回网络的输出。但是为了做到这一点，需要传递来自输入层节点的输入信号，通过隐藏层，最后从输出层输出。当信号馈送至给定的隐藏层节点或输出层节点时，我们使用链接权重W调节信号，还用sigmoid函数来抑制这些节点的信号。

### 神经网络类

```python
#神经网络类

class neuralNetwork:
    #********************************初始化神经网络*****************************************************
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes
        #权重
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
        #学习率
        self.lr = learningrate
        
        #激活函数，sigmoid函数
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    #*************************************训练网络*************************************
    def train(self,inputs_list,targets_list):
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T,output_errors)
        #更新权重
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),np.transpose(inputs))
        
        pass
    #*************************************查询网络*************************************
    
    def query(self,inputs_list):
        #把输入转化成二维数组
        inputs = np.array(inputs_list,ndmin=2).T
        #隐藏层   计算隐藏层的输入信号
        hidden_inputs = np.dot(self.wih,inputs)
        #calculate the signals emerging from hidden laywe
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #输出层
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
```



#### 由于readlines()会将整个文件读取到内存中，当文件很大时使用这种方法不如一行一行读的有效率

#### 但我们现在的文件小所以使用readlines代码简单一些;

```python

#number of input,hidden,and output nodes

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate =0.3

#create instance of neual network
n  = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open("mnist_train_100.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
```

### 训练

```python
# train the neural network
epochs = 5#增加训练次数
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) +0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        pass
    pass
```

### 测试

```python
#load the mnist test data CSV file into a list
test_data_file = open("mnist_test_10.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
```

2134466890-

```python
#test the neural network
scorecard  = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

#结束
```

### 结束