from array_solution import *
import random
import math

class Ann:
    def __init__(self,RawData_arr):
        self.Rawdata_array = RawData_arr
        for r in range(len(self.Rawdata_array)):
            for c in range(len(self.Rawdata_array[0])-1):
                self.Rawdata_array[r][c] = float(self.Rawdata_array[r][c])
        self.Input_array = Array(len(self.Rawdata_array))
        self.Target_list = list()
        #self.Target_array = Array(len(self.Rawdata_array))
        for i in range(len(self.Rawdata_array)):
                self.Input_array[i] = self.Rawdata_array[i][0:len(self.Rawdata_array[i])-1]
                self.Target_list.append(self.Rawdata_array[i][-1])
                #self.Target_array[i] = self.Rawdata_array[i][-1]
        self.NumData, self.NumInput = len(self.Input_array),len(self.Input_array[0])
        Class = list(set(self.Target_list))
        self.Target_array = Array(self.NumData)
        self.Target_array.clear(0)
        for i in range(self.NumData):
            #ปรับให้อยู่ในรูปนี้ class1 =  0 , class2 =  1 
            self.Target_array[i] = Class.index(self.Target_list[i])
        self.NumOutput = 1

    def refdata(self):
        #ลองพิมพ์ทั้งส่วนที่เป็น feature และ target เพื่อการตรวจสอบ
        print("Example of Input[{0:3d}]: {1}".format(0,self.Input_array[0]))
        print("Example of Input[{0:3d}]: {1}".format(1,self.Input_array[1]))
        print("         :         :              :  ")
        print("Example of Input[{0:3d}]: {1}".format(len(self.Input_array)-2,self.Input_array[-2]))
        print("Example of Input[{0:3d}]: {1}".format(len(self.Input_array)-1,self.Input_array[-1]))
        print("Shape of Input: ",len(self.Input_array),len(self.Input_array[0]))

        print("\nExample of Target[{0:3d}]: {1}".format(0,self.Target_array[0]))
        print("Example of Target[{0:3d}]: {1}".format(1,self.Target_array[1]))
        print("         :      :           :  ")
        print("Example of Target[{0:3d}]: {1}".format(len(self.Target_array)-2,self.Target_array[-2]))
        print("Example of Target[{0:3d}]: {1}".format(len(self.Target_array)-1,self.Target_array[-1]))
        print("Number of Target:", len(self.Target_array))
       

    def train_test_set(self,numfold,k):
        self.numfold = numfold
        self.Input_Train = Array(self.NumData-(self.NumData//self.numfold))
        self.Input_Test = Array(self.NumData//self.numfold)
        self.Target_Train = Array(self.NumData-(self.NumData//self.numfold))
        self.Target_Test = Array(self.NumData//self.numfold)
        self.NumTest = 0
        self.NumTrain = 0
            
        for i in range(self.NumData):
        #round 1 fold 1,2,3 เป็น train set และ fold4 เป็น test set
            if (i%self.numfold) == k:
                self.Input_Test[self.NumTest] = self.Input_array[i]
                self.Target_Test[self.NumTest] = self.Target_array[i]
                self.NumTest += 1
            else:
                self.Input_Train[self.NumTrain] = self.Input_array[i]
                self.Target_Train[self.NumTrain] = self.Target_array[i]
                self.NumTrain += 1
        print('Number of Train data: ',self.NumTrain)
        print('Number of Test data: ',self.NumTest)

        random.seed(12)
            #to matrix
        self.Input = Matrix(len(self.Input_Train),len(self.Input_Train[0]))
        for r in range(len(self.Input_Train)):
            for c in range(len(self.Input_Train[0])):
                self.Input[r,c] = self.Input_Train[r][c]
        self.Input = self.Input.transpose()
        print("Input shape",self.Input.shape())

        self.Target = Matrix(len(self.Target_Train) ,1)
        for r in range(len(self.Target_Train)):
            for c in range(1):
                self.Target[r,c] = self.Target_Train[r]
        self.Target = self.Target.transpose()
        print("Target shape",self.Target.shape())
            
         
        print("Shape of Cleaned Input :",self.NumData,self.NumInput)
            #Number of Hidden Layer = 1
        self.NumHidden = round(2/3 * self.NumInput + self.NumOutput)
            # กำหนดค่าเริ่มต้นของ Weight
        self.weights_input_hidden = self.random_Matrix(-0.1,0.1,(self.NumInput, self.NumHidden))
        self.weights_hidden_output = self.random_Matrix(-0.1,0.1,(self.NumHidden, self.NumOutput))
        

    def fit(self,epoch,learn_rate,numfold,k=None):
        if k ==None:
            k = (numfold-1)
        self.train_test_set(numfold,k)
        # กำหนดจำนวน Epoch
        self.epochs = epoch
        # กำหนดค่า learning rate
        self.alpha = learn_rate
        self.losses = []
        for epoch in range(self.epochs):
            # random ลำดับของ input ที่จะถูกป้อนเข้า nerual net เพื่อใช้ในการ train
            self.Train_seq = self.random_permutation(self.NumTrain)
            
            # Start Train
            for i in range(self.NumTrain):
            
                #########################
                ## Forward Propogation ##
                #########################""
                # calculating hidden layer activations
                #Input[:,Train_seq[0]].reshape(NumInput,1)
                g = Matrix(self.NumInput,1)
                for i in range(self.NumInput):
                    g[i,0] =self.Input[i,self.Train_seq[0]]
                # calculate hidden units and activated hidden units
                self.Hidden = self.dot_product(self.weights_input_hidden.transpose(),g)
                #Hidden = sigmoid(Hidden)
                for r in range(self.Hidden.numRows()):
                    for c in range(self.Hidden.numCols()):
                        self.Hidden[r,c] = round(self.sigmoid(self.Hidden[r,c]),8)

                # calculating the output
                # at output layer
                self.outputLayer_linearTransform = self.dot_product(self.weights_hidden_output.transpose(),self.Hidden)
                self.Output = Matrix(self.outputLayer_linearTransform.numRows(),self.outputLayer_linearTransform.numCols())
                for r in range(self.Output.numRows()):
                    for c in range(self.Output.numCols()):
                        self.Output[r,c] = round(self.sigmoid(self.outputLayer_linearTransform[r,c]),8)
                
                ##########################
                ## Backward Propagation ##
                ##########################
                # calculating error
                #Error = np.square(Target[:,Train_seq[i]].reshape(NumOutput,1) - Output) / 2
                self.y = Matrix(self.NumOutput,1)
                for i in range(self.NumOutput):
                    self.y[i,0] =float(self.Target[i,self.Train_seq[0]])
                self.e = self.y-self.Output

                self.Error = Matrix(self.NumOutput,1)
                for r in range(self.Error.numRows()):
                    for c in range(self.Error.numCols()):
                        self.Error[r,c] = round((self.e[r,c]**2)/2,8)

                # calculating rate of change of error w.r.t weight between hidden and output layer
                self.Error_wrt_output = Matrix(self.NumOutput,1)
                for i in range(self.NumOutput):
                    self.Error_wrt_output[i,0] = float(self.Target[i,self.Train_seq[0]])
                self.Error_wrt_output = self.Error_wrt_output - self.Output
                for r in range(self.Error_wrt_output.numRows()):
                    for c in range(self.Error_wrt_output.numCols()):
                        self.Error_wrt_output[r,c] = - self.Error_wrt_output[r,c]
                        
                #Error_wrt_output = -(Target[:,Train_seq[i]].reshape(NumOutput,1) - Output)
                self.scale = Matrix(self.Output.numRows(),self.Output.numCols())
                for r in range(self.scale.numRows()):
                    for c in range(self.scale.numCols()):
                        self.scale[r,c] = round((1-self.Output[r,c]),8)
                self.Output_wrt_outputLayer_LinearTransform = self.multiply(self.Output,self.scale)
                #Output_wrt_outputLayer_LinearTransform = np.multiply(Output, (1 - Output))
                self.OutputLayer_LinearTransform_wrt_weights_hidden_output = self.Hidden
                #
                self.Error_wrt_weights_hidden_output = self.dot_product(self.OutputLayer_LinearTransform_wrt_weights_hidden_output,
                                                        (self.Error_wrt_output * self.Output_wrt_outputLayer_LinearTransform).transpose())
                #Error_wrt_weights_hidden_output = np.dot(OutputLayer_LinearTransform_wrt_weights_hidden_output,
                                                        #(Error_wrt_output * Output_wrt_outputLayer_LinearTransform).T,)

                # calculating rate of change of error w.r.t weights between input and hidden layer
                self.OutputLayer_LinearTransform_wrt_hiddenLayer_activations = self.weights_hidden_output
                #
                self.scale2 = Matrix(self.Hidden.numRows(),self.Hidden.numCols())
                for r in range(self.scale2.numRows()):
                    for c in range(self.scale2.numCols()):
                        self.scale2[r,c] = round((1-self.Hidden[r,c]),8)
                # rate of change of hidden layer activations w.r.t. Z1
                self.HiddenLayer_activations_wrt_hiddenLayer_linearTransform =  self.multiply(self.Hidden,self.scale2)
                #HiddenLayer_activations_wrt_hiddenLayer_linearTransform = np.multiply(Hidden, (1 - Hidden))
                
                self.HiddenLayer_linearTransform_wrt_weights_input_hidden = Matrix(self.NumInput,1)
                for i in range(self.NumInput):
                    self.HiddenLayer_linearTransform_wrt_weights_input_hidden[i,0] = self.Input[i,self.Train_seq[0]]
                #HiddenLayer_linearTransform_wrt_weights_input_hidden = Input[:,Train_seq[i]].reshape(NumInput,1)



                self.Error_wrt_weights_input_hidden = self.dot_product(self.HiddenLayer_linearTransform_wrt_weights_input_hidden,
                    (self.HiddenLayer_activations_wrt_hiddenLayer_linearTransform
                        * self.dot_product(self.OutputLayer_LinearTransform_wrt_hiddenLayer_activations,
                        (self.Output_wrt_outputLayer_LinearTransform * self.Error_wrt_output),)).transpose())

                # updating the weights
                self.ooi = self.Error_wrt_weights_hidden_output.scaleBy(self.alpha)
                self.weights_hidden_output =  self.weights_hidden_output - self.ooi    
                #weights_hidden_output = weights_hidden_output - alpha * Error_wrt_weights_hidden_output
                self.oi = self.Error_wrt_weights_input_hidden.scaleBy(self.alpha)
                self.weights_input_hidden = self.weights_input_hidden - self.oi
                #weights_input_hidden = weights_input_hidden - alpha * Error_wrt_weights_input_hidden

            # print error at every 100th epoch
            self.epoch_loss = self.average_Matrix(self.Error)
            if epoch % 100 == 0:
                print(f"Error at epoch {epoch} is {self.epoch_loss:.5f}")

            # appending the error of each epoch
            self.losses.append(self.epoch_loss)
        import matplotlib.pyplot as plt
        x =[]
        for i in range(self.epochs):
            x.append(i)
        plt.plot(x, self.losses)
    
    def test_model(self):
        #Input = Input_Test.T
        self.Input = Matrix(len(self.Input_Test),len(self.Input_Test[0]))
        for r in range(len(self.Input_Test)):
            for c in range(len(self.Input_Test[0])):
                self.Input[r,c] = self.Input_Test[r][c]
        self.Input = self.Input.transpose()

        #Target = Target_Test.T

        self.Target = Matrix(len(self.Target_Test) ,1)
        for r in range(len(self.Target_Test)):
            for c in range(1):
                self.Target[r,c] = self.Target_Test[r]
        self.Target = self.Target.transpose()

            
        #########################
        ## Forward Propogation ##
        #########################
        # calculating hidden layer activations
        self.Hidden = self.weights_input_hidden.transpose()*self.Input
        #Hidden = sigmoid(Hidden)
        for r in range(self.Hidden.numRows()):
            for c in range(self.Hidden.numCols()):
                self.Hidden[r,c] = round(self.sigmoid(self.Hidden[r,c]),8)

        # calculating the output
        self.Output = self.weights_hidden_output.transpose()* self.Hidden
        for r in range(self.Output.numRows()):
            for c in range(self.Output.numCols()):
                self.Output[r,c] = round(self.sigmoid(self.Output[r,c]),8)

        #check ผล
        self.Class_Output=Array(self.NumTest)
        self.Class_Target=Array(self.NumTest)
        for r in range(1):
            for c in range(self.NumTest):
                self.Class_Output[c] = round(self.Output[r,c])
                self.Class_Target[c] = round(self.Target[r,c])

        #Class_Output = argmax(Output_arr)
        #Class_Target = argmax(Target_arr)
        self.correct = 0
        self.miss = 0
        #confusion matrix (row=Actual, col=Predicted)
        #conf_matrix = np.zeros((NumOutput,NumOutput),dtype='i')
        self.conf_matrix = Matrix(2,2)
        self.conf_matrix.clear(0)
        for i in range(self.NumTest):
            if self.Class_Output[i] == self.Class_Target[i]:
                self.correct += 1
            else:
                self.miss += 1
            self.conf_matrix[self.Class_Target[i],self.Class_Output[i]] += 1
        print('Confusion Matrix (row=Actual, col=Predicted)')
        print(self.conf_matrix)
        print('accuracy = {0:.2f}'.format(self.correct/self.NumTest))
        return self.correct/self.NumTest

    def fit_average(self,epoch,learn_rate,flod):
        self.acc=[]
        self.numfold = flod
        for k in range(self.numfold):
            self.fit(epoch,learn_rate,self.numfold,k)
            self.acc.append(self.test_model())
        self.Accuracy = round(self.average(self.acc) *100,1)
        print ("Num of test= {0} \nAverage Accuracy = {1}%".format(len(self.acc),self.Accuracy))
            
        

    # กำหนด Acitvate function ที่จะนำมาใช้
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))
       

    def random_permutation(self,size):
        n =size
        perm=[]
        for i in range(n):
            perm.append(i)
        for j in range(n):
            t =[]
            for k in range(j,n):
                t.append(k)
            x = random.sample(t,1)
            y = x[0] 
            old = perm[j]
            perm[j] = perm[y]
            perm[y] = old
        return perm
    
    def dot_product(self,M1,M2):
        x = 0
        new_Matrix = Matrix(M1.shape()[0],M2.shape()[1])
        for r in range (new_Matrix.numRows()):
            for c in range (new_Matrix.numCols()):
                for i in range(M1.shape()[1]):
                    x += float(M1[r,i])*float(M2[i,c])
                    new_Matrix[r,c] = round(x,8)
                x = 0
        return new_Matrix
    
    def multiply(self,M1,M2):
        newMatrix = Matrix(M1.numRows(), M1.numCols())
        # Add the corresponding elements in the two matrices.
        for r in range(newMatrix.numRows()):
            for c in range(newMatrix.numCols()):
                newMatrix[r, c] = round(M1[r,c] * M2[r,c] , 8)
        return newMatrix

    def average_Matrix(self,Ma):
        sum_num = 0
        count = 0 
        for r in range(Ma.numRows()):
            for c in range (Ma.numCols()):
                sum_num += Ma[r,c]
                count +=1
        avg = sum_num / count
        return round(avg,8)

    def random_Matrix(self,numLow,numHigh,size):
        unsorted_seq = Matrix(size[0],size[1])
        for r in range(0 , size[0] ):
            for c in range(0 , size[1] ):
                unsorted_seq[r,c] = round(random.uniform( numLow, numHigh ),8)
        return unsorted_seq

    def average(self,arr):
        sum_num = 0
        for t in arr:
            sum_num = sum_num + float(t)
        avg = sum_num / len(arr)
        return round(avg,3)