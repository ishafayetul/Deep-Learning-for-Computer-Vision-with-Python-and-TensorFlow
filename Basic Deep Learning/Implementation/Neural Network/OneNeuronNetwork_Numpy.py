import numpy as np

class OneNeuron:
    def __init__(self,input_size,output_size):
        self.row,self.col=input_size
        self.output_size=output_size

        self.w1=1 #np.random.rand(1)
        self.w2=1 #np.random.rand(1)
        self.b=0 #np.random.rand(1)
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z)) 
    
    def forward(self,x):
        self.z=self.w1*x[:,0]+self.w2*x[:,1]+self.b
        self.p=self.sigmoid(self.z)
        return self.p
    
    def calcuLoss(self,y_true,y_pred):
        epsilon=1e-15
        y_pred = [max(i, epsilon) for i in y_pred]
        y_pred = [min(i, 1-epsilon) for i in y_pred]
        y_pred = np.array(y_pred)
        return np.mean(-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred))
    
    def derivative(self,x,y_true,y_pred):
        error = y_pred - y_true
        dw1=np.mean(error*x[:,0])
        dw2=np.mean(error*x[:,1])
        db=np.mean(error)
        return dw1,dw2,db
    
    def update_weights(self,learning_rate):
        self.w1=self.w1-learning_rate*self.dw1
        self.w2=self.w2-learning_rate*self.dw2
        self.b=self.b-learning_rate*self.db
    
    def calcuAccuracy(self,y_true,y_pred):
        y_pred=(y_pred>0.5).astype(int)
        accuracy=np.mean(y_true==y_pred)
        return accuracy
    
    def predict(self,x):
        y_pred=self.forward(x)
        return y_pred

    def train(self,x,y_true,learning_rate):
        y_pred=self.forward(x)
        loss=self.calcuLoss(y_true,y_pred)
        accuracy=self.calcuAccuracy(y_true,y_pred)
        self.dw1,self.dw2,self.db=self.derivative(x,y_true,y_pred)
        self.update_weights(learning_rate)
        return loss,accuracy
    
    def fit(self,x,y_true,epochs,learning_rate):
        for i in range(epochs):
            self.loss,self.accuracy=self.train(x,y_true,learning_rate)
            if i%100==0:
                print(f"Epoch {i}, Loss: {self.loss}, Accuracy: {self.accuracy}")
            if self.loss <= 0.5298:
                break
        print(f"Training Finished at Epoch {i+1}/{epochs}, Final Loss: {self.loss}, Final Accuracy: {self.accuracy}")
    
    def getParameters(self):
        return self.w1,self.w2,self.b 
    
