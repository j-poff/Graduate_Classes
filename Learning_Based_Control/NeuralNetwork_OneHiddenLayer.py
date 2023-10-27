import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Functions    
def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_accuracy(predictions, y_test):
    passorfail = np.argmax(predictions, axis=1)
    correct = 0
    for predict in range(len(passorfail)):
        if passorfail[predict] == y_test[predict,1]:
            correct += 1
    return correct/len(passorfail)
  
class OneLayerNN:
    # Expects training to be a 1 or 0, 1 if pa
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        
        # Initialize weights and biases for input and hidden layers
        self.w_input_hidden = np.random.uniform(-0.1, 0.1, size=(input_dim, hidden_dim))
        self.b_input_hidden = np.zeros((1, hidden_dim))

        # Initialize weights and biases for hidden and output layers
        self.w_hidden_output = np.random.uniform(-0.1, 0.1, size=(hidden_dim, output_dim))
        self.b_hidden_output = np.zeros((1, output_dim))
        
    def forward(self, x):
        # Forward propagation through the network
        transformed_input_hidden = np.dot(x, self.w_input_hidden) + self.b_input_hidden
        hidden_activations = self._relu(transformed_input_hidden)
        # hidden_activations = self._sigmoid(transformed_input_hidden)
        self.hidden_activations = hidden_activations
    
        # Calculate the output of the two output layers simultaneously
        transformed_hidden_outputs = (np.dot(hidden_activations, self.w_hidden_output) 
                                      + self.b_hidden_output)
        output_activations = self._sigmoid(transformed_hidden_outputs)

        return output_activations
    
    def back_propagation(self, x, y, predictions):
        ### Gradient of loss, dE/da. Using MSE
        loss = 2 * (predictions-y)
        ### Gradient of loss for output layer 
        # dz/dw or previous_layer * da/dz sigmoid derivative or (a(l) * (1-a(l))) * 
        # dE/da derivative of loss function (defined below)
        grad_outputs = predictions * (1 - predictions) * loss # don't multply by previous 
        # layer yet! da/dz
        ### Gradient of loss for hidden layer
        # dz/dw or previous_layer * da/dz relu derivative (0 if below 0, 1 otherwise) * 
        # dE/da(l-1) which is dz/da(l-1)* da/dz* dE/da
        grad_hidden = (np.dot(grad_outputs, self.w_hidden_output.T) * 
                        (self.hidden_activations > 0)) # don't multiply by previous layer da/dz
        # grad_hidden = (np.dot(grad_outputs, self.w_hidden_output.T) * 
        #                 (self.hidden_activations *(1-self.hidden_activations)) ) #sigmoid
        ### Finish calculating gradients and update weights and bias
        # multiply previous layer (first activations) with rest of gradient
        self.w_hidden_output -= self.lr * np.dot(self.hidden_activations.T, grad_outputs) 
        # previous activations don't influence bias, just use other part of gradient
        self.b_hidden_output -= self.lr * np.sum(grad_outputs, axis=0, keepdims=True) 
        # multiply previous layer (inputs) with rest of gradient
        self.w_input_hidden -= self.lr * np.dot(x.T, grad_hidden) 
        #previous activations don't influence bias, just use other part of gradient
        self.b_input_hidden -= self.lr * np.sum(grad_hidden, axis=0, keepdims=True)  
        
    def mean_squared_error(self, y_true, y_pred):
        N = len(y_true)
        loss = (1 / (self.output_dim * N)) * np.sum((y_true - y_pred) ** 2)
        return loss
    
    def fit(self, x_train, y_train, x_test, y_test, epochs=10000):
        training_loss = []
        training_accruacy = []
        testing_accuracy = []
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(x_train)
            
            # Backpropagation
            self.back_propagation(x_train, y_train, predictions)
            
            # Compute and print the loss (optional)
            loss = self.mean_squared_error(y_train, predictions)
            training_loss.append(loss)
            training_accruacy.append(get_accuracy(predictions,y_train))
            testing_predictions = self.forward(x_test)
            testing_accuracy.append(get_accuracy(testing_predictions,y_test))
            if epoch % 1000 == 0: # every 1000 epochs, print.
                print(f"Epoch {epoch}: Loss = {loss}")   
                #print(self.w_hidden_output, self.b_input_hidden)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1,epochs+1), training_loss, label='Training Loss', marker='o')
        plt.plot(range(1,epochs+1), testing_accuracy, label='Testing Accuracy', marker='x')
        plt.plot(range(1,epochs+1), training_accruacy, label='Training Accuracy', marker='v')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy/Loss')
        plt.title('Training Loss and Validation Accuracy over Epochs')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)  
        plt.show()

    def predict(self, x):
       predictions = self.forward(x)
       return predictions

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _relu(self, x):
        return np.maximum(0, x)

#%% Load Data from csv files
train1 = pd.read_csv(r"C:\Users\poffja\Box\Fall_2023\ROB513\hw1\train1.csv",header=None)
train2 = pd.read_csv(r"C:\Users\poffja\Box\Fall_2023\ROB513\hw1\train2.csv",header=None)
test1 = pd.read_csv(r"C:\Users\poffja\Box\Fall_2023\ROB513\hw1\test1.csv",header=None)
test2 = pd.read_csv(r"C:\Users\poffja\Box\Fall_2023\ROB513\hw1\test2.csv",header=None)

#%% Training set 1
# Training from first data set
X_train = np.array(train1.iloc[:,0:5]) # Extract all data from training set
y_train = np.array(train1.iloc[:,5:7])
X_test = np.array(test1.iloc[:,0:5])
y_test = np.array(test1.iloc[:,5:7])

# Scale
# X_train = scaleData(X_train)
# X_test = scaleData(X_test)

# Parameters
learning_rate = 0.005
hidden_layers = 30

# Train Neural Network
training1_NN = OneLayerNN(5,hidden_layers,2,lr=learning_rate)
training1_NN.fit(X_train,y_train,X_test,y_test,epochs=1000)

# Make Predictions
predictions = training1_NN.predict(X_test)
accuracy = get_accuracy(predictions, y_test)
print(f'Accuracy: {accuracy*100:.1f}%')

#%% Training set 2
X_train = np.array(train2.iloc[:,0:5]) # Extract all data from training set
y_train = np.array(train2.iloc[:,5:7])
X_test = np.array(test2.iloc[:,0:5])
y_test = np.array(test2.iloc[:,5:7])

# Scale
# X_train = scaleData(X_train)
# X_test = scaleData(X_test)

# Parameters
learning_rate = 0.005 
hidden_layers = 100

# Train Neural Network
training1_NN = OneLayerNN(5,hidden_layers,2,lr=learning_rate)
training1_NN.fit(X_train,y_train,X_test,y_test,epochs=8000)

# Make Predictions
predictions = training1_NN.predict(X_test)
accuracy = get_accuracy(predictions, y_test)
print(f'Accuracy: {accuracy*100:.1f}%')


