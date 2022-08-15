#---------- Imported libraries ----------

import numpy as np
from random import shuffle
from datetime import datetime



#---------- Layer class ----------

class Layer:

    def __init__(self, input_vector_size : int, layer_size : int, activation_func : str, is_first : bool = False) -> None:

        # The first layer in the neural network will not have weights and biases as it simply functions as input 
        if not is_first:
            self.weights = np.random.randn(layer_size, input_vector_size)
            self.biases = np.random.randn(layer_size)

            if activation_func == "sigmoid": self.activation_func = self.sigmoid
            elif activation_func == "sign": self.activation_func = self.sign
            elif activation_func == "tanh": self.activation_func = self.tanh
        
        else:
            self.biases = np.zeros(layer_size)
        self.is_first = is_first


    def size(self) -> int:
        return self.biases.size



    #---------- Core mathematical functions ----------

    def z(self, vector : np.ndarray, neuron : int) -> float:
        output = 0
        for index in range(self.weights[neuron].size):
            # vector[index] * self.weights[neruon][index] will adjust the input for the weight for the given neuron
            output += vector[index] * self.weights[neuron][index]

        # Adding the bias of the neuron to the output value
        output += self.biases[neuron]

        return output

    
    def dzdw(self, input_vector : np.ndarray, index : int) -> float:
        return input_vector[index]


    def dzdpa(self, neuron : int, weight : int) -> float:
        return self.weights[neuron][weight]

    
    def dzdb(self) -> float:
        return 1

    
    def dadz(self, z : float) -> float:
         
        if self.activation_func == self.sigmoid:
            return np.exp(-z)/((1 + np.exp(-z))**2)
        
        elif self.activation_func == self.sign:
            return (z * np.cos(z) - np.sin(z))/(z**2)

        elif self.activation_func == self.tanh:
            return 4/((np.exp(z) + np.exp(-z))**2)
    

    def dcda(self, z : float, label : float) -> float:
        return 2 * (self.activation_func(z) - label)
        

    
    #---------- Activation functions ----------

    def sigmoid(self, z : float) -> float:
        return 1/(1 + np.exp(-z))

    
    def sign(self, z : float) -> float:
        return np.sin(z)/z

    
    def tanh(self, z : float) -> float:
        return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))


    #---------- Core functions ----------

    def forward_vector(self, vector : np.ndarray) -> np.ndarray:

        if self.is_first: return vector

        output_vector = np.zeros(self.biases.size)
     
        for neuron in range(self.biases.size):
            
            # passing on the output from the neuron to the output vector
            z_value = self.z(vector, neuron)
            output_vector[neuron] = self.activation_func(z_value)
        
        return output_vector


    def adjust_weights_and_biases(self, delta_weights : np.ndarray, delta_biases : np.ndarray) -> None:
        self.weights += delta_weights
        self.biases += delta_biases
        

    def backpropogate_layer(self, labels : np.ndarray, input_vector : np.ndarray) -> np.ndarray:

        delta_weights = np.zeros(shape=self.weights.shape)
        delta_biases = np.zeros(shape=self.biases.shape)
        delta_input_vector = np.zeros(shape=input_vector.shape)

        for neuron in range(self.biases.size):
            # Determining the z value for the given neuron for the provided input vector
            z = self.z(input_vector, neuron)
            dcdz = self.dadz(z) * self.dcda(z, labels[neuron])
            
            # Using the chain rule to determine the partial derivative of the cost function with respect to the bias (All values are subtracted due to gradient descent)
            delta_biases[neuron] -= self.dzdb() * dcdz

            for index in range(self.weights[neuron].size):
                # Using the chain rule to determine the partial derivative of the cost function with respect to the weight (index gives the index of the weight)
                delta_weights[neuron][index] -= self.dzdw(input_vector, index) * dcdz
                delta_input_vector[index] -= self.dzdpa(neuron, weight=index) * dcdz

        return delta_weights, delta_biases, delta_input_vector



#---------- ANN class ----------

class ANN:

    def __init__(self, topology : list, activation_function : str, epochs : int, epsilon : float) -> None:
        self.layers = np.array([Layer(topology[(i-1)*(i>0)], topology[i], activation_func=activation_function, is_first=i==0) for i in range(len(topology))])
        self.a = activation_function
        self.epochs = epochs
        self.epsilon = epsilon

    
    def forward_vector(self, input_vector : np.ndarray) -> np.ndarray:

        if input_vector.size != self.layers[0].size():
            raise IndexError(f"Size of the input vector ({input_vector.size}) does not fit the size of the first layer ({self.layers[0].size()})")

        for layer in self.layers:
            # The input vector for the next layer is that outputted from the previous one
            input_vector = layer.forward_vector(input_vector)

        return input_vector


    def backpropogate_vector(self, input_vector : np.ndarray, labels : np.ndarray, delta_weights : list, delta_biases : list) -> np.ndarray:

        # A list of the input vectors to the different layers
        input_vectors = []
         # Getting the output vector from all layers
        for layer in self.layers[:-1]:
            input_vector = layer.forward_vector(input_vector)
            input_vectors.append(input_vector)

        output_vectors = input_vectors[1:] + [labels]

        for i in range(len(output_vectors)-1, 0, -1):

            res = self.layers[i+1].backpropogate_layer(output_vectors[i], input_vectors[i])
            delta_weights[i] += res[0]
            delta_biases[i] += res[1]
            output_vectors[i-1] += res[2]*self.epsilon

        return delta_weights, delta_biases


    def fit(self, vectors : list, labels : list) -> None:
            # The sum of the adhustments to be made to the neural network for the given epoch
            delta_weights = [np.zeros(shape=layer.weights.shape) for layer in self.layers[1:]]
            delta_biases = [np.zeros(shape=layer.biases.shape) for layer in self.layers[1:]]

            for sample in range(len(vectors)):
                self.backpropogate_vector(vectors[sample], labels[sample], delta_weights, delta_biases)

            for layer in range(len(delta_biases)):
                self.layers[1+layer].adjust_weights_and_biases(delta_weights[layer]*self.epsilon, delta_biases[layer]*self.epsilon)


    def predict(self, vectors : list) -> list:
        # Predict the class for a list vectors
        predictions = [self.forward_vector(vector) for vector in vectors]

        for i in range(len(predictions)):
            max_index = predictions[i].argmax()
            predictions[i] = np.zeros(predictions[i].size)
            predictions[i][max_index] = 1

        return predictions
            

    def score(self, vectors : list, labels : list) -> float:
        # Determine the accuracy of the neural network´s predictions for a list of vectors
        acc = 0
        predictions = self.predict(vectors)

        for i in range(len(labels)):
            acc += np.array_equal(predictions[i], labels[i])

        return acc/len(vectors)



#---------- NeuralNetwork class ----------

class NeuralNetwork():

    def __init__(self, topology : list, activation_function : str, epochs : int, epsilon : float, show_progression : bool = False) -> None:
        # Initiating 50 neural networks with similar topoloogy, activation function, and learning rate
        self.topology = topology
        self.activation_function = activation_function
        self.epochs = epochs
        self.epsilon = epsilon
        self.show_progress = show_progression

        self.ANN = None


    def fit(self, training_data : list, labels : list) -> None:
         # Initiating 50 neural networks with similar topoloogy, activation function, and learning rate
         # It is neceassary to initiate several networks as graident descent is prone to choosing local minimas
        nets = [ANN(self.topology, self.activation_function, self.epochs, self.epsilon) for i in range(100)]

        c = 1
        # Trainig the networks on the provided data
        for i in range(len(nets)): 
            nets[i].fit(training_data, labels)
            if ((i >= c * 10) * self.show_progress):
                print(str(datetime.now())[:-7] + f" - Training at {c}0 %")
                c += 1

        # Determining the precision of the neural networks
        scores = np.zeros(len(nets))
        for i in range(len(nets)):
            scores[i] += nets[i].score(training_data, labels)
        #scores = np.array([net.score(training_data, labels) for net in nets])

        # Selecting the neural network with the highes accuracy
        self.ANN = nets[scores.argmax()]


    def predict(self, vectors : list, labels) -> list:
        if self.ANN is not None: return self.ANN.score(vectors, labels)
        raise NotImplemented("Train the neural network to proceed")


    def score(self, trainig_data : list, labels : list) -> float:
        if self.ANN is not None: return self.ANN.score(trainig_data, labels)
        raise NotImplemented("Train the neural network to proceed")



#---------- Util functions ----------

def get_data(filename : str, label_first : bool = False) -> tuple:
    training_data, labels = [], []
    samples = []

    # Retrieving dataset values
    with open(filename, 'r') as file:
        file.readline()
        for line in file.readlines():
            samples.append(line.split(","))
        
    # Finding the unique lables in the dataset
    label_index = -1 + label_first
    size = len(samples)
    classes = []
    for sample in samples:
        label = sample[label_index].strip()
        if label not in classes: classes.append(label)

    # Retrive the values and labels
    for sample in samples:
        training_data.append(np.array([float(val) for val in sample[1 + label_index: size*label_first - (not label_first)]])) 
        label = np.zeros(len(classes))
        label[classes.index(sample[label_index].strip())] = 1
        labels.append(label)

    # Normalizing data
    for col in range(training_data[0].size):
        col_vals = [sample[col] for sample in training_data]
        col_max = max(col_vals)
        col_min = min(col_vals)
        for sample in range(len(training_data)):
            training_data[sample][col] = (training_data[sample][col] - col_min)/(col_max-col_min)

    # Randomizing the order
    index = [i for i in range(len(training_data))]
    shuffle(index)
    training_data = [training_data[i] for i in index]
    labels = [labels[i] for i in index]

    return training_data, labels


def perform_cross_validation(NN : NeuralNetwork, filename : str, label_first : bool = False) -> list:

    training_data, labels = get_data(filename, label_first=label_first)
    size = len(training_data)
    sum = 0
    for k in range(5):

        # Training data
        trd, trl = [], []
        # Test data
        ted, tel = [], []

        for i in range(size):
            if i*(5/size) < k or i*(5/size) >= k+1:
                trd.append(training_data[i])
                trl.append(labels[i])
            else:
                ted.append(training_data[i])
                tel.append(labels[i])

        NN.fit(trd, trl)
        score = NN.score(ted, tel)*100
        sum += score
        print(f"Fold {k+1} with {score} % accuracy\n")

    return sum/5


def write_cross_validation_to_file():
    with open("res.csv", "a") as csv:
        csv.write("Activation function, Learning rate, epochs, Avgerage_cluster_accuracy\n")
        for af in ["sigmoid", "sign"]:
            for epochs in [10, 100]:
                for lr in [0.01, 0.1, 1]:
                    NN = NeuralNetwork([4, 4, 4, 2], af, epochs, lr)
                    csv.write(f"{af}, {epochs}, {lr}, {perform_cross_validation(NN, '/Users/larsveggeland/Documents/Python/assignment3/irisBinaryClasssification.csv')}\n")



# Simply specify desires topology, activation function, number of epochs, and learning rate
NN = NeuralNetwork([4, 2], "sigmoid", 100, 0.1, True)

# Provide the path to the datafile
perform_cross_validation(NN, "/Users/larsveggeland/Documents/Python/assignment3/irisBinaryClasssification.csv")