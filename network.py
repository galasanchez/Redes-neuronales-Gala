
#### Libraries

import random 
import numpy as np 

import mnist_loader

class Network(object): 
    def __init__(self, sizes): 

        self.num_layers = len(sizes) 
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a): #definir la función feedforward
        #es la evaluación que el algoritmo requiere para calcular la z^l en cada capa
        for b, w in zip(self.biases, self.weights): 
            a = sigmoid(np.dot(w, a)+b) #función de activación sigmoide, el nuevo valor de a es sigma con argumento aw+b
        return a #siempre tiene valor entre 0 y 1

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #definimos la función Stochastic Gradient Descent
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y): #definir algoritmo backpropagation
        #el gradiente de C nos indica la dirección de máximo cambio de la función de costo
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b #darle el nuevo valor a z
            zs.append(z) #agregar el nuevo z a la lista
            activation = sigmoid(z) #definir nueva a
            activations.append(activation) #agregar a la lista de activaciones
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta #este delta^L es el error de la última capa, a partir de aquí propaga hacia atrás para calcular delta^l
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w) #vector nabla C
        #obtengo la derivada de la función de costo C con respecto a bias y weights de todos los parámetros del modelo

    def evaluate(self, test_data): #suma de pruebas en las que la red obtiene el resultado correcto
        #prueba de funcionamiento de la red
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

#definimos funcion de costo cross-entropy
    #a es la salida de la red neuronal y es un vector 10-dimensional
    #y son las etiquetas reales del conjunto de datos de entrenamiento
    #ambos están en one-hot-encoding
    #calcula la pérdida comparando las dos distribuciones de probabilidad
def cross_entropy(a, y):
    loss = -np.sum(y * np.log(a))
    return loss

#derivada de la función cross-entropy con respecto a a
#necesaria para el entrenamiento de la red
def cross_entropy_derivative(a, y):
    return (a - y) / (a * (1 - a))

#función de activación softmax en la capa de salida
#convierte las salidas de la red en una distribución de probabilidad real (suman 1)
def softmax(z):
    ez = z-np.max(z, axis = 0) #ajusta las entradas de z para evitar que los exponentes sean demasiado grandes
    return np.exp(ez)/np.sum(ez, axis = 0) #divide e^z ebtre la suma total
#la clase con la probabilidad más alta es la predicción de la red

#### Miscellaneous functions
def sigmoid(z): #función de activación sigmoide
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z): #derivada de la sigmoide, necesaria para el backpropagation
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


##Carga de datos de entrenamiento, validación y prueba con MNIST_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

## Creación de la red neuronal con capas (x,y,z)
net = Network([784,30,10])

## Entrenamiento de la red
net.SGD(training_data, 30, 10, 0.01, test_data=test_data)