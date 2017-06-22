from numpy import exp, array, random, dot


class NeuralNetwork():
    
    def __init__(self, ):
        random.seed(1)
        self.synamptic_weights = 2 * random.random((3,1)) - 1

    def _sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def _sigmoid_derivative(self ,x):
        return x * (1 - x)


    def train(self, training_set_inputs,training_set_outputs,number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.predict(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot (training_set_inputs.T,error * self._sigmoid_derivative(output))
            self.synamptic_weights += adjustment 
            
        
    def predict(self,inputs):
        return self._sigmoid(dot(inputs,self.synamptic_weights))
        

if __name__ == "__main__":
    
    neural_network = NeuralNetwork()
    
    print ("Random starting synaptic weights: ")
    print (neural_network.synamptic_weights)

    training_set_inputs = array ([ [0,0,1] , [1,1,1] , [1,0,1] , [0,1,1] ])
    training_set_outputs = array ( [[0,1,1,0]]).T 

    neural_network.train(training_set_inputs,training_set_outputs,10000)

    print ("new synaptic weights after training :")
    print (neural_network.synamptic_weights)


    print ('predicting:')
    print (neural_network.predict(array([1,0,0])))
