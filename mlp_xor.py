import numpy as np

class MLP:
    def __init__(self, train_data, y_true, weights_01=None, weights_12=None, b01=None, b12=None, lr=0.1, num_epochs=100, num_input=2, num_hidden=2, num_output=1):
        self.train_data = train_data
        self.y_true = y_true
        self.lr = lr
        self.num_epochs = num_epochs

        # valeu, python.
        if weights_01 or weights_12 or b01 or b12  == None:
            self.weights_01 = np.random.uniform(size=(num_input, num_hidden))
            self.weights_12 = np.random.uniform(size=(num_hidden, num_output))
            self.b01 = np.random.uniform(size=(1, num_hidden))
            self.b12 = np.random.uniform(size=(1, num_output))

    def _sigmoid(self, x):
        """
        Implementa a função sigmoide
        Essa é uma função de ativação possível para os nós da rede neural.
        """
        return 1 / (1 + np.exp(-x))

    def _delsigmoid(self, x):
        """
        A derivada da função sigmoide em relação a x.
        """
        return x * (1 - x)

    def forward(self, data):
        """
        Implementa a etapa de inferência (feedforward) do perceptron.
        """
        # forward pass da hidden layer
        self.hidden_ = np.dot(data, self.weights_01) + self.b01
        self.hidden_out = self._sigmoid(self.hidden_)

        # forward pass da output layer
        self.output_ = np.dot(self.hidden_out, self.weights_12) + self.b12
        self.output_final = self._sigmoid(self.output_)

        return self.output_final

    def update_weights(self):
        output_errors = self.y_true - self.output_final

        delta_output = output_errors * self._delsigmoid(self.output_final)
        delta_hidden = delta_output.dot(self.weights_12.T) * self._delsigmoid(self.hidden_out)

        # gradient da output layer
        grad12 = self.hidden_out.T.dot(delta_output)

        # gradient da hidden layer
        grad01 = self.train_data.T.dot(delta_hidden)

        # retropopagação - atualiza weights and biases
        self.weights_01 += self.lr * grad01
        self.weights_12 += self.lr * grad12
        self.b01 += self.lr * np.sum(delta_hidden, axis=0, keepdims=True)
        self.b12 += self.lr * np.sum(delta_output, axis=0, keepdims=True)

    def fit(self):
        for epoch in range(self.num_epochs):
            self.forward(self.train_data)
            self.update_weights()

    def classify(self, data):
        output = self.forward(data)
        return 1 if output >= 0.5 else 0

def cli_run_mlp_na_mao():
    input_data = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_true = np.array([[0], [1], [1], [0]])

    mlp = MLP(train_data=input_data, y_true=y_true, lr=0.5, num_epochs=10000)
    mlp.fit()

    # list comprehension com forward pass pra cada porta lógica
    [print(f"Input: {x}, Output predito: {mlp.classify(x)}") for x in input_data]
