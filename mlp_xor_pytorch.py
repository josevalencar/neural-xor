import torch
from torch import nn, Tensor
from torch.optim import SGD

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # pytorch faz todo trabalho.
        self.layers = nn.Sequential(
            # uma camada linear com 2 neurônios de entrada e 2 neurônios na camada escondida.
            nn.Linear(2, 2),
            nn.Sigmoid(),
            # uma segunda camada linear que recebe as 2 saídas da camada escondida e as transforma
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        # otimizador para fazer o backpropagation
        self.optimizer = SGD(self.parameters(), lr=0.1)        
        self.loss = nn.MSELoss()
        
    def forward(self, X):
        return self.layers(X)
    
    def fit(self, X, y_true):
        self.optimizer.zero_grad()
        y_pred = self.forward(X)
        loss = self.loss(y_true, y_pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

def cli_run_mlp_pytorch():
    input_data = Tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
    y_true = Tensor([[0], [1], [1], [0]])

    mlp = MLP()
    num_epochs = 20000
    for i in range(num_epochs):
        loss = mlp.fit(input_data, y_true)

    with torch.no_grad():
        predictions = mlp(input_data)
        print("Predições: ")
        print(predictions)