import torch
import torch.nn as nn
import torch.optim as optim

class AE(torch.nn.Module):
    def __init__(self,L1,L2,L3,input_size,hidden_size):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, L1),
            torch.nn.ReLU(),
            torch.nn.Linear(L1, L2),
            torch.nn.ReLU(),
            torch.nn.Linear(L2, L3),
            torch.nn.ReLU(),
            torch.nn.Linear(L3, hidden_size)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, L3),
            torch.nn.ReLU(),
            torch.nn.Linear(L3, L2),
            torch.nn.ReLU(),
            torch.nn.Linear(L2, L1),
            torch.nn.ReLU(),
            torch.nn.Linear(L1,input_size)
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define the training function
def train(model, dataloader, num_epochs, learning_rate,model_path=None):
    criterion = nn.MSELoss()
    print_interval=1000
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    best_loss = 1e+5
    for epoch in range(num_epochs):
        for data in dataloader:
            recon = model(data)
            loss = criterion(recon, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % print_interval == 0:
            print('Epoch [{}/{}], Loss:{:.8e}'.format(epoch+1, num_epochs, loss.item()))
            if loss.item() < best_loss:
                best_loss = loss.item()
                print('(--> new model saved @ epoch {})'.format(epoch+1))
                best_model_state = model.state_dict()
                torch.save(best_model_state,model_path)