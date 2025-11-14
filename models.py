import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_epoch (model: nn.Module,
                 dataloader: utils.data.DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: nn.functional,
                 epoch: int,
                 show_progress: bool = True,
                 N: int = 30) -> tuple[float, float]:
    # Training mode
    model.train()

    # Define variables
    acc_loss = 0.
    acc_accuracy = 0.

    # Main loop
    for i, (input, labels) in enumerate(dataloader):
        input, labels = input.to(device), labels.to(device)
        # Optimization step
        # TODO: Should set_to_none be optional?
        optimizer.zero_grad(set_to_none = True)
        output = model(input)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        # Accumulation
        pred = output.argmax(dim = 1, keepdim = True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item()
        acc_loss += loss
        acc_accuracy += accuracy
        # Print iteration
        if show_progress and i % N == N - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i + 1) * len(input), len(dataloader.dataset),
                100. * (i + 1) / len(dataloader), loss.item()))

    # Compute mean of loss and accuracy
    loss = acc_loss / len(dataloader)
    accuracy = acc_accuracy / len(dataloader.dataset)

    return loss, accuracy


def test_epoch (model: nn.Module,
                dataloader: utils.data.DataLoader,
                criterion: nn.functional) -> tuple[float, float]:
    # Training mode
    model.eval()

    # Define variables
    acc_loss = 0.
    acc_accuracy = 0.

    # Main loop
    for i, (input, labels) in enumerate(dataloader):
        # Input and labels should be tensors
        input, labels = input.to(device), labels.to(device)
        # Calc
        output = model(input)
        loss = criterion(output, labels)
        # Accumulation
        pred = output.argmax(dim = 1, keepdim = True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item()
        acc_loss += loss
        acc_accuracy += accuracy

    # Compute mean of loss and accuracy
    loss = acc_loss / len(dataloader)
    accuracy = acc_accuracy / len(dataloader.dataset)

    return loss, accuracy


class ModelTemplate (nn.Module):

    # ...

    def __init__ (self) -> None:
        # Define the model's layers here. Use entry parameters if necessary
        super().__init__()
        ...
        self = self.to(device)  # No estic segur de que això sigui correcte
    
    def forward (self, x: torch.Tensor) -> torch.Tensor:
        # Compute the forward pass of the network
        ...
    

class ModelExample (nn.Module):

    conv_layers: nn.Sequential
    mlp: nn.Sequential

    def __init__ (self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
        )
        self.mlp = nn.Sequential(
            nn.Linear(8 * 8 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(-1),
        )
        self = self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers
        x = self.conv_layers(x)
        # MLP part (must reshape the data, it must be a flat array)
        bsz, nc, width, height = x.shape
        x = x.reshape((bsz, nc * width * height))
        x = self.mlp(x)
        return x
