# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor


# BATCH_SIZE = 4096
# EPOCHS=10
# LR = 0.001

# class FeedForearNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.dense_layers = nn.Sequential(
#             nn.Linear(28*28, 256),
#             nn.ReLU(),
#             nn.Linear(256, 10)
#         )
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input_data):
#         flattened_data = self.flatten(input_data)
#         logits = self.dense_layers(flattened_data)
#         predictions = self.softmax(logits)
#         return predictions
        
# def download_mnist():
#     train_data = datasets.MNIST(
#         root="data",
#         download=True,
#         train=True,
#         transform=ToTensor()
#     )
#     valid_data = datasets.MNIST(
#         root="data",
#         download=True,
#         train=False,
#         transform=ToTensor()
#     )
#     return train_data, valid_data

# def train_epoch(model, dataloader, loss_fn, optimizer, device):
#     for inputs, targets in dataloader:
#         inputs, targets = inputs.to(device), targets.to(device)

#         predictions = model(inputs)
#         loss = loss_fn(predictions, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"loss: {loss.item()}")
    

# def train(model, dataloader, loss_fn, optimizer, device, epochs):
#     for i in range(epochs):
#         print(f"epoch {i+1}")
#         train_epoch(model, dataloader, loss_fn, optimizer, device)
#         print("-------------------")
#     print("training is done!")


# if __name__ == "__main__":
#     train_data,_ = download_mnist()
#     train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
#     device = ("cuda" if torch.cuda.is_available else "cpu")
#     print("current device: {}".format(device))
#     feed_forward_net = FeedForearNetwork().to(device)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr=LR)
#     train(feed_forward_net, train_dataloader, loss_fn, optimizer, device, EPOCHS)
#     torch.save(feed_forward_net.state_dict(), "feed_forward_net.pth")
#     torch.save(feed_forward_net, "feed_forward_net_pt.pt")
#     print("model saved!")



import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 4096
EPOCHS = 10
LR = 0.001

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        # Removed softmax layer as it's included in CrossEntropyLoss

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        return logits

def download_mnist():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    valid_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, valid_data

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train(model, train_dataloader, valid_dataloader, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        valid_loss, valid_acc = validate(model, valid_dataloader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
        print("-------------------")
    print("Training is done!")

if __name__ == "__main__":
    train_data, valid_data = download_mnist()
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    feed_forward_net = FeedForwardNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr=LR)

    train(feed_forward_net, train_dataloader, valid_dataloader, loss_fn, optimizer, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feed_forward_net.pth")
    torch.save(feed_forward_net, "feed_forward_net_pt.pt")
    print("Model saved!")