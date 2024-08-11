import os
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from cnn import CNNNetwork
from UrbanSoundDataset import UrbanSoundDataset
from tqdm import tqdm

BATCH_SIZE = 4096
EPOCHS = 10
LR = 0.001
ANNOTATIONS_FILE =  os.path.join("data",
                                     "UrbanSound8K",
                                     "UrbanSound8K",
                                     "metadata",
                                     "UrbanSound8K.csv")
AUDIO_DIRECTORY = os.path.join("data",
                                "UrbanSound8K",
                                "UrbanSound8K",
                                "audio")
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(data, batch_size, shuffle_files=False):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle_files)
    return dataloader

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in progress_bar:
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
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, targets in progress_bar:
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
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss, train_acc = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        valid_loss, valid_acc = validate(model, valid_dataloader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
        print("-------------------")
    print("Training is done!")

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("using device: ", device)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIRECTORY, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    train_dataloader = create_data_loader(usd, batch_size=BATCH_SIZE, shuffle_files=True)
    valid_dataloader = create_data_loader(usd, batch_size=BATCH_SIZE)


    cnn = CNNNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

    train(cnn, train_dataloader, valid_dataloader, loss_fn, optimizer, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnn.pth")
    torch.save(cnn, "cnn_pt.pt")
    print("Model saved!")


