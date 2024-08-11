import os
import torch
import torchaudio
from cnn import CNNNetwork
from train import ANNOTATIONS_FILE, AUDIO_DIRECTORY, SAMPLE_RATE, NUM_SAMPLES
from UrbanSoundDataset import UrbanSoundDataset



class_mapping = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

def predict(model, inputs, targets, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[targets]

    return predicted, expected

if __name__ == '__main__':
    cnn = CNNNetwork()
    state_dict = torch.load("cnn.pth", weights_only=True)
    cnn.load_state_dict(state_dict)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIRECTORY, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device="cpu")
    input_data, target_data = usd[0][0], usd[0][1]
    input_data.unsqueeze_(0)
    predicted, expected = predict(cnn, input_data, target_data, class_mapping)

    print("Predicted", predicted, "expected", expected)
