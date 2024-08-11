from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
import os

class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device) -> None:
        super().__init__()
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        
    def __len__(self):  
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path, backend="soundfile")
        signal = signal.to(self.device)
        signal = self._resample_signal_if_needed(signal, sample_rate)
        signal = self._mix_down_channels(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing = self.num_samples - length_signal
            last_dim_padding = (0, num_missing)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _get_audio_sample_path(self, index) -> None:
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0
        ])
        return path
    
    def _get_audio_sample_label(self, index):
        
        return self.annotations.iloc[index, 6]
    
    @staticmethod
    def _mix_down_channels(signal):
        if  signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _resample_signal_if_needed(self, signal, sample_rate ):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal


if __name__ == "__main__":
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

    print(f"There are {len(usd)} audio samples available")
    signal, label = usd[0]
 
