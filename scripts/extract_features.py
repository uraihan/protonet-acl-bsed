import librosa
import os
import glob
import numpy as np
import pandas as pd


# NOTE: At the moment this script can only extract waveform, melspec, logmel, and PCEN


class FeatureExtractor:
    def __init__(self, dataset_path, config):
        self.dataset_path = dataset_path
        self.config = config

        self.SR = config.params.sr
        self.N_FFT = config.params.n_fft
        self.HOP = config.params.hop_len
        self.N_MELS = config.params.n_mels
        self.FMAX = config.params.fmax

    # GET FEATURES FROM EXTRACTED AUDIO FILES
    def get_waveform(self, audio):
        waveform, _ = librosa.load(audio, sr=self.SR)
        waveform = waveform / np.max(np.abs(waveform))  # normalization
        return waveform

    def get_mel(self, waveform):
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.SR,
            n_fft=self.N_FFT,
            hop_length=self.HOP,
            n_mels=self.N_MELS,
            fmax=self.FMAX)
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def get_logmel(self, mel_spec):
        mel_spec = np.log(mel_spec + 1e-8)
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def get_pcen(self, waveform):
        mel_spec = librosa.feature.melspectrogram(
            y=waveform * (2**32),
            sr=self.SR,
            n_fft=self.N_FFT,
            hop_length=self.HOP,
            n_mels=self.N_MELS,
            fmax=self.FMAX
        )
        pcen = librosa.core.pcen(mel_spec, sr=self.SR)
        pcen = pcen.astype(np.float32)
        return pcen

    def get_features(self):
        print("Checking availability of the development dataset in current local folder")

        if not os.path.exists(self.dataset_path):
            print(
                "Development dataset folder can not be found\nNow retrieving .zip file from the web")
            os.system(
                "wget https://zenodo.org/record/6482837/files/Development_Set.zip?download=1")
            os.system(f"unzip {self.dataset_path}")

        training_data = [f for f in glob.glob(
            f"{self.dataset_path}/Training_Set/*/*.wav")]

        for file in training_data:
            try:
                result = {}

                result['waveform'] = self.get_waveform(file)
                result['melspec'] = self.get_mel(result['waveform'])
                result['logmel'] = self.get_logmel(result['melspec'])
                result['pcen'] = self.get_pcen(result['waveform'])

                for feat in result.keys():
                    np_path = file.replace(".wav", f"_{feat}.npy")
                    np.save(np_path, feat)
                    print(f"File {np_path} is successfully created!")
            except Exception as e:
                print(f"Encountered error in {file}. Error: {e}")
                continue


# def main():
#     import utils
#     config_file = "../config.yaml"
#     config = utils.open_yaml_namedtuple(config_file)
#     dataset_path = "../dataset/Development_Set/"
#     feat_extractor = FeatureExtractor(dataset_path, config)
#     feat_extractor.get_features()
#
#
# if __name__ == "__main__":
#     main()
