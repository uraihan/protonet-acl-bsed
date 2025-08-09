import librosa
import os
# import glob
import numpy as np
# import pandas as pd
import h5py
import gc
from . import utils


# NOTE: At the moment this script can only extract waveform, melspec, logmel, and PCEN


class FeatureExtractor:
    def __init__(self, dataset_path, config, csv_files):
        # def __init__(self, config):
        self.dataset_path = dataset_path
        self.config = config
        self.csv_files = csv_files

        self.SR = config.params.sr
        self.N_FFT = config.params.n_fft
        self.HOP = config.params.hop_len
        self.N_MELS = config.params.n_mels
        self.FMAX = config.params.fmax
        self.features_list = config.features

    # GET FEATURES FROM EXTRACTED AUDIO FILES
    def get_waveform(self, audio):
        waveform, _ = librosa.load(audio, sr=self.SR)
        waveform = waveform / np.max(np.abs(waveform))  # normalization
        return waveform

    def get_melspec(self, waveform):
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
        """
        Extract feature from audio file and store it as .h5 files.

        The .h5 files are formatted in such structure:
            <feature1>: <data>
            <feature2>: <data>
            ...
            <featureN>: <data>
        """

        # print("Checking availability of the development dataset in current local folder")
        # if not os.path.exists(self.dataset_path):
        #     print(
        #         "Development dataset folder can not be found\nNow retrieving .zip file from the web")
        #     os.system(
        #         "wget https://zenodo.org/record/6482837/files/Development_Set.zip?download=1")
        #     os.system(f"unzip {self.dataset_path}")

        # training_data = [f for f in glob.glob(
        #     f"{self.dataset_path}/Training_Set/*/*.wav")]
        # print("Collecting all CSV files")
        # training_data = utils.get_all_csv(self.dataset_path)

        print("Extracting features from training data...")
        for file in self.csv_files:
            save_path = file.replace(".csv", ".h5")
            audio_path = file.replace(".csv", ".wav")
            # if os.path.exists(save_path):
            #     continue
            # else:
            try:
                result = {}

                result['waveform'] = self.get_waveform(audio_path)
                result['melspec'] = self.get_melspec(result['waveform'])
                result['logmel'] = self.get_logmel(result['melspec'])
                result['pcen'] = self.get_pcen(result['waveform'])

                save_file = h5py.File(save_path, 'w')
                for feat in result.keys():
                    save_file.create_dataset(feat, data=result[feat])

                save_file.close()

                # Test
                with h5py.File(save_path, 'r') as f:
                    for feat in result.keys():
                        assert (f[feat].dtype == result[feat].dtype)

                print(f"File {audio_path} is successfully processed!")
                gc.collect()

                # for feat in result.keys():
                #     np_path = file.replace(".wav", f"_{feat}.npy")
                #     np.save(np_path, feat)
                #     print(f"File {np_path} is successfully created!")
            except Exception as e:
                print(f"Encountered error in {audio_path}. Error: {e}")
                continue


# def get_features(dataset_path, config):
#     """
#     Extract feature from audio file and store it as .h5 files.
#
#     The .h5 files are formatted in such structure:
#         <feature1>: <data>
#         <feature2>: <data>
#         ...
#         <featureN>: <data>
#     """
#
#     print("Checking availability of the development dataset in current local folder")
#     if not os.path.exists(dataset_path):
#         print(
#             "Development dataset folder can not be found\nNow retrieving .zip file from the web")
#         os.system(
#             "wget https://zenodo.org/record/6482837/files/Development_Set.zip?download=1")
#         os.system(f"unzip {dataset_path}")
#
#     # training_data = [f for f in glob.glob(
#     #     f"{self.dataset_path}/Training_Set/*/*.wav")]
#     training_data = utils.get_all_csv(dataset_path)
#     feature_extractor = FeatureExtractor(dataset_path)
#
#     for file in training_data:
#         try:
#             result = {}
#
#             result['waveform'] = feature_extractor.get_waveform(file)
#             result['melspec'] = feature_extractor.get_melspec(
#                 result['waveform'])
#             result['logmel'] = feature_extractor.get_logmel(result['melspec'])
#             result['pcen'] = feature_extractor.get_pcen(result['waveform'])
#
#             save_path = file.replace(".wav", ".h5")
#             save_file = h5py.File(save_path, 'w')
#             for feat in result.keys():
#                 save_file.create_dataset(feat, data=result[feat])
#
#             save_file.close()
#
#             # Test
#             with h5py.File(save_path, 'r') as f:
#                 for feat in result.keys():
#                     assert (f[feat].dtype == result[feat].dtype)
#
#             # for feat in result.keys():
#             #     np_path = file.replace(".wav", f"_{feat}.npy")
#             #     np.save(np_path, feat)
#             #     print(f"File {np_path} is successfully created!")
#         except Exception as e:
#             print(f"Encountered error in {file}. Error: {e}")
#             continue
