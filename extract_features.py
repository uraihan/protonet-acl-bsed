import librosa
import os
import glob
import numpy as np
import utils

config_file = "./config.yaml"
config = utils.open_yaml_namedtuple(config_file)

SR = config.params.sr
N_FFT = config.params.n_fft
HOP = config.params.hop_len
N_MELS = config.params.n_mels
FMAX = config.params.fmax


def get_mel(waveform):
    mel_spec = librosa.feature.melspectrogram(
        waveform, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX)
    mel_spec = mel_spec.astype(np.float32)
    return mel_spec


def get_logmel(mel_spec):
    mel_spec = np.log(mel_spec + 1e-8)
    mel_spec = mel_spec.astype(np.float32)
    return mel_spec


def get_pcen(waveform):
    mel_spec = librosa.feature.melspectrogram(
        waveform * (2**32), sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmax=FMAX
    )
    pcen = librosa.core.pcen(mel_spec, sr=SR)
    pcen = pcen.astype(np.float323)
    return pcen


def get_waveform(audio):
    waveform = librosa.load(audio, sr=SR)
    waveform = waveform / np.max(np.abs(waveform))  # normalization
    return waveform


def main(devdata: str = config.dataset.devset):
    print("Checking availability of the development dataset in current local folder")

    if not os.path.exists(devdata):
        print(
            f"Development dataset folder can not be found\nNow retrieving .zip file from the web")
        os.system(
            "wget https://zenodo.org/record/6482837/files/Development_Set.zip?download=1")
        os.system(f"unzip {devdata}")

    training_data = [f for f in glob.glob(
        f"{devdata}/Training_Set/*/*.wav")]

    for file in training_data:
        try:
            result = {}

            result['waveform'] = get_waveform(file)
            result['melspec'] = get_mel(result['waveform'])
            result['logmel'] = get_logmel(result['melspec'])
            result['pcen'] = get_pcen(result['waveform'])

            for feat in result.keys():
                np_path = file.replace(".wav", f"_{feat}.npy")
                np.save(np_path, feat)
        except Exception:
            print(f"Encountered error in {file}")
            continue


if __name__ == "__main__":
    main()
