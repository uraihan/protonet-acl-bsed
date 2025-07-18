import os
import glob
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ProtoDataset(Dataset):
    def __init__(self, path: str, config):
        self.path = path
        # self.features = features # TODO: see if this is still needed

        # Sampling parameters
        self.segment_len = config.params.segment_len
        self.length = int(3600 * 8 / self.segment_len)
        self.fps = self.config.sr / self.config.hop_len

        self.csv_files = self.get_all_csv()
        self.class_name = self.get_class_name(path)

    def __len__(self):
        return self.length

    # TODO: define function __getitem__

    def select_segment(self, start, end, feature, seg_len):
        start, end = int(start * self.fps), int(end * self.fps)
        if start < 0:
            start = 0

        total_duration = end - start
        if total_duration < seg_len:
            y = feature[start:end]
            tile_times = np.ceil(seg_len / total_duration)
            y = np.tile(y, (int(tile_times), 1))
            y = y[:seg_len]
        else:
            randomize = np.random.uniform(low=start, high=end -
                                          seg_len)
            y = feature[int(randomize): int(randomize) + seg_len]

        if y.shape[0] != seg_len:
            print(f"Shape error! Padding. {y.shape} {seg_len} {
                  start} {end} {randomize} {feature.shape}")
            y = np.pad(y, ((0, seg_len - y.shape[0]), (0, 0)))

        return y

    def get_all_csv(self):
        return [f for f in glob.glob(f"{self.path}/Training_Set/*/*.csv")]

    def get_meta(self):
        for file in self.csv_files:
            df_pos = self.get_df_pos(file)
            start_time, end_time = self.get_time(df_pos)

    def get_df_pos(self, file):
        df = pd.read_csv(file, header=0, index_col=False)
        return df[(df == "POS").any(axis=1)]

    def get_time(self, df):
        # 25ms margin around onset and offset
        # TODO: ask why
        df.loc[:, "Starttime"] = df["Starttime"] - 0.025
        df.loc[:, "Endtime"] = df["Endtime"] + 0.025

        # convert to frame per second (fps)
        start_time = [start for start in df["Starttime"]]
        end_time = [end for end in df["Endtime"]]
        return start_time, end_time

    def get_class_name(self, file):
        split_name = file.split("/")
        return split_name[-2]
    # def positive_sample(self, class_name):
    #     segment_idx = np.random.randint(len())
