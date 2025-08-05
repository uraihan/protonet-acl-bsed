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
        self.length = int(3600 * 8 / self.segment_len)  # might need to
        # be replaced/deleted
        self.fps = self.config.sr / self.config.hop_len

        self.csv_files = self.get_all_csv()
        self.label_names = self.get_label_name(
            path)  # this might not be needed
        self.metadata = {}
        self.features = {}

        self.build_metadata()

    def __len__(self):
        return self.length

    # TODO: define function __getitem__ based on corresponding csv file

    # METADATA
    # GET ALL CORRESPONDING CSV FILES IN TRAINING SET
    def get_all_csv(self):
        return [f for f in glob.glob(f"{self.path}/Training_Set/*/*.csv")]

    # GET NAMES OF THE LABEL BASED ON FOLDER NAME
    # NOTE: This is a special case for the JD dataset (might not be needed)
    def get_label_name(self, file):
        split_name = file.split("/")
        return split_name[-2]

    # GET DATAFRAME THAT CONTAINS ONLY POSITIVE CALLS
    def get_pos_df(self, file):
        df = pd.read_csv(file, header=0, index_col=False)

        return df[(df == "POS").any(axis=1)]

    # BUILD LABEL LIST FOR EACH AUDIO FILE
    def build_label_list(self, file, pos_df, start_time):
        label_name = file.split("/")[-2]
        if "CALL" in pos_df.columns:
            label_list = [label_name] * len(start_time)
        else:
            # pos_list = df[df.apply(
            #     lambda r: r.str.contains("POS").any(), axis=1)]
            label_list = pos_df.eq("POS").dot(pos_df.columns).to_list()

        return label_list

    # GET START AND END TIME WITH ONSET/OFFSET
    def get_time(self, df):
        # 25ms margin around onset and offset
        # TODO: ask why
        df.loc[:, "Starttime"] = df["Starttime"] - 0.025
        df.loc[:, "Endtime"] = df["Endtime"] + 0.025

        # convert to frame per second (fps)
        start_time = [start for start in df["Starttime"]]
        end_time = [end for end in df["Endtime"]]
        return start_time, end_time

    # build metadata
    def build_metadata(self):
        for file in self.csv_files:
            df_pos = self.get_pos_df(file)
            start_time, end_time = self.get_time(df_pos)
            label_list = self.build_label_list(file, df_pos, start_time)

            audio_file = file.replace("csv", "wav")
            for start, end, label in zip(start_time, end_time, label_list):
                if label not in self.metadata.keys():
                    self.metadata[label]["file"] = []
                    self.metadata[label]["duration"] = []
                    self.metadata[label]["onset_offset"] = []

                self.metadata[label]["file"].append(audio_file)
                self.metadata[label]["duration"].append(end - start)
                self.metadata[label]["onset_offset"].append((start, end))

    # GET FEATURES FROM EXTRACTED AUDIO FILES

    # SELECT POSITIVE SEGMENT ON FEATURE ARRAY BASED ON CORRESPONDING METADATA
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
    # def positive_sample(self, label_name):
    #     segment_idx = np.random.randint(len())
