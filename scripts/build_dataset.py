# import glob
import pandas as pd
import numpy as np
import h5py
from . import utils
from collections import defaultdict
from tqdm import tqdm
# import utils  # FOR DEBUGGING ONLY (REMOVE AFTER DONE)
# from pathlib import Path
# import os


class DataSampler:
    def __init__(self, dataset_path, config, feature, is_single=False):
        """
        Params:
            dataset_path: Path to dataset
            config: Config object
            feature: Feature to be used for dataset sampling
            is_single: This is mainly for debugging purpose. Default
            value is False. If you want to test the data sampler only
            on one file, pass True.
        """
        self.dataset_path = dataset_path
        self.config = config
        self.sr = config.params.sr
        self.hop_mel = config.params.hop_mel
        self.fps = self.sr / self.hop_mel
        self.seg_len = int(config.params.segment_len * self.fps)
        self.seg_hop = int(self.seg_len // 2)
        self.feature = feature
        self.is_single = is_single

        # self.csv_files = self.get_all_csv(self.dataset_path)
        if self.is_single and ("csv" in dataset_path):
            self.csv_files = [self.dataset_path]
        else:
            self.csv_files = utils.get_all_csv(self.dataset_path)

    # GET ALL CORRESPONDING CSV FILES IN TRAINING SET
    # def get_all_csv(self, dataset_path):
    #     return [f for f in glob.glob(f"{dataset_path}/Training_Set/*/*.csv")]

    # GET METADATA DATAFRAME THAT CONTAINS ONLY POSITIVE CALLS
    def get_pos_df(self, file):
        df = pd.read_csv(file, header=0, index_col=False)

        return df[(df == "POS").any(axis=1)]

    # BUILD LABEL LIST FOR EACH AUDIO FILE
    def build_label_list(self, file, pos_df, start_time):
        if "CALL" in pos_df.columns:
            label_name = file.split("/")[-2]
            label_list = [label_name] * len(start_time)
        else:
            # pos_list = df[df.apply(
            #     lambda r: r.str.contains("POS").any(), axis=1)]
            label_list = pos_df.eq("POS").dot(pos_df.columns).to_list()

        return label_list

    def map_label_toint(self, label_list):
        unique_label, label_int = np.unique(label_list, return_inverse=True)

        return unique_label.tolist(), label_int.tolist()

    # GET START AND END TIME WITH ONSET/OFFSET
    def get_time(self, df):
        # 25ms margin around onset and offset
        df.loc[:, "Starttime"] = df["Starttime"] - 0.025
        df.loc[:, "Endtime"] = df["Endtime"] + 0.025

        # convert to frame per second (fps)
        start_time = [start for start in df["Starttime"]]
        end_time = [end for end in df["Endtime"]]
        return start_time, end_time

    # SELECT POSITIVE SEGMENT ON FEATURE ARRAY BASED ON CORRESPONDING METADATA
    def select_segment(self, start, end, feature, seg_len, seg_hop):
        """
        Select positive segment on a feature array based on label annotation.

        Args:
            start: Start time of the sample in seconds.
            end: End time of the sample in seconds.
            feature: Feature array of the whole audio file.
            seg_len: Desired sample segment length.
            seg_hop: Segment hop between each window.

        Returns:
            y: Sampled segment in the desired length (configurable in
            config.yaml). If the queried segment length is longer than the
            positive call duration, padding will be added at the end of the
            positive call sample.
        """
        start, end = int(start * self.fps), int(end * self.fps)
        if start < 0:
            start = 0

        total_duration = end - start
        samples = []

        if total_duration < seg_len:
            y = feature[..., start:end]
            tile_times = np.ceil(seg_len / total_duration)
            y = np.tile(y, (1, int(tile_times)))
            y = y[..., :seg_len]
            samples.append(y)
        else:
            while end - start >= seg_len:
                # rng = np.random.default_rng()
                # randomize = rng.uniform(low=start, high=end - seg_len)
                # randomize = int(randomize)
                # y = feature[..., randomize: randomize + seg_len]

                y = feature[..., start: start + seg_len]
                start += seg_hop
                samples.append(y)

        # if y.shape[1] != seg_len:
            # print(f"Shape error! Padding. Original shape: {y.shape}, segment length: {seg_len}, starttime: {
            #       start}, endtime:{end}, randomized start number:{randomize}, feature shape: {feature.shape}")
            # print(f"seg_len: {seg_len}, yshape: {y.shape[0]}")
            # y = np.pad(y, ((0, seg_len - y.shape[1]), (0, 0)))
            # y = np.tile(y, (y.shape[0], y.shape[1]))
            # print(y.shape)

        return np.array(samples)

    # MAIN SAMPLING FUNCTION
    def sample_data(self):
        # x = []
        # y = []
        dataset = defaultdict()
        for file in (pbar := tqdm(self.csv_files, ncols=50)):
            # collecting metadata
            df_pos = self.get_pos_df(file)
            start_time, end_time = self.get_time(df_pos)
            label_list = self.build_label_list(file, df_pos, start_time)
            # audio_path = file.replace("csv", "wav")

            # collecting feature array
            feature_path = file.replace("csv", "h5")
            pbar.set_postfix_str(feature_path.split("/Development_Set/")[1])
            with h5py.File(feature_path, 'r') as f:
                for start, end, label in zip(start_time, end_time, label_list):
                    sampled_data = self.select_segment(start, end,
                                                       f[self.feature],
                                                       self.seg_len,
                                                       self.seg_hop)
                    if sampled_data.shape[1] == 15:
                        print(f"Shape 15 on file {feature_path}")
                    # x.append(sampled_data)
                    # y.append(label)
                    # print(label, sampled_data.shape)
                    if label in dataset.keys():
                        dataset[label] = np.vstack(
                            (dataset[label], sampled_data))
                    else:
                        dataset[label] = sampled_data

        # dataset.keys()

        # x = np.array(x)
        # y = self.map_label_toint(y)
        x = []
        y = []

        for label, feat in dataset.items():
            if len(x) == 0:
                x = dataset[label]
            else:
                x = np.vstack((x, dataset[label]))

            y.extend([label] * dataset[label].shape[0])

        # if not self.is_single:
        # y = np.array(y, dtype="str_")
        unique_labels, y = self.map_label_toint(y)
        with h5py.File(
                f"{self.dataset_path}/{self.feature}.h5", 'w') as f:
            # print(f"feature shape: {len(x)}, label shape: {len(y)}")
            # print(f"x: {x} \ny: {y}")

            # list approach
            f.create_dataset('feature', data=x)
            f.create_dataset('labels', data=y)
            ds = f.create_dataset('unique_labels', shape=len(unique_labels),
                                  dtype=h5py.string_dtype())
            ds[:] = unique_labels

            # dictionary approach
            # for label in dataset.keys():
            # TODO: Test the file size of these two options
            #     group = f.create_group(label)
            #     group.create_dataset('feature', data=dataset[label])
            #
            #     # this one is slightly smaller
            # f.create_dataset(label, data=dataset[label])
            print(f"File {f} was successfully created!")

        # with h5py.File(self.dataset_path.replace(".csv", "test2.h5"), 'r') as f:
        #     feat = np.array(f['feature'])
        #     lbl = np.array(f['label'])
        #     print("halt")


def run(config, dataset_path, return_object=False):
    feature_list = config.features
    for feature in feature_list:
        print(f"Processing samples for feature {feature}...")
        data_sampler = DataSampler(dataset_path, config, feature)
        data_sampler.sample_data()

    if return_object:
        return data_sampler


# WARN: for debugging purpose only
# WARN: Think of a more elegant solution than this
# config_path = os.path.join(Path.cwd(), "config.yaml")
# config = utils.open_yaml_namedtuple(config_path)
# dataset_path = os.path.join(Path.cwd(), "dataset/Development_Set")
# run(config, dataset_path)
