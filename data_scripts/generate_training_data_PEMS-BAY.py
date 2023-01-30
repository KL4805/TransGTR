import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))


def standard_transform(data: np.array, output_dir: str, train_index: list, history_seq_len: int, future_seq_len: int) -> np.array:
    """Standard normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.
        history_seq_len (int): historical sequence length.
        future_seq_len (int): future sequence length.

    Returns:
        np.array: normalized raw time series data.
    """

    # data: L, N, C, C=1
    data_train = data[:train_index[-1][1], ...]

    mean, std = data_train[..., 0].mean(), data_train[..., 0].std()

    print("mean (training data):", mean)
    print("std (training data):", std)
    scaler = {}
    scaler["func"] = re_standard_transform.__name__
    scaler["args"] = {"mean": mean, "std": std}
    # label to identify the scaler for different settings.
    with open(output_dir + "/scaler_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        return (x - mean) / std

    data_norm = normalize(data)
    return data_norm

def re_standard_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:
    """Standard re-transformation.

    Args:
        data (torch.Tensor): input data.

    Returns:
        torch.Tensor: re-scaled data.
    """

    mean, std = kwargs["mean"], kwargs["std"]
    data = data * std
    data = data + mean
    return data

def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.
    Default settings of METR-LA dataset:
        - Normalization method: standard norm.
        - Dataset division: 7:1:2.
        - Window size: history 12, future 12.
        - Channels (features): three channels [traffic speed, time of day, day of week]
        - Target: predict the traffic speed of the future 12 time steps.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path

    # read data
    df = pd.read_hdf(data_file_path)
    data = np.expand_dims(df.values, axis=-1)

    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    l, n, f = data.shape
    print(l, n, f)
    num_samples = l - (history_seq_len + future_seq_len) + 1
    # keep same number of validation and test samples with Graph WaveNet (input 12, output 12)
    test_num_short = 10419
    valid_num_short = 5209
    train_num_short = num_samples - valid_num_short - test_num_short
    # train_num_short = round(num_samples * train_ratio)
    # valid_num_short = round(num_samples * valid_ratio)
    # test_num_short = num_samples - train_num_short - valid_num_short
    print("number of training samples:{0}".format(train_num_short))
    print("number of validation samples:{0}".format(valid_num_short))
    print("number of test samples:{0}".format(test_num_short))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index = index_list[train_num_short +
                            valid_num_short: train_num_short + valid_num_short + test_num_short]

    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)

    # add external feature
    feature_list = [data_norm]
    if add_time_of_day:
        # numerical time_of_day
        tod = (
            df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # dump data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(data, f)
    # copy adj
    # shutil.copyfile(graph_file_path, output_dir + "/adj_mx.pkl")


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 2016
    FUTURE_SEQ_LEN = 12

    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0]                   # target channel(s)

    DATASET_NAME = "PEMS-BAY"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.h5".format(DATASET_NAME)
    GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                        default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    args_metr = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args_metr).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if not os.path.exists(args_metr.output_dir):
        os.makedirs(args_metr.output_dir)
    generate_data(args_metr)