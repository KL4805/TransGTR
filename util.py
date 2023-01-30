from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pickle
import os
import scipy.sparse as sp
import math
import pandas as pd

def load_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data

# this is the one that should be used in train_forecast.py
class ForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str, seq_len:int, data_number=0) -> None:
        """Init the dataset in the forecasting stage.

        Args:
            data_file_path (str): data file path.
            index_file_path (str): index file path.
            mode (str): train, valid, or test.
            seq_len (int): the length of long term historical data.
        """

        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)
        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        # read index
        self.index = load_pkl(index_file_path)[mode]
        # length of long term historical data
        self.seq_len = seq_len
        # print('self.seq_len', self.seq_len) 2016
        # mask
        self.mask = torch.zeros(self.seq_len, self.data.shape[1], self.data.shape[2])
        if data_number != 0:
            self.index = self.index[-288*data_number:]
        self.data_number = data_number
        self.mode = mode

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("Can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("Can not find index file {0}".format(index_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])

        history_data = self.data[idx[0]:idx[1]]     # 12
        future_data = self.data[idx[1]:idx[2]]      # 12
        if (self.data_number == 0 and idx[1] - self.seq_len < 0) or (self.data_number != 0 and idx[1] - self.index[0][0] < self.seq_len):
            # the first condition is for full data
            # the second condition is for few-shot data
            # not enough long history data
            if self.mode != 'train':
                print("not enough long history data")
            long_history_data = self.mask
            # print("long_history_data.min", long_history_data.min())
            # print("long_history_data.max", long_history_data.max())
            
            ## if self.data_number != 0:
                ## print("idx[1] = %d, starting index = %d, end index = %d, not enough long history data" % (idx[1], self.index[0][0], self.index[-1][2]))
            # print("history data, mean %.2f, std %.2f" % (long_history_data.mean().item(), long_history_data.std().item()))

        else:
            long_history_data = self.data[idx[1] - self.seq_len:idx[1]]     # 11
            # print("enough history data")
            # print("history_data", long_history_data.shape)
        # both history data are of the same shape, (2016, num_node, 3)
        
        return future_data, history_data, long_history_data

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)

class TimeSeriesForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str, data_number=0) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)
        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        # read index
        self.index = load_pkl(index_file_path)[mode]
        if data_number != 0:
            self.index = self.index[-data_number * 288:]
        

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("Can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("Can not find index file {0}".format(index_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # continuous index
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]
        else:
            # discontinuous index or custom index
            # NOTE: current time $t$ should not included in the index[0]
            history_index = idx[0]    # list
            assert idx[1] not in history_index, "current time t should not included in the idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]

        return future_data, history_data

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)


class TimeSeriesForecastingDatasetWithLongFeat(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str, tsformer, device, data_number=0) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)
        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        # read index
        self.index = load_pkl(index_file_path)[mode]
        if data_number != 0:
            self.index = self.index[-data_number * 288:]
        
        # load long-term pattern
        self.long_term = []
        total_number = len(self.index)
        for i in range(total_number):
            # sample batch
            
            idx = self.index[i]
            history_data = self.data[idx[0]:idx[1]]
            with torch.no_grad():
                history_data = history_data.unsqueeze(0).to(device)
                hidden_states = tsformer(history_data[...,[0]])
                # print('hidden_states', hidden_states.shape)
                # if i == 0:
                    # print(hidden_states[0,0,0,:])
            self.long_term.append(hidden_states.cpu()[:,:,-1,:])
        self.long_term = torch.cat(self.long_term, dim=0) 
        print("long_term %s" % mode, self.long_term.shape)
 
        

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("Can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("Can not find index file {0}".format(index_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # continuous index
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]
            long_term = self.long_term[index]
        else:
            # discontinuous index or custom index
            # NOTE: current time $t$ should not included in the index[0]
            history_index = idx[0]    # list
            assert idx[1] not in history_index, "current time t should not included in the idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]

        return future_data, history_data, long_term

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)

class Scaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x):
        return (x * self.std) + self.mean


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = ((labels-null_val).abs() > 1e-5)
        # mask = labels!=null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = ((labels-null_val).abs() > 1e-5)
        # mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        # update if null_val is 0
        mask = ((labels-null_val).abs() > 1e-5)
        # mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

def batch_cosine_similarity(x, y):
    # 计算分母
    if len(x.shape) == 3:

        l2_x = torch.norm(x, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
        l2_y = torch.norm(y, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
        l2_m = torch.matmul(l2_x.unsqueeze(dim=2), l2_y.unsqueeze(dim=2).transpose(1, 2))
        # 计算分子
        l2_z = torch.matmul(x, y.transpose(1, 2))
        # l2_z: (batch, 207, 207)
        # print('l2_z', l2_z.shape)
        # print('l2_m', l2_m.shape)
        # cos similarity affinity matrix
        cos_affnity = l2_z / l2_m
        adj = cos_affnity
    else:
        # no batch dimension
        l2_x = torch.norm(x, dim=1, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
        l2_y = torch.norm(y, dim=1, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
        l2_m = torch.matmul(l2_x.unsqueeze(dim=1), l2_y.unsqueeze(dim=1).transpose(0, 1))
        # 计算分子
        l2_z = torch.matmul(x, y.transpose(0, 1))
        # l2_z: (batch, 207, 207)
        # print('l2_z', l2_z.shape)
        # print('l2_m', l2_m.shape)
        # cos similarity affinity matrix
        cos_affnity = l2_z / l2_m
        adj = cos_affnity
    return adj

def batch_dot_similarity(x, y):
    QKT = torch.bmm(x, y.transpose(-1, -2)) / math.sqrt(x.shape[2])
    W = torch.softmax(QKT, dim=-1)
    return W

def load_adj_csv(filename, adjtype, thresh = 0):
    adj_mx = pd.read_csv(filename).values
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_mx / std))   
    if thresh > 0:
        adj_mx = adj_mx * (adj_mx > thresh)
    print((adj_mx > 0).sum())
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx), calculate_transition_matrix(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx

def load_adj_npy(filename, adjtype, thresh=0):
    adj_mx = np.load(filename)
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_mx / std))   
    if thresh > 0:
        adj_mx = adj_mx * (adj_mx > thresh)
    # the main diagonal of HKTSM may not be all 1 
    for i in range(adj_mx.shape[0]):
        adj_mx[i][i] = 1
    print((adj_mx > 0).sum())
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx), calculate_transition_matrix(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx

def load_adj(file_path: str, adj_type: str):
    """load adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """

    try:
        # METR and PEMS_BAY
        _, _, adj_mx = load_pkl(file_path)
    except ValueError:
        # PEMS04
        adj_mx = load_pkl(file_path)
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32).todense()]
    elif adj_type == "original":
        adj = adj_mx
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx

def calculate_transition_matrix(adj: np.ndarray) -> np.matrix:
    """Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Transition matrix P
    """

    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    prob_matrix = d_mat.dot(adj).astype(np.float32).todense()
    return prob_matrix
