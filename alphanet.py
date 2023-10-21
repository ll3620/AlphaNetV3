import numpy as np
import pandas as pd
import copy
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import datetime
import statsmodels.api as sm
from tqdm import tqdm
from pandarallel import pandarallel
import logging

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

pandarallel.initialize(nb_workers=32)

def AlphaNetV3(spot_data_dict, future_data_dict, mask_spot_data, mask_future_data, id, horizon_used):
    """
    Predicts the Information Coefficient (IC) of assets using a neural network.
    
    This function encompasses the stages of data preparation, training, validation, and prediction.

    Note: While this function was originally part of a larger code framework, in this demo we are focusing on the following two parameters:

    :param future_data_dict: dictionary, each key-value pair consists of a pandas DataFrame representing one feature.
    :param mask_future_data: pandas DataFrame with boolean values, indicating whether an asset is traded at a specific timestamp.
    """

    # Train model on cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Following hyperparams are not optimal, due to be tuned.
    hyperparams = {
        'Inception1_stride': int(1440*2 // 3),
        'Inception2_stride': int(1440*2 // 6),
        'Inception_conv_length': int(6),
        'feature_num': int(14),
        'lookback_period': int(1440*2),
        'training_interval': int(20),
        'validation_interval': int(20),
        'service_interval': int(10),
        'training_length': int(1440*15),
        'validation_length': int(1440*15),
        'GRU_dropout': float(0.5),
        'GRU_hidden_dim': int(32),
        'GRU_output_dim': int(30),
        'GRU_n_layers': int(2),
        'truncated_normal_mean': int(0),
        'truncated_normal_std': int(1),
        'learning_rate': float(1e-5),
        'weight_decay': float(1e-6),
        'steplr_step_size': int(2),
        'steplr_gamma': float(0.5),
        'stopping_window': int(1),
        'stopping_patience': int(5),
    }


    class AuxiliaryFunc:
        """
        This class contains all the auxiliary functions to calculate statistics.
        """
        @staticmethod
        def cal_target(trade_value_df, trade_size_df, back_period=720):
            # Target for training, the return is calculated through vwap
            vwap = trade_value_df / trade_size_df
            rtn = vwap.shift(-back_period).fillna(method ='ffill')/vwap -1
            return rtn 
        
        @staticmethod
        def norm1_feature(data,mask_data,k, dataframe=False):
            # Normalise the features
            if dataframe:
                data = data.where(mask_data, np.nan)
                normed_data = data.parallel_apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            else:
                data = data.where(mask_data, np.nan)
                normed_data = data.parallel_apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            if k:
                k=abs(k)
                normed_data[normed_data>k]=k
                normed_data[normed_data<-k]=-k

            return  normed_data
        
        @staticmethod
        def IC(alpha_return, ground_truth_return):
            # Information coefficient
            alpha_return = alpha_return.flatten()
            ground_truth_return = ground_truth_return.flatten()

            data1_mean = torch.mean(alpha_return)
            data2_mean = torch.mean(ground_truth_return)

            data1_std = torch.std(alpha_return)
            data2_std = torch.std(ground_truth_return)

            covariance = torch.mean((alpha_return - data1_mean) * (ground_truth_return - data2_mean))

            # Add a small constant to the denominator to avoid division by zero
            IC = covariance / ((data1_std * data2_std) + 1e-7)

            # My model will try to maximise the IC (equivalent to minimise -IC)
            return IC
        
        @staticmethod
        def cal_vwap(trade_value_df, trade_size_df):
            # vwap
            return trade_value_df / trade_size_df
                
        @staticmethod
        def cal_rtn(df, back_period=int(1440*24), del_jump = True, jump = 0.2):
            """
            log return: log(close / close.shift(window))
            """
            rtn = np.log( df / df.shift(back_period))
            if del_jump:
                rtn.where(rtn.abs() < jump, np.nan)
            return rtn

        @staticmethod
        def cal_buy_ask_ratio(ask, buy):
            return np.log(ask / buy)
        
        @staticmethod
        def generate_index_pairs(N=hyperparams.get('feature_num')):

            index_pairs = [(i, j) for i in range(1, N) for j in range(i)]

            return index_pairs


    class Features:
        """
        This class contains all the features in the dictionary, and any additional customised features.
        """
        def __init__(self, future_data_dict, mask_future_data):
            self.future_data_dict = future_data_dict
            self.mask_future_data = mask_future_data
            
            
            self.open = future_data_dict['open']
            self.close = future_data_dict['close']
            self.high = future_data_dict['high']
            self.low = future_data_dict['low']
            self.trade_value = future_data_dict['trade_value']
            self.trade_size = future_data_dict['trade_size']
            self.buy_size = future_data_dict['buy_size']
            self.ask_size = future_data_dict['ask_size']
            self.buy_value = future_data_dict['buy_value']
            self.ask_value = future_data_dict['ask_value']
            self.fundinrate = future_data_dict['fundingRate']

            self.vwap = AuxiliaryFunc.cal_vwap(self.trade_value, self.trade_size)
            self.target = AuxiliaryFunc.cal_target(self.trade_value, self.trade_size)

        ################### Basic Features ###################
        def open_f(self):
            return AuxiliaryFunc.norm1_feature(self.open, mask_future_data, 5, True)

        def close_f(self):
            return AuxiliaryFunc.norm1_feature(self.close, mask_future_data, 5, True)

        def high_f(self):
            return AuxiliaryFunc.norm1_feature(self.high, mask_future_data, 5, True)

        def low_f(self):
            return AuxiliaryFunc.norm1_feature(self.low, mask_future_data, 5, True)

        def trade_size_f(self):
            return AuxiliaryFunc.norm1_feature(self.trade_size, mask_future_data, 5, True)
        
        def open_count_f(self):
            return AuxiliaryFunc.norm1_feature(self.open / self.trade_size, mask_future_data, 5, True)

        def vwap_f(self):
            return AuxiliaryFunc.norm1_feature(self.vwap, mask_future_data, 5, True)
        
        def log_ask_buy_value_f(self):
            return AuxiliaryFunc.norm1_feature(AuxiliaryFunc.cal_buy_ask_ratio(self.ask_value, self.buy_value), mask_future_data, 5, True)
        
        def log_ask_buy_size_f(self):
            return AuxiliaryFunc.norm1_feature(AuxiliaryFunc.cal_buy_ask_ratio(self.ask_size, self.buy_size), mask_future_data, 5, True)
        
        def return1_f(self):
            return AuxiliaryFunc.norm1_feature(AuxiliaryFunc.cal_rtn(self.close), mask_future_data, 5, True)
        
        def volume_low_f(self):
            return AuxiliaryFunc.norm1_feature(self.trade_size / self.low, mask_future_data, 5, True)
        
        def vwap_high_f(self):
            return AuxiliaryFunc.norm1_feature(self.vwap / self.high, mask_future_data, 5, True)

        def low_high_f(self):
            return AuxiliaryFunc.norm1_feature(self.low / self.high, mask_future_data, 5, True)
        
        def vwap_close_f(self):
            return AuxiliaryFunc.norm1_feature(self.vwap / self.close, mask_future_data, 5, True)
        
        def target_f(self):
            return AuxiliaryFunc.norm1_feature(self.target, mask_future_data, 5, True)
        

    class DataPreparation:
        """
        This class creates object in the required input shape of the neural model.
        """
        def __init__(self, future_data_dict, mask_future_data):
            self.future_data_dict = future_data_dict
            self.mask_future_data = mask_future_data
            self.all_input_tensor, self.all_target_tensor = self.prepare_data()
        
        def prepare_data(self):
            future_data_features = Features(self.future_data_dict, self.mask_future_data)

            # Input matrix
            feature_array = []
            
            for feature_name in dir(future_data_features):
                if feature_name.endswith("_f") and feature_name != "target_f":
                    feature_method = getattr(future_data_features, feature_name)
                    feature_array.append(feature_method().values)

            # (number of cryptos, 1, number of features, sequence length)
            feature_4d_array = np.stack(feature_array, axis=0).transpose(2,0,1)[:, np.newaxis,:,:]

            # Target matrix
            returns = future_data_features.target_f().values.T # (number of cryptos, time sequence)

            return torch.from_numpy(feature_4d_array), torch.from_numpy(returns)

        def get_train_val_data(self, idx, training_length, validation_length):
            train_start = max(0, idx - training_length - validation_length - 720)
            train_end = max(0, idx - validation_length - 720)
            val_start = max(0, idx - validation_length)
            val_end = idx

            mask_data = self.mask_future_data.copy(deep=True).reset_index(drop=True)
            train_mask_tensor = torch.from_numpy(mask_data.iloc[train_start:train_end, :].values)
            val_mask_tensor = torch.from_numpy(mask_data.iloc[val_start:val_end, :].values)

            train_input_tensor, train_target_tensor = self.split_data_func(train_start, train_end)
            val_input_tensor, val_target_tensor = self.split_data_func(val_start, val_end)

            return train_input_tensor, train_target_tensor, val_input_tensor, val_target_tensor, train_mask_tensor, val_mask_tensor
        
        def split_data_func(self, start, end):
            input_list, target_list = [], []
            num_cryptos = self.all_input_tensor.size(0)

            for crypto in range(num_cryptos):
                input_data = self.all_input_tensor[crypto, :, :, start:end]
                target_data = self.all_target_tensor[crypto, start:end]

                input_list.append(input_data.unsqueeze(0))
                target_list.append(target_data.unsqueeze(0))

            input_tensor = torch.cat(input_list, dim=0)
            target_tensor = torch.cat(target_list, dim=0).squeeze(-1)

            return input_tensor, target_tensor

        def get_use_data(self, idx):
            input_data = self.all_input_tensor[:, :, :, idx - hyperparams.get('lookback_period'):idx]
            mask = torch.tensor(self.mask_future_data.copy(deep=True).reset_index(drop=True).iloc[idx,:])
            target_data = self.all_target_tensor[:, idx]
            
            included_cryptos_idx = mask.nonzero(as_tuple=True)[0]
            
            used_data = input_data[included_cryptos_idx]
            used_data = torch.where(torch.isnan(used_data), torch.tensor(float(0)), used_data)
            
            return used_data, included_cryptos_idx, target_data.shape
        

    class Inception(nn.Module): 
        """
        Represents an inception layer that replaces the conventional convolution layer in CNN.
        
        This layer extracts features using customized feature extracting functions.
        """

        def __init__(self, pair, stride):

            super(Inception, self).__init__()
            self.pair = pair
            self.stride = stride

            norm_feature_length = len(AuxiliaryFunc.generate_index_pairs())
            self.bc1 = nn.BatchNorm1d(norm_feature_length, eps = 1e-5, affine=True)
            self.bc2 = nn.BatchNorm1d(norm_feature_length, eps = 1e-5, affine = True)
            self.bc3 = nn.BatchNorm1d(hyperparams.get('feature_num'), eps = 1e-5, affine = True)
            self.bc4 = nn.BatchNorm1d(hyperparams.get('feature_num'), eps = 1e-5, affine = True)
            self.bc5 = nn.BatchNorm1d(hyperparams.get('feature_num'), eps = 1e-5, affine = True)
            self.bc6 = nn.BatchNorm1d(hyperparams.get('feature_num'), eps = 1e-5, affine = True)

        
        def forward(self, input_tensor):
            """
            Performs forward pass through the inception layer.
            """

            pair = self.pair
            stride = self.stride

            conv1 = self._cov(input_tensor, pair, stride) # (number of crypto, combination_length, len(Index_list) - 1)
            conv2 = self._corr(input_tensor, pair, stride) # (number of crypto, combination_length, len(Index_list) - 1)
            conv3 = self._decaylinear(input_tensor, stride) # (number of crypto, feature_num, len(Index_list) - 1)
            conv4 = self._stddev(input_tensor, stride) # (number of crypto, feature_num, len(Index_list) - 1)
            conv5 = self._zscore(input_tensor, stride) # (number of crypto, feature_num, len(Index_list) - 1)
            conv6 = self._return(input_tensor, stride) # (number of crypto, feature_num, len(Index_list) - 1)

            batch1 = self.bc1(conv1)
            batch2 = self.bc2(conv2)
            batch3 = self.bc3(conv3)
            batch4 = self.bc4(conv4)
            batch5 = self.bc4(conv5)
            batch6 = self.bc6(conv6)
            
            # (number of crypto, 2*comb_length + 4*feature_num, len(Index_list) - 1)
            feature = torch.cat([batch1, batch2, batch3, batch4, batch5, batch6], axis = 1)

            return feature.permute(2,0,1)
        
        ###### Customised feature extracting functions
        def _cov(self, input_tensor, index_pairs, stride):
            """
            input_tensor: 4d tensor, num of crypto * 1 * feature_num * time_seq
            """
            time_seq = input_tensor.shape[3]
            # feature_num = input_tensor.shape[2]
            # combination_length = len(index_pairs)
            if time_seq % stride == 0:
                Index_list = list(range(0, time_seq + stride, stride))
            else:
                mod = time_seq % stride
                Index_list = list(range(0, time_seq + stride - mod, stride)) + [time_seq]

            l = [] 
            for i in range(len(Index_list) - 1):
                start_index, end_index = Index_list[i], Index_list[i + 1]
                data1 = input_tensor[:, :, index_pairs, start_index:end_index] # (number of crypto, 1, combination_length, 2, stride)
                mean1 = data1.mean(dim=4, keepdims = True) # (number of crypto, 1, combination_length, 2, 1)
                spread1 = data1-mean1  # (number of crypto, 1, combination_length, 2, stride)
                cov=spread1[:,:,:,0,:]*spread1[:,:,:,1,:]  # (number of crypto, 1, combination_length, stride)
                cov=cov.mean(dim=3,keepdims = True)  # (number of crypto, 1, combination_length,1)
                l.append(cov)
            l=torch.cat(l,dim=3) # (number of crypto, 1, combination_length, len(Index_list) - 1)
            l=l.squeeze(1)  # (number of crypto, combination_length, len(Index_list) - 1)

            if torch.isnan(l).any():
                logging.warning('NaN in cov')

            return l

        def _corr(self, input_tensor, index_pairs, stride):
            """
            input_tensor: 4d tensor, num of crypto * 1 * feature_num * time_seq
            """
            time_seq = input_tensor.shape[3]

            if time_seq % stride == 0:
                Index_list = list(np.arange(0, time_seq + stride, stride))
            else:
                mod = time_seq % stride
                Index_list = list(np.arange(0, time_seq + stride - mod, stride)) + [time_seq]

            l = []
            for i in range(len(Index_list)-1):
                start_index, end_index = Index_list[i], Index_list[i + 1]
                data = input_tensor[:, :, index_pairs, start_index:end_index] # (number of crypto, 1, combination_length, 2, stride)
                std = data.std(dim=4, keepdim=True) # (number of crypto, 1, combination_length, 2, 1)
                std = std[:,:,:,0,:] * std[:,:,:,1,:] # (number of crypto, 1, combination_length, 1)
                l.append(std)

            l=torch.cat(l,dim=3) # (number of crypto, 1, combination_length, len(Index_list) - 1)
            l=l.squeeze(1)  # (number of crypto, combination_length, len(Index_list) - 1)

            cov  = self._cov(input_tensor, index_pairs, stride) # (number of crypto, combination_length, len(Index_list) - 1)

            l /= (cov + 1e-6)

            if torch.isnan(l).any():
                logging.warning('NaN in corr')

            return l
        
        def _stddev(self, input_tensor, stride):
            """
            input_tensor: 4d tensor, (num of crypto * 1 * feature_num * time_seq)
            """
            time_seq = input_tensor.shape[3]

            if time_seq % stride == 0:
                Index_list = list(np.arange(0, time_seq + stride, stride))
            else:
                mod = time_seq % stride
                Index_list = list(np.arange(0, time_seq + stride - mod, stride)) + [time_seq]

            l = []
            for i in range(len(Index_list)-1):
                start, end = Index_list[i], Index_list[i+1]
                data = input_tensor[:, :, :, start:end] # (num of crypto * 1 * feature_num * stride)
                stddev = data.std(dim=3, keepdim=True) # (num of crypto * 1 * feature_num * 1)
                l.append(stddev)
            
            l = torch.cat(l, dim=3) # (number of crypto, 1, feature_num, len(Index_list) - 1)
            l = l.squeeze(1) # (number of crypto, feature_num, len(Index_list) - 1)

            if torch.isnan(l).any():
                logging.warning('NaN in stddev')

            return l
        
        def _zscore(self, input_tensor, stride):
            time_seq = input_tensor.shape[3]

            if time_seq % stride == 0:
                Index_list = list(np.arange(0, time_seq + stride, stride))
            else:
                mod = time_seq % stride
                Index_list = list(np.arange(0, time_seq + stride - mod, stride)) + [time_seq]

            l = []
            for i in range(len(Index_list)-1):
                start, end = Index_list[i], Index_list[i+1]
                data = input_tensor[:, :, :, start:end] # (num of crypto * 1 * feature_num * stride)
                mean = data.mean(dim=3, keepdims=True) # (num of crypto * 1 * feature_num * 1)
                std = data.std(dim=3, keepdims=True) + 1e-2 # (num of crypto * 1 * feature_num * 1)
                zscore = mean / std
                l.append(zscore)
            
            l = torch.cat(l, dim=3) 
            l = l.squeeze(1) # (number of crypto, feature_num, len(Index_list) - 1)

            if torch.isnan(l).any():
                logging.warning('NaN in zscore')

            return l
        
        def _decaylinear(self, input_tensor, stride):
            time_seq = input_tensor.shape[3]
            if time_seq % stride == 0:
                Index_list = list(np.arange(0, time_seq + stride, stride))
            else:
                mod = time_seq % stride
                Index_list = list(np.arange(0, time_seq + stride - mod, stride)) + [time_seq]

            l = []
            for i in range(len(Index_list)-1):
                start, end = Index_list[i], Index_list[i+1]
                range_ = end-start
                weight = torch.arange(1, range_+1, dtype=input_tensor.dtype).to(input_tensor.device)
                weight /= weight.sum()
                data = input_tensor[:, :, :, start:end] # (num of crypto * 1 * feature_num * stride)
                wd = (data*weight).mean(dim=3, keepdims=True) # (num of crypto * 1 * feature_num * 1)
                l.append(wd)
            
            l = torch.cat(l, dim=3)
            l = l.squeeze(1) # (number of crypto, feature_num, len(Index_list) - 1)

            if torch.isnan(l).any():
                logging.warning('NaN in lineardecay')

            return l
        
        def _return(self, input_tensor, stride):
            time_seq = input_tensor.shape[3]
            if time_seq % stride == 0:
                Index_list = list(np.arange(0, time_seq + stride, stride))
            else:
                mod = time_seq % stride
                Index_list = list(np.arange(0, time_seq + stride - mod, stride)) + [time_seq]

            l = []
            for i in range(len(Index_list)-1):
                start, end = Index_list[i], Index_list[i+1]
                data = input_tensor[:, :, :, start:end] # (num of crypto * 1 * feature_num * stride)
                return_ = (data[:, :, :,-1] / (data[:, :, :, 0] + 1e-4) + 0.1) - 1 # (num of crypto * 1 * feature_num)
                l.append(return_.unsqueeze(-1))
            
            l = torch.cat(l, dim=3)
            l = l.squeeze(1) # (number of crypto, feature_num, len(Index_list) - 1)

            if torch.isnan(l).any():
                logging.warning('NaN in return1')

            return l

    class GRUNet(nn.Module):
        """
        A GRU-based neural network architecture.
        """
        def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob):

            super(GRUNet, self).__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers

            # (number of cryptos, combined feature length, time sequence)
            self.gru = nn.GRU(input_dim, hidden_dim, n_layers, dropout=drop_prob)

            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
            self.fc2 = nn.Linear(hidden_dim // 4, output_dim)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(drop_prob)
            self.bn = nn.BatchNorm1d(output_dim)
        
        def forward(self, X):
            # out: (time sequence, number of cryptos, hidden dim)
            # h: (number of layers, number of cryptos, hidden dim)
            _, h = self.gru(X)

            h = self.relu(self.fc1(h[-1]))
            h = self.dropout(h)
            h = self.fc2(h)
            h_out = self.bn(h) # (number of cryptos, output dim) 

            return h_out
    

    class AlphaNet(nn.Module):
        """
        This class creates the full neural network by combining the Inception object and GRU object.
        """
        def __init__(self, pair_index, inception1_stride, inception2_stride, 
                     gru_input_dim, gru_hidden_dim, gru_output_dim, gru_n_layers, gru_dropout,
                     output_layer_dim):
            super(AlphaNet, self).__init__()
            self.pair_index = pair_index
            self.gru_dropout = gru_dropout

            self.Inception1 = Inception(self.pair_index, stride=inception1_stride)
            self.Inception2 = Inception(self.pair_index, stride=inception2_stride)

            self.gru1 = GRUNet(gru_input_dim, gru_hidden_dim, gru_output_dim, gru_n_layers, gru_dropout)
            self.gru2 = GRUNet(gru_input_dim, gru_hidden_dim, gru_output_dim, gru_n_layers, gru_dropout)

            self.fc = nn.Linear(2 * gru_output_dim, output_layer_dim)
            self.fc.apply(self._truncated_normal)
        
        def forward(self, X):

            X1 = self.Inception1(X)
            X2 = self.Inception2(X)
            

            # (number of cryptos, hidden dim)
            X1 = self.gru1(X1)
            X2 = self.gru2(X2)
            
            # (number of cryptos, hidden dim * 2)
            cat = torch.cat([X1, X2], axis=1)

            # (number of cryptos,)
            X = self.fc(cat).squeeze(1)
            return X
        
        def _truncated_normal(self, module, 
                              mean=hyperparams.get('truncated_normal_mean'), std=hyperparams.get('truncated_normal_std')):
            if isinstance(module, nn.Linear):
                tensor = module.weight
                size = tensor.shape
                tmp = tensor.new_empty(size + (4,)).normal_()
                valid = (tmp < 2) & (tmp > -2)
                ind = valid.max(-1, keepdim=True)[1]
                tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
                tensor.data.mul_(std).add_(mean)


    class EarlyStopping:
        """
        This class creats an object to perform early stopping
        """
        def __init__(self, patience, avg_window):
            """
            patience (int): Number of checks without improvement before early stop is triggered.
            avg_window (int): Number of epochs over which to calculate the moving average IC.
            """
            self.patience = patience
            self.avg_window = avg_window
            self.counter = 0
            self.best_score = None
            self.best_model = None
            self.early_stop = False
            self.avg_ic_history = []
            self.optimal_epoch = None   

        def __call__(self, model, avg_ic):
            self.avg_ic_history.append(avg_ic)

            if len(self.avg_ic_history) < self.avg_window:
                return

            smoothed_score = self.get_smoothed_score()
            
            if self.best_score is None or smoothed_score > self.best_score:
                self.best_score = smoothed_score
                self.best_model = self.save_checkpoint(model)
                self.optimal_epoch = len(self.avg_ic_history)
                self.optimal_epoch -= self.avg_window // 2  
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        def save_checkpoint(self, model):
        
            return copy.deepcopy(model.state_dict())
        
        def get_smoothed_score(self):
            
            return abs(sum(self.avg_ic_history[-self.avg_window:])) / self.avg_window
        
        def load_best_model(self, model):
            
            if self.best_model:
                model.load_state_dict(self.best_model)
            return model

        def should_stop(self):
            
            return self.early_stop


    class DataSet(Dataset):
            """
            This class iterates the training/validation set.
            """
            def __init__(self, input_tensor, target_tensor, mask_tensor, batch_length, bar):
                """
                input tensor: (number of crypto, 1, number of feature, sequence length)
                target tensor: (number of crypto, sequence length)
                mask tensor: (sequence length, number of crypto)
                """
                self.input = input_tensor
                self.target = target_tensor
                self.mask = mask_tensor
                self.batch_length = batch_length

                self.target = torch.where(torch.isnan(self.target) & self.mask.transpose(0,1), torch.tensor(float(0)), self.target)

                self.length = input_tensor.size()[-1] - self.batch_length
                
                self.num_epoch = 0
                self.bar = bar
                self.random = np.random.randint(self.bar, size=999999)

            def __getitem__(self, idx):
                idx = idx * self.bar + self.random[self.num_epoch] + self.batch_length
                
                # Extract Target
                last_mask = self.mask[idx, :]
                last_target = self.target[:, idx][last_mask]

                # Extract input data
                our_input = self.input[:, :, :, idx - self.batch_length:idx]
                out_input = our_input[last_mask]
                out_input = torch.where(torch.isnan(out_input), torch.tensor(float(0)), out_input)

                return out_input, last_target


            def __len__(self):
                # return the equence length
                return self.length // self.bar


    class ModelTrainer:
        """
        This class is to train the neural network.
        """
        def __init__(self, pair_index, inception1_stride, inception2_stride, 
                     gru_input_dim, gru_hidden_dim, gru_output_dim, gru_n_layers, gru_dropout,
                     output_layer_dim,
                     learning_rate, weight_decay,
                     steplr_size, steplr_gamma,
                     early_stopping_avg_window, early_stopping_patience,
                     device):
            
            self.model = AlphaNet(pair_index, inception1_stride, inception2_stride, 
                                  gru_input_dim, gru_hidden_dim, gru_output_dim, gru_n_layers,
                                  gru_dropout, output_layer_dim).to(device)
            
            self.train_criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=steplr_size, gamma=steplr_gamma)
            
            self.valid_criterion = AuxiliaryFunc.IC
            self.early_stopping = EarlyStopping(avg_window=early_stopping_avg_window, patience=early_stopping_patience)
            
            self.device = device
        
        def _train_epoch(self, Training_dataloader, epoch):

            self.model.train()
            Training_dataloader.dataset.num_epoch = epoch
            total_loss, counter = 0, 0

            progress_bar = tqdm(Training_dataloader, desc=f"Epoch {epoch} Train")
            for X, y in progress_bar:
                X, y = X.squeeze(0), y.squeeze(0)

                y_pred = self.model(X.to(self.device).float())
                if torch.isnan(y_pred).any(): logging.warning('NaN value in model output')

                loss = self.train_criterion(y_pred, y.to(self.device).float())
                if torch.isnan(loss).any(): logging.warning('NaN value in loss')

                self.optimizer.zero_grad()
                loss.backward()
                if any(torch.isnan(param.grad).any() for _, param in self.model.named_parameters() if param.grad is not None):
                    logging.warning("NaN value appears in gradient.")

                total_loss += loss.item()
                counter += 1

                self.optimizer.step()
                self.scheduler.step()
                
            logging.critical(f'Average Loss at Epoch{epoch}: {total_loss / counter}')

        def _validate(self, Validation_dataloader, epoch):
            
            self.model.eval()
            with torch.no_grad():
                total_ic, counter = 0, 0
                
                progress_bar_validation = tqdm(Validation_dataloader, desc=f"Epoch {epoch} Validation")
                for X, y in progress_bar_validation:
                    X, y = X.squeeze(0), y.squeeze(0)

                    y_pred = self.model(X.to(self.device).float())

                    ic = self.valid_criterion(y_pred, y.to(self.device).float())
                    
                    total_ic += ic.item()
                    counter += 1

                avg_ic = total_ic / counter
                logging.critical(f'Average IC at Epoch{epoch}: {avg_ic}')

            return avg_ic

        def train_model(self, Training_dataloader, Validation_loader, epochs):
            for epoch in range(1, epochs+1):
                self._train_epoch(Training_dataloader, epoch)
                avg_ic = self._validate(Validation_loader, epoch)

                self.early_stopping(self.model, avg_ic)
                if self.early_stopping.should_stop():
                    logging.critical(f'Early Stopping at Epoch {self.early_stopping.optimal_epoch}')
                    self.model = self.early_stopping.load_best_model(self.model)
                    break
            
            return self.model
        
        def service(self, X):
            
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X.to(self.device).float())
                y_pred = (y_pred - torch.mean(y_pred)) / torch.std(y_pred)
                y_pred = y_pred.cpu()

                return y_pred


    all_data = DataPreparation(future_data_dict, mask_future_data)
    
    # DataFrame to store all the IC predicted
    predicted_df = pd.DataFrame(np.nan, index=mask_future_data.index, columns=mask_future_data.columns)

    # Extract timestamps from dataset
    timestamps = list(future_data_dict.values())[0].index
    start_date = datetime.datetime.fromtimestamp(timestamps[0]/1000).date()
    first_train_idx, search_first_train_index = 999999999, True

    service_progress = None

    for i, time in enumerate(timestamps):
        # turn Unix into datetime
        time_obj = datetime.datetime.fromtimestamp(time/1000)
        current_date = time_obj.date()

        # Retrain the model at 10th/20th/30th of each month
        # First training timestamp
        if search_first_train_index:
            if current_date >= start_date + datetime.timedelta(days=30) and time_obj.day in [10,20,30] and time_obj.hour + time_obj.minute + time_obj.second == 0:
                first_train_idx = i
                search_first_train_index = False

        if i >= first_train_idx:
            if time_obj.day in [10,20,30] and time_obj.hour + time_obj.minute + time_obj.second == 0:

                if service_progress is not None:
                    service_progress.close()

                # Calculate the total number of 10-minute intervals until the next training day
                if time_obj.day == 10:
                    next_training_day = 20
                elif time_obj.day == 20:
                    next_training_day = 30
                else:  # if current day is 30
                    next_training_day = 10  # Next training will be on the 10th of the next month

                # Find the difference in days
                days_to_next_training = next_training_day - time_obj.day if next_training_day > time_obj.day else next_training_day + 30 - time_obj.day
                
                # Create a new progress bar for the service part
                service_progress = tqdm(total=int(1440/10*days_to_next_training), desc="Service progress until next training")

                X_train, Y_train, X_valid, Y_valid, mask_train, mask_valid =\
                    all_data.get_train_val_data(idx=i, training_length=hyperparams.get('training_length'), 
                                                    validation_length=hyperparams.get('validation_length'))

                Training_set = DataSet(X_train, Y_train, mask_train, batch_length=hyperparams.get('lookback_period'), bar=hyperparams.get('training_interval'))
                Training_dataloader = DataLoader(Training_set, batch_size=int(1), shuffle=True)

                Validation_set = DataSet(X_valid, Y_valid, mask_valid, batch_length=hyperparams.get('lookback_period'), bar=hyperparams.get('validation_interval'))
                Validation_dataloader = DataLoader(Validation_set, batch_size=int(1), shuffle=False)

                model_trainer = ModelTrainer(pair_index=AuxiliaryFunc.generate_index_pairs(),
                                             inception1_stride=hyperparams.get('Inception1_stride'), inception2_stride=hyperparams.get('Inception2_stride'),
                                             gru_input_dim = 2 * len(AuxiliaryFunc.generate_index_pairs()) + 4 * hyperparams.get('feature_num'),
                                             gru_hidden_dim=hyperparams.get('GRU_hidden_dim'), gru_output_dim=hyperparams.get('GRU_output_dim'),
                                             gru_n_layers=hyperparams.get('GRU_n_layers'), gru_dropout=hyperparams.get('GRU_dropout'),
                                             output_layer_dim=int(1),
                                             learning_rate=hyperparams.get('learning_rate'), weight_decay=hyperparams.get('weight_decay'),
                                             steplr_size=hyperparams.get('steplr_step_size'), steplr_gamma=hyperparams.get('steplr_gamma'),
                                             early_stopping_avg_window=hyperparams.get('stopping_window'), early_stopping_patience=hyperparams.get('stopping_patience'),
                                             device=device)

                print('Current timestamp:', time_obj.strftime("%Y-%m-%d %H:%M:%S"), 'Model training starts.')

                model = model_trainer.train_model(Training_dataloader, Validation_dataloader, epochs=100)
            
            # Serivce (Predicting)
            if i % hyperparams.get('service_interval') == 0 and service_progress is not None:
                X_use, crypto_idx, original_shape = all_data.get_use_data(idx=i)

                y_pred = model_trainer.service(X_use)

                full_y_pred = torch.full(original_shape, float('nan'))
                full_y_pred[crypto_idx] = y_pred
                predicted_df.iloc[i] = full_y_pred.numpy()

                service_progress.update()
            
    # After the loop, close the last progress bar
    if service_progress is not None:
        service_progress.close()

    feature_dict = {}

    def finnna(data):
        data = data.fillna(method='ffill', axis=0, limit=10)
        data = data.fillna(value=0)
        return data

    feature_dict = {}
    feature_dict[f'factor_demo_l{id}'] = finnna(predicted_df)

    return feature_dict       