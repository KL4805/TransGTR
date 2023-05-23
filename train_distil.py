from util import *
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from model import TSFormer, DistilTSFormer
from engine import distil_trainer
import time
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument('--sdata', type=str, default='../data/METR-LA')
parser.add_argument('--tdata', type=str, default='../data/PEMSD7M')
parser.add_argument("--long_his", type=int, default=2016)
parser.add_argument('--short_his', type=int, default=288*3) # 3 days as a temporary setting
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--output_len', type=int, default=12)
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--embed_dim', type=int, default=96)
parser.add_argument('--patch_size', type=int, default=12)
parser.add_argument("--encoder_depth", type=int, default=4, help='Number of transformer blocks in encoder')
parser.add_argument("--decoder_depth", type=int, default=1, help='Number of transformer blocks in decoder')
parser.add_argument("--num_heads", type=int, default=4, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.15)
parser.add_argument('--num_token', type=int, default=168, help='Number of tokens in a sequence')
parser.add_argument('--in_channel', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=4, help='Width of MLP w.r.t embed_dim')
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--expid', type=str, default='test')
parser.add_argument('--data_number', type=int, default=0, help='Target Data Number. 0 for all data')
parser.add_argument('--teacher_model_path', type=str, default=None, help='Path to the teacher model')
parser.add_argument("--student_model", type=str, help='[TSFormer, DistilFormer]')
parser.add_argument('--lambda_d', type=float, default=1, help='Weight of distilation loss')
args = parser.parse_args()

def main(args):
    train_dataset_s = TimeSeriesForecastingDataset(args.sdata+'/data_in2016_out12.pkl', 
        args.sdata+'/index_in2016_out12.pkl', mode='train')
    train_dataset_t = TimeSeriesForecastingDataset(args.tdata+'/data_in2016_out12.pkl', 
        args.tdata+'/index_in2016_out12.pkl', mode='train', data_number=args.data_number)
    val_dataset_s = TimeSeriesForecastingDataset(args.sdata+'/data_in2016_out12.pkl', 
        args.sdata+'/index_in2016_out12.pkl', mode='valid')
    val_dataset_t = TimeSeriesForecastingDataset(args.tdata+'/data_in2016_out12.pkl', 
        args.tdata+'/index_in2016_out12.pkl', mode='valid')
    test_dataset_s = TimeSeriesForecastingDataset(args.sdata+'/data_in2016_out12.pkl', 
        args.sdata+'/index_in2016_out12.pkl', mode='test')
    test_dataset_t = TimeSeriesForecastingDataset(args.tdata+'/data_in2016_out12.pkl', 
        args.tdata+'/index_in2016_out12.pkl', mode='test')
    print("Source data length: train %d, valid %d, test %d; Num node %d" % (len(train_dataset_s), 
        len(val_dataset_s), len(test_dataset_s), train_dataset_s.data.shape[1]))
    print("Target data length: train %d, valid %d, test %d; Num node %d" % (len(train_dataset_t), 
        len(val_dataset_t), len(test_dataset_t), train_dataset_t.data.shape[1]))
    scaler_pkl_s = load_pkl(args.sdata+'/scaler_in2016_out12.pkl')
    scaler_pkl_t = load_pkl(args.tdata+'/scaler_in2016_out12.pkl')
    mean_s, std_s = scaler_pkl_s['args']['mean'], scaler_pkl_s['args']['std']
    scaler_s = Scaler(mean_s, std_s)
    mean_t, std_t = scaler_pkl_t['args']['mean'], scaler_pkl_t['args']['std']
    scaler_t = Scaler(mean_t, std_t)
    train_loader_s = DataLoader(train_dataset_s, batch_size=args.batch_size, shuffle=True)
    train_loader_t = DataLoader(train_dataset_t, batch_size=args.batch_size, shuffle=True)
    val_loader_s = DataLoader(val_dataset_s, batch_size=args.batch_size)
    val_loader_t = DataLoader(val_dataset_t, batch_size=args.batch_size)
    test_loader_s = DataLoader(test_dataset_s, batch_size=args.batch_size)
    test_loader_t = DataLoader(test_dataset_t, batch_size=args.batch_size)


    
    engine = distil_trainer(args, scaler_s, scaler_t)
    print("Distilling TSFormer from %s to %s with %d target data" % (args.sdata, args.tdata, args.data_number))
    if 'METR-LA' in args.sdata:
        sdata_prefix = 'METR-LA'
    elif 'PEMS-BAY' in args.sdata:
        sdata_prefix = 'PEMS-BAY'
    if 'PEMSD7M' in args.tdata:
        save_distilformer = 'garage_nf/PEMSD7M/' + sdata_prefix+'_'+args.expid + '/'
        if not os.path.exists(save_distilformer):
            os.makedirs(save_distilformer)
    elif 'HKTSM' in args.tdata:
        save_distilformer = 'garage_nf/HKTSM/' + sdata_prefix+'_'+args.expid + '/'
        if not os.path.exists(save_distilformer):
            os.makedirs(save_distilformer)

    val_distil_t = []
    val_mae = []
    val_rmse = []
    val_mape = []
    train_distil_s = []
    train_distil_t = []
    train_mae = []

    train_time = []
    val_time = []
    for ep in range(1, args.num_epoch+1):
        epoch_distil_s = []
        epoch_distil_t = []
        epoch_mae = []
        epoch_valdistil_t = []
        epoch_valmae = []
        epoch_valrmse = []
        epoch_valmape = []
        s1 = time.time()
        for i, (_, xt) in enumerate(train_loader_t):
            # print('xt', xt.shape)
            batch_size = xt.shape[0]
            xs = sample_batch(train_dataset_s, batch_size)
            xt = xt.to(args.device)
            xs = xs.to(args.device)
            metrics = engine.train_distil(xs, xt)
            epoch_distil_s.append(metrics[0])
            epoch_distil_t.append(metrics[1])
            epoch_mae.append(metrics[2])
            if i % args.print_every == 0:
                print("Distil epoch %d, Iter %d, train distil loss source %.4f, target %.4f, mae %.4f, time spent %.4fs" %\
                    (ep, i, epoch_distil_s[-1], epoch_distil_t[-1], epoch_mae[-1], time.time() - s1))
            # print('x', x.shape) (B, len, node, feat)
        engine.scheduler.step()
        s2 = time.time()
        print('Distil epoch %d, train distil loss source %.4f, target %.4f, mae %.4f, time %.4fs' % \
            (ep, np.mean(epoch_distil_s), np.mean(epoch_distil_t), np.mean(epoch_mae), s2-s1))
        train_time.append(s2-s1)
        train_distil_s.append(np.mean(epoch_distil_s))
        train_distil_t.append(np.mean(epoch_distil_t))
        train_mae.append(np.mean(epoch_mae))
        t1 = time.time()
        for y, x in val_loader_t:
            x = x.to(args.device)
            metrics = engine.eval_distil(x)
            epoch_valdistil_t.append(metrics[0])
            epoch_valmae.append(metrics[1])
            epoch_valrmse.append(metrics[2])
            epoch_valmape.append(metrics[3])
            # print(metrics[3])
        t2 = time.time()

        val_time.append(t2-t1)
        val_mae.append(np.mean(epoch_valmae))
        val_rmse.append(np.mean(epoch_valrmse))
        val_mape.append(np.mean(epoch_valmape))
        val_distil_t.append(np.mean(epoch_valdistil_t))
        print("Distil epoch %d, val distil loss %.4f, mae %.4f, rmse %.4f, mape %.4f, time %.4fs" % \
            (ep, val_distil_t[-1], val_mae[-1], val_rmse[-1], val_mape[-1], t2-t1))
        torch.save(engine.student_model.state_dict(), save_distilformer+'epoch_%d_%.4f.pth' % (ep, val_mae[-1]))

    print("Distillation training finish")
    print("Average epoch training time is %.4fs, validation time is %.4fs" % (np.mean(train_time), np.mean(val_time)))
    np.save(save_distilformer+"val_mae.npy", arr = np.array(val_mae))
    np.save(save_distilformer+'val_rmse.npy', arr=np.array(val_rmse))
    np.save(save_distilformer+'val_mape.npy', arr=np.array(val_mape))
    np.save(save_distilformer+'val_distil_t.npy', arr=np.array(val_distil_t))
    np.save(save_distilformer+'train_distil_t.npy', arr = np.array(train_distil_t))
    np.save(save_distilformer+'train_loss.npy', arr = np.array(train_mae))


def sample_batch(dataset, batch_size):
    # sample a batch of data from a dataset
    data_size = len(dataset)
    data_idx = np.random.randint(0, data_size, batch_size)
    sampled_x = []
    for idx in data_idx:
        _, x = dataset[idx]
        sampled_x.append(x)

    return torch.stack(sampled_x, dim=0)



if __name__ == '__main__':
    main(args)
