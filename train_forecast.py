import numpy as np
import torch
import argparse
from util import *
import torch.nn as nn
from torch.utils.data import DataLoader
from engine import forecast_trainer
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--sdata', type=str, default='METR-LA')
parser.add_argument('--tdata', type=str, default='PEMSD7M')
parser.add_argument('--long_his', type=int, default=288*3, help='The length of long-term history fed to TSFormer')
parser.add_argument('--short_his', type=int, default=12, help='The length of input sequence to GWNet')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--output_len', type=int, default=12, help='Forecasting horizon')
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--source_epoch', type=int, default=50)
parser.add_argument('--target_epoch', type=int, default=50)
parser.add_argument('--data_number', type=int, default=0, help='Target data number')
parser.add_argument('--adaptadj', action='store_true', help='whether to use additional adaptive adjacency in gwnet')
parser.add_argument("--degree_reg", type=float, default=0.1, help='Regularization on the graph degree.')
parser.add_argument("--coral_reg", type=float, default=0.001, help='Regularization on CORAL loss between st features')

# pre-trained tsformer model param
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--embed_dim', type=int, default=96)
parser.add_argument('--patch_size', type=int, default=12)
parser.add_argument("--encoder_depth", type=int, default=3, help='Number of transformer blocks in encoder')
parser.add_argument("--decoder_depth", type=int, default=1, help='Number of transformer blocks in decoder')
parser.add_argument("--num_heads", type=int, default=4, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--num_token', type=int, default=72, help='Number of tokens in a sequence')
parser.add_argument('--in_channel', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=4, help='Width of MLP w.r.t embed_dim')
parser.add_argument('--tsformer_path', type=str)
parser.add_argument('--model', type=str, default='TSFormer', help='[TSFormer, DistilFormer]')

# experiment logging params
parser.add_argument('--expid', type=str, default='test')
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--eval_every', type=int, default=5)
parser.add_argument('--source_model_path', type=str, default=None, help='If specified, load this model instead of source training.')
args = parser.parse_args()

def main(args):
    sdata = '../data/' + args.sdata
    tdata = '../data/' + args.tdata
    train_dataset_s = ForecastingDataset(sdata+'/data_in12_out12.pkl', 
        sdata+'/index_in12_out12.pkl', mode='train', seq_len=args.long_his)
    train_dataset_t = ForecastingDataset(tdata+'/data_in12_out12.pkl', 
        tdata+'/index_in12_out12.pkl', mode='train', data_number=args.data_number, seq_len=args.long_his)
    val_dataset_s = ForecastingDataset(sdata+'/data_in12_out12.pkl', 
        sdata+'/index_in12_out12.pkl', mode='valid', seq_len=args.long_his)
    val_dataset_t = ForecastingDataset(tdata+'/data_in12_out12.pkl', 
        tdata+'/index_in12_out12.pkl', mode='valid', seq_len=args.long_his)
    test_dataset_s = ForecastingDataset(sdata+'/data_in12_out12.pkl', 
        sdata+'/index_in12_out12.pkl', mode='test', seq_len=args.long_his)
    test_dataset_t = ForecastingDataset(tdata+'/data_in12_out12.pkl', 
        tdata+'/index_in12_out12.pkl', mode='test', seq_len=args.long_his)
    print("Source data length: train %d, valid %d, test %d; num nodes %d" % (len(train_dataset_s), 
        len(val_dataset_s), len(test_dataset_s), train_dataset_s.data.shape[1]))
    print("Target data length: train %d, valid %d, test %d, num nodes %d" % (len(train_dataset_t), 
        len(val_dataset_t), len(test_dataset_t), train_dataset_t.data.shape[1]))
    scaler_pkl_s = load_pkl(sdata+'/scaler_in12_out12.pkl')
    scaler_pkl_t = load_pkl(tdata+'/scaler_in12_out12.pkl')
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


    # ground truth graph
    # adj_s = load_adjacency(args.sdata)
    # adj_s = [torch.Tensor(i).to(args.device) for i in adj_s]
    # adj_t = load_adjacency(args.tdata)
    # adj_t = [torch.Tensor(i).to(args.device) for i in adj_t]

    adj_s = []
    adj_t = []

    engine = forecast_trainer(args, scaler_s, scaler_t, adj_s, adj_t)
    print("Training forecast model, transfer from %s to %s, with %d day target data" % (args.sdata, args.tdata, args.data_number))
    save_forecast = 'garage_forecast/%s/%s_%s/' % (args.tdata, args.sdata, args.expid)
    if not os.path.exists(save_forecast):
        os.makedirs(save_forecast)

    if args.source_model_path is not None:
        # load the source model and directly start fine-tuning
        gwnet_sd = torch.load(args.source_model_path, map_location=args.device)
        engine.gwnet_model.load_state_dict(gwnet_sd)
        print("Load source gwnet model from %s success" % args.source_model_path)
        try:
            dgl_path = args.source_model_path.replace("gwnet", 'dgl')
            dgl_sd = torch.load(dgl_path, map_location = args.device)
            engine.dglv2.load_state_dict(dgl_sd)
            print("Load source dgl model from %s success" % dgl_path)
        except:
            pass
        try:
            adp_path = args.source_model_path.replace("gwnet", 'adp_t')
            adp_sd = torch.load(adp_path, map_location = args.device)
            engine.adp_t.load_state_dict(adp_sd)
            print("Load source adp model from %s success" % adp_path)
        except:
            pass
        # do evaluation
        evaluate_12horizon(args, engine, test_loader_t,phase='t')

        


    elif args.source_epoch > 0:
        # start source training from scratch. 
        train_loss_s = []
        val_loss_s = []
        val_rmse_s = []
        val_mape_s = []
        train_time_s = []
        val_time_s = []
        for ep in range(1, args.source_epoch + 1):
            epoch_trainloss_s = []
            epoch_valloss_s = []
            epoch_valrmse_s = []
            epoch_valmape_s = []
            epoch_deg_s = []
            epoch_coral_s = []
            epoch_recons_s = []
            s1 = time.time()
            for i, (ys, xs, long_xs) in enumerate(train_loader_s):
                # print('xs', xs.shape) # (batch, seq_len, num_node, channel)
                # print('ys', ys.shape) 
                # print("long_xs", long_xs.shape)

                # sample a batch of target data
                xt, yt = sample_batch(train_dataset_t, xs.shape[0])

                metrics = engine.source_train(xs, ys, long_xs, xt, yt)
                if i % 50 == 0:
                    print("Epoch %d, Iter %d, train loss %.4f, Degree loss %.4f, CORAL loss %.4f, Recons loss %.4f, Time spent %.4fs" % (ep, i, metrics[0], metrics[3], metrics[4], metrics[5], time.time() - s1))
                epoch_trainloss_s.append(metrics[0])
                epoch_deg_s.append(metrics[3])
                epoch_coral_s.append(metrics[4])
                epoch_recons_s.append(metrics[5])

            s2 = time.time()
            print('Epoch %d, train loss %.4f, Degree loss %.4f, CORAL loss %.4f, Recons loss %.4f, Time spent %.4fs' % (ep, np.mean(epoch_trainloss_s),np.mean(epoch_deg_s), np.mean(epoch_coral_s), np.mean(epoch_recons_s), s2-s1))
            train_loss_s.append(np.mean(epoch_trainloss_s))
            train_time_s.append(s2-s1)
            t1 = time.time()
            for i, (ys, xs, long_xs) in enumerate(val_loader_s):
                metrics=engine.source_eval(xs,ys,long_xs)
                epoch_valloss_s.append(metrics[0])
                epoch_valrmse_s.append(metrics[1])
                epoch_valmape_s.append(metrics[2])
            t2 = time.time()
            val_time_s.append(t2-t1)
            val_loss_s.append(np.mean(epoch_valloss_s))
            val_rmse_s.append(np.mean(epoch_valrmse_s))
            val_mape_s.append(np.mean(epoch_valmape_s))
            print("Epoch %d, val loss %.4f, val rmse %.4f, val mape %.4f, val time %.4fs" % (ep, val_loss_s[-1], val_rmse_s[-1], val_mape_s[-1], t2-t1))
            if ep % args.eval_every == 0:
                evaluate_12horizon(args, engine, test_loader_s)
            torch.save(engine.gwnet_model.state_dict(), save_forecast+'source_epoch_%d_%.4f_gwnet.pth' % (ep,val_loss_s[-1]))
            try:
                torch.save(engine.dglv2.state_dict(), save_forecast+'source_epoch_%d_%.4f_dgl.pth' % (ep,val_loss_s[-1]))
            except:
                pass
            try:
                torch.save(engine.reconstructors.state_dict(), save_forecast+'source_epoch_%d_%.4f_recons.pth' % (ep,val_loss_s[-1]))
            except:
                pass
    print("Source training finish. Begin Fine-tuning")

    if args.target_epoch > 0:
        if engine.adj_t is not None:
            engine.gwnet_model.supports = engine.adj_t
        train_loss = []
        val_loss = []
        val_rmse = []
        val_mape = []
        train_time = []
        val_time = []
        for ep in range(1, args.target_epoch+1):
            epoch_trainloss = []
            epoch_valloss = []
            epoch_valrmse = []
            epoch_valmape = []
            epoch_deg = []
            s1 = time.time()

            for i, (yt, xt, long_xt) in enumerate(train_loader_t):

                metrics = engine.fine_tune(xt, yt, long_xt)
                if metrics[0] == 0 and metrics[1] == 0 and metrics[2] == 0:
                    # no long term
                    continue
                if i % 50 == 0:
                    print("Finetune epoch %d, iter %d, train loss %.4f, degree loss %.4f, Time spent %.4fs" % (ep, i, metrics[0], metrics[3], time.time() - s1))
                epoch_trainloss.append(metrics[0])
                epoch_deg.append(metrics[3])
            s2 = time.time() 
            print("Finetune epoch %d, train loss %.4f, degree loss %.4f, train time %.4fs" % (ep, np.mean(epoch_trainloss), np.mean(epoch_deg), s2-s1))
            train_loss.append(np.mean(epoch_trainloss))
            train_time.append(s2-s1)
            t1 = time.time()
            for i, (yt, xt, long_xt) in enumerate(val_loader_t):
        
                metrics = engine.target_eval(xt, yt, long_xt)
                epoch_valloss.append(metrics[0])
                epoch_valrmse.append(metrics[1])
                epoch_valmape.append(metrics[2])
            t2 = time.time()
            val_time.append(t2-t1)
            val_loss.append(np.mean(epoch_valloss))
            val_rmse.append(np.mean(epoch_valrmse))
            val_mape.append(np.mean(epoch_valmape))
            print("Finetune epoch %d, val loss %.4f, val rmse %.4f, val mape %.4f, val time %.4fs" % (ep, val_loss[-1], val_rmse[-1], val_mape[-1], t2-t1))
            torch.save(engine.gwnet_model.state_dict(), save_forecast+'target_epoch_%d_%.4f_gwnet.pth' % (ep,val_loss[-1]))
            try:
                torch.save(engine.dgl_t.state_dict(), save_forecast+'target_epoch_%d_%.4f_dgl.pth' % (ep, val_loss[-1]))
            except:
                torch.save(engine.dglv2.state_dict(), save_forecast+'target_epoch_%d_%.4f_dgl.pth' % (ep, val_loss[-1]))
            if ep % args.eval_every==0:
                evaluate_12horizon(args, engine, test_loader_t,phase='t')
            if args.adaptadj:
                torch.save(engine.adp_t.state_dict(), save_forecast+'target_epoch_%d_%.4f_adp_t.pth' % (ep,val_loss[-1]))
        print("Fine-tuning finish. Begin final evaluation.")
        bestid = np.argmin(val_loss)
        print("Best epoch is %d, val_loss is %.4f" % (bestid+1, val_loss[bestid]))
        best_model = torch.load(save_forecast+'target_epoch_%d_%.4f_gwnet.pth'%(bestid+1, val_loss[bestid]))
        engine.gwnet_model.load_state_dict(best_model)
        best_dgl = torch.load(save_forecast+'target_epoch_%d_%.4f_dgl.pth' % (bestid+1, val_loss[bestid]))
        try:
            engine.dgl_t.load_state_dict(best_dgl)
        except:
            engine.dglv2.load_state_dict(best_dgl)
        if args.adaptadj:
            best_adp = torch.load(save_forecast+'target_epoch_%d_%.4f_adp_t.pth'%(bestid+1, val_loss[bestid]))
            engine.adp_t.load_state_dict(best_adp)
        mae, rmse, mape = evaluate_12horizon(args, engine, test_loader_t, phase='t')
        np.save(save_forecast+'/val_loss.npy', arr=np.array(val_loss))
        np.save(save_forecast+'/train_loss.npy', arr = np.array(train_loss))
        np.save(save_forecast+'/final_mae.npy', arr = np.array(mae))
        np.save(save_forecast+'/final_rmse.npy', arr=np.array(rmse))
        np.save(save_forecast+'/final_mape.npy', arr=np.array(mape))
            


def evaluate_12horizon(args, engine_, loader_, phase='s'):
    print("Evaluating over 12 horizons...")
    ytrue = []
    ypred = []
    for y, x, long_x in loader_:
        if phase == 's':
            out, ytrue_ = engine_.source_eval(x, y, long_x, return_val=True)
        elif phase == 't':
            out, ytrue_ = engine_.target_eval(x, y, long_x, return_val=True)
        ytrue.append(ytrue_)
        ypred.append(out)
    ytrue = torch.cat(ytrue, dim=0)
    ypred = torch.cat(ypred, dim=0)
    maes = []
    rmses = []
    mapes = []
    for i in range(12):
        mae = masked_mae(ypred[:,:,i],ytrue[:,:,i],0.0)
        rmse = masked_rmse(ypred[:,:,i], ytrue[:,:,i], 0.0)
        mape = masked_mape(ypred[:,:,i], ytrue[:,:,i], 0.0)
        maes.append(mae.item())
        rmses.append(rmse.item())
        mapes.append(mape.item())
        print("Horizon %d, Test MAE %.4f, RMSE %.4f, MAPE %.4f" % (i+1, mae, rmse, mape))
    print("On average over 12 horizons, Test MAE %.4f, RMSE %.4f, MAPE %.4f" % (np.mean(maes), np.mean(rmses), np.mean(mapes)))
    return maes, rmses, mapes
    

def load_adjacency(datapath):
    if 'METR-LA' in datapath:
        adj, _ = load_adj('../data/sensor_graph/adj_mx.pkl', 'doubletransition')
    if 'PEMS-BAY' in datapath:
        adj, _ = load_adj('../data/sensor_graph/adj_mx_bay.pkl', 'doubletransition')
    if 'PEMSD7M' in datapath:
        adj, _ = load_adj_csv('../data/PEMSD7M/W_228.csv', 'doubletransition')
    if 'HKTSM' in datapath:
        adj, _ = load_adj_npy("../data/HKTSM/hk_dist.npy", 'doubletransition')
    return adj

def sample_batch(dataset, batch_size):
    # sample a batch of data from a dataset
    data_size = len(dataset)
    data_idx = np.random.randint(0, data_size, batch_size)
    sampled_x = []
    sampled_y = []
    for idx in data_idx:
        y, x, _ = dataset[idx]
        sampled_x.append(x)
        sampled_y.append(y)

    return torch.stack(sampled_x, dim=0), torch.stack(sampled_y, dim=0)


if __name__ == '__main__':
    main(args)
