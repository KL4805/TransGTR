from model import *
import torch
import torch.nn as nn
import torch.optim as optim 
import util
import random

class tsformer_trainer:
    def __init__(self, args, scaler):
        self.args = args
        if args.model == 'TSFormer':
            self.tsformer_model = TSFormer(args.patch_size, args.in_channel, args.embed_dim, args.num_heads, args.mlp_ratio, args.dropout, 
                args.num_token, args.mask_ratio, args.encoder_depth, args.decoder_depth, mode='pre-train')
        elif args.model == 'DistilFormer':
            self.tsformer_model = DistilTSFormer(args.patch_size, args.in_channel, args.embed_dim, args.num_heads, args.mlp_ratio, args.dropout, 
                args.num_token, args.mask_ratio, args.encoder_depth, args.decoder_depth, mode='pre-train')
        self.device = torch.device(args.device)
        self.tsformer_model = self.tsformer_model.to(self.device)
        self.tsformer_opt = optim.Adam(self.tsformer_model.parameters(), betas=(0.9, 0.95), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.tsformer_opt, milestones=[50], gamma = 0.5)
        self.scaler = scaler 
        self.clip = 5
    
    def pre_train(self, history_seq):
        self.tsformer_model.train()
        self.tsformer_opt.zero_grad()
        history_seq = history_seq.to(self.device)
        if self.args.model == 'TSFormer':
            reconstructed, label = self.tsformer_model(history_seq[:,:,:,:self.args.in_channel])
            ## print('reconstructed', reconstructed.shape)
            ## print('label', label.shape)
        elif self.args.model == 'DistilFormer':
            reconstructed, label = self.tsformer_model(history_seq, mask=True)
            ## print('reconstructed', reconstructed.shape)
            ## print('label', label.shape)
        reconstructed = self.scaler.inverse_transform(reconstructed)
        label = self.scaler.inverse_transform(label)
        mae_loss = util.masked_mae(reconstructed, label, 0.0)
        mae_loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(self.tsformer_model.parameters(), self.clip)
        self.tsformer_opt.step()
        rmse_loss = util.masked_rmse(reconstructed, label, 0.0)
        mape_loss = util.masked_mape(reconstructed, label, 0.0)
        return mae_loss.item(), rmse_loss.item(), mape_loss.item()
        
    
    def eval_pretrain(self, history_seq):
        self.tsformer_model.eval()
        with torch.no_grad():
            history_seq = history_seq.to(self.device)
            if self.args.model == 'TSFormer':
                reconstructed, label = self.tsformer_model(history_seq[:,:,:,:self.args.in_channel])
            elif self.args.model == 'DistilFormer':
                reconstructed, label = self.tsformer_model(history_seq, mask=True)
            reconstructed = self.scaler.inverse_transform(reconstructed)
            label = self.scaler.inverse_transform(label)
            mae_loss = util.masked_mae(reconstructed, label, 0.0)
            rmse_loss = util.masked_rmse(reconstructed, label, 0.0)
            mape_loss = util.masked_mape(reconstructed, label, 0.0)
        return mae_loss.item(), rmse_loss.item(), mape_loss.item()


class distil_trainer:
    def __init__(self, args, scaler_s, scaler_t):
        self.args = args
        self.device = torch.device(args.device)
        self.teacher_model = TSFormer(patch_size=12,in_channel=1, embed_dim=96, num_heads=4, 
            mlp_ratio=4, dropout=0.15, num_token=168, mask_ratio=0.75,encoder_depth=4, 
            decoder_depth=1, mode='forecasting').to(self.device)
        try:
            self.teacher_model.load_state_dict(torch.load(args.teacher_model_path, map_location=args.device)['model_state_dict'])
        except:
            self.teacher_model.load_state_dict(torch.load(args.teacher_model_path, map_location=args.device))
        print("Loaded teacher model from %s" % args.teacher_model_path)
        # we all use num_token 168 to adapt to long sequences (e.g. 7 day target data)
        if args.student_model == 'TSFormer':
            self.student_model = TSFormer(patch_size=args.patch_size, in_channel=args.in_channel, embed_dim=args.embed_dim,
                num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, dropout=args.dropout, num_token=168, 
                mask_ratio=args.mask_ratio, encoder_depth=args.encoder_depth, decoder_depth=1, mode='forecasting').to(self.device)
        elif args.student_model == 'DistilFormer':
            self.student_model = DistilTSFormer(patch_size=args.patch_size, in_channel=args.in_channel, embed_dim=args.embed_dim,
                num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, dropout=args.dropout, num_token=168, 
                mask_ratio=args.mask_ratio, encoder_depth=args.encoder_depth, decoder_depth=1, mode='forecasting').to(self.device)
        print("student num token", int(args.short_his/args.patch_size))
        
        # self.student_model = DistilTSFormer(patch_size=12,in_channel=1,embed_dim=96,num_heads=4,mlp_ratio=args.mlp_ratio, 
        #     dropout=args.dropout, num_token=args.num_token, mask_ratio=args.mask_ratio, encoder_depth=args.encoder_depth,
        #     decoder_depth=args.decoder_depth, mode='pre-train').to(self.device)
        self.scaler_s = scaler_s
        self.scaler_t = scaler_t
        self.clip = 5
        self.student_optim = optim.Adam(self.student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.student_optim, milestones=[50], gamma = 0.5)


    def train_distil(self, source_x, target_x):
        # this function does two things: 
        # first, distillation on both source and target data
        self.teacher_model.eval()
        self.student_model.train()
        self.student_model.mode='forecasting'
        with torch.no_grad():
            source_teacher = self.teacher_model(source_x[:,:,:,:self.args.in_channel])
            target_teacher = self.teacher_model(target_x[:,:,:,:self.args.in_channel])
        short_history_len = self.args.short_his
        max_position = int((2016-self.args.short_his)//self.args.patch_size)
        ## print('max_position', max_position)
        pos_sample = random.randint(0, max_position)
        ## print("sampled_position", pos_sample)
        start_pos = pos_sample * self.args.patch_size
        end_pos = start_pos + self.args.short_his
        ## print("start_pos, end_pos", start_pos, end_pos)
        source_x_short = source_x[:,start_pos:end_pos,:,:]
        target_x_short = target_x[:,start_pos:end_pos,:,:]

        ## print("source_x_short", source_x_short.shape)
        ## print("target_x_short", target_x_short.shape)
        if isinstance(self.student_model, DistilTSFormer):
            # print('distilformer')
            source_student = self.student_model(source_x_short, mask=False,index=(pos_sample, int(end_pos/self.args.patch_size)))
            target_student = self.student_model(target_x_short, mask=False,index=(pos_sample, int(end_pos/self.args.patch_size)))
        elif isinstance(self.student_model, TSFormer):
            source_student = self.student_model(source_x_short[:,:,:,:self.args.in_channel])
            target_student = self.student_model(target_x_short[:,:,:,:self.args.in_channel])
        source_teacher = source_teacher[:,:,pos_sample:int(end_pos/self.args.patch_size),:]
        target_teacher = target_teacher[:,:,pos_sample:int(end_pos/self.args.patch_size),:]
        # print('source_teacher', source_teacher.shape)
        # print('target_teacher', target_teacher.shape)
        # print('source_student', source_student.shape)
        # print('target_student', target_student.shape)
        distil_loss_s = (source_student-source_teacher).pow(2).mean() 
        distil_loss_t = (target_student-target_teacher).pow(2).mean()

        # second, masked autoencoding training on target data
        self.student_model.mode='pre-train'
        if isinstance(self.student_model, DistilTSFormer):
            ##print('distilformer')
            reconstructed, label = self.student_model(target_x_short, mask=True)
        elif isinstance(self.student_model, TSFormer):
            reconstructed, label = self.student_model(target_x_short[:,:,:,:self.args.in_channel])
        reconstructed = self.scaler_t.inverse_transform(reconstructed)
        label = self.scaler_t.inverse_transform(label)
        mae_loss = util.masked_mae(reconstructed, label, 0.0)

        ## print('reconstructed', reconstructed.shape)
        ## print('label', label.shape)

        loss = self.args.lambda_d * (distil_loss_s + distil_loss_t) + mae_loss
        self.student_optim.zero_grad()
        loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(self.student_model.parameters(), self.clip)
        self.student_optim.step()
        return distil_loss_s.item(), distil_loss_t.item(), mae_loss.item()
    
    def eval_distil(self, target_x):
        self.teacher_model.eval()
        self.student_model.eval()
        self.student_model.mode='forecasting'
        with torch.no_grad():
            target_teacher = self.teacher_model(target_x[:,:,:,:self.args.in_channel])
            short_history_len = self.args.short_his
            max_position = int((2016-self.args.short_his)//self.args.patch_size)
            pos_sample = random.randint(0, max_position)
            start_pos = pos_sample * self.args.patch_size
            end_pos = start_pos + self.args.short_his
            target_x_short = target_x[:,start_pos:end_pos,:,:]
            # print("source_x_short", source_x_short.shape)
            # print("target_x_short", target_x_short.shape)
            if isinstance(self.student_model, TSFormer):
                target_student = self.student_model(target_x_short[:,:,:,:self.args.in_channel])
            elif isinstance(self.student_model, DistilTSFormer):
                target_student = self.student_model(target_x_short, mask=False,index=(pos_sample, int(end_pos/self.args.patch_size)))
            target_teacher = target_teacher[:,:,pos_sample:int(end_pos/self.args.patch_size),:]

            distil_loss_t = (target_student-target_teacher).pow(2).mean().item()

            # second, masked autoencoding training on target data
            self.student_model.mode='pre-train'
            if isinstance(self.student_model, DistilTSFormer):
                ##print('distilformer')
                reconstructed, label = self.student_model(target_x_short, mask=True)
            elif isinstance(self.student_model, TSFormer):
                reconstructed, label = self.student_model(target_x_short[:,:,:,:self.args.in_channel])

            reconstructed = self.scaler_t.inverse_transform(reconstructed)
            label = self.scaler_t.inverse_transform(label)
            # print('reconstructed mean', reconstructed.mean().item())
            # print('label mean', label.mean().item())
            # print("label min", label.min().item())      
            mae_loss = util.masked_mae(reconstructed, label, 0.0).item()
            rmse_loss = util.masked_rmse(reconstructed, label, 0.0).item()
            mape_loss = util.masked_mape(reconstructed, label, 0.0).item()
        return distil_loss_t, mae_loss, rmse_loss, mape_loss
        

class forecast_trainer:
    def __init__(self, args, scaler_s, scaler_t, adj_s = None, adj_t = None):
        self.args = args
        self.device = torch.device(args.device)
        self.scaler_s = scaler_s
        self.scaler_t = scaler_t
        num_node = {"METR-LA":207, "PEMS-BAY":325, "PEMSD7M":228, 'HKTSM':608}
        for k in num_node:
            if k in args.sdata:
                self.source_name = k
                num_nodes_s = num_node[k]
            if k in args.tdata:
                self.target_name = k
                num_nodes_t = num_node[k]
        # keep everything else the same
        self.clip=3
        self.degree_reg = args.degree_reg
        self.coral_reg = args.coral_reg
        self.coral_loss_unit = CORAL_loss()
        self.adj_s = adj_s
        self.adj_t = adj_t
        self.gwnet_model = GraphWaveNet(adj_s, args.dropout, args.adaptadj).to(self.device)
        
        # discrete graph learning v2, tsformer fixed + link predictor trainable

        # fix the input seq len here depending on the target data
        if args.data_number == 7:
            long_his=2016
        elif args.data_number == 3:
            long_his=864
        else:
            long_his=args.long_his
        self.dglv2 = DiscreteGraphLearningV2(args.device, self.source_name, self.target_name, 10, long_his, args.short_his, args.data_number, args.model).to(self.device)
        if args.adaptadj:
            self.adp_s = AdaptiveAdjacency(num_nodes_s, 10).to(self.device)
            self.adp_t = AdaptiveAdjacency(num_nodes_t, 10).to(self.device)
            # dglv2
            self.optimizer_s = optim.Adam(list(self.gwnet_model.parameters())+list(self.adp_s.parameters())+list(self.dglv2.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
            self.optimizer_t = optim.Adam(list(self.gwnet_model.parameters())+list(self.adp_t.parameters())+list(self.dglv2.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            # dglv2
            self.optimizer_s = optim.Adam(list(self.gwnet_model.parameters())+list(self.dglv2.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
            self.optimizer_t = optim.Adam(list(self.gwnet_model.parameters())+list(self.dglv2.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.tsf_type = args.model
        if self.tsf_type == 'TSFormer':
            self.tsformer = TSFormer(args.patch_size, args.in_channel, args.embed_dim, args.num_heads, args.mlp_ratio, args.dropout, 
                    args.num_token, args.mask_ratio, args.encoder_depth, args.decoder_depth, mode='forecasting').to(self.device)
        elif self.tsf_type == 'DistilFormer':
            # print(long_his/12)
            self.tsformer = DistilTSFormer(args.patch_size, args.in_channel, args.embed_dim, args.num_heads, args.mlp_ratio, args.dropout, 
                    int(long_his/12), args.mask_ratio, args.encoder_depth, args.decoder_depth, mode='forecasting').to(self.device)
        sd = torch.load(args.tsformer_path, map_location=self.device)
        self.tsformer.load_state_dict(sd)
        for param in self.tsformer.parameters():
            param.requires_grad=False
        # print("tsformer in train?", self.tsformer.training) True
        self.tsformer.eval()
        print("Loading state dict from %s success" % args.tsformer_path)

        # reconstructors
        self.reconstructors = nn.ModuleList()
        for i in range(7):
            reconstructor = FCReconstructor(32, 32, 32).to(self.device)
            self.reconstructors.append(reconstructor)
        self.recons_opt = torch.optim.Adam(self.reconstructors.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    def source_train(self, xs, ys, long_xs, xt=None, yt=None):
        # dglv2
        self.dglv2.train()
        self.gwnet_model.train()
        self.optimizer_s.zero_grad()
        xs = xs.to(self.device)
        ys = ys.to(self.device)
        long_xs = long_xs.to(self.device)

        # dglv2
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.dglv2(long_xs, tsformer=self.tsformer, compute_hidden=False, domain='s')
        if self.args.adaptadj:
            adp = self.adp_s()
            out, tonly_s, st_s = self.gwnet_model(xs[:,:,:,:2], adp=adp, sampled_adj=sampled_adj,return_st=True)
        else:
            out, tonly_s, st_s = self.gwnet_model(xs[:,:,:,:2], adp=None, sampled_adj=sampled_adj, return_st=True)
        
        
        if xt is not None and yt is not None:
            xt = xt.to(self.device)
            yt = yt.to(self.device)
            _, _, _, sampled_adj_t = self.dglv2(xt, tsformer = self.tsformer, compute_hidden=False, domain='t')
            if self.args.adaptadj:
                adp_t = self.adp_t()
                _, tonly_t, st_t = self.gwnet_model(xt[:,:,:,:2], adp=adp_t, sampled_adj=sampled_adj_t,return_st=True)
            else:
                _, tonly_t, st_t = self.gwnet_model(xt[:,:,:,:2], adp=None, sampled_adj=sampled_adj_t,return_st=True)
        

        ys = ys[:,:,:,0].transpose(-2,-1)
        out = self.scaler_s.inverse_transform(out)
        ys = self.scaler_s.inverse_transform(ys)
        mae_loss = util.masked_mae(out, ys, 0.0)
        degree_loss = sampled_adj.mean().pow(2) 
        coral_loss = 0
        # apply coral on st features
        """
        for st_s_, st_t_ in zip(st_s, st_t):
            num_feat = st_s_.shape[1]
            s_idx = torch.randint(0, st_s_.shape[2], size=(16,), device=self.device)
            t_idx = torch.randint(0, st_t_.shape[2], size=(16,), device=self.device)
            sampled_st_s_ = st_s_[:,:,s_idx, -1].transpose(1, 2).reshape(-1, num_feat)
            sampled_st_t_ = st_t_[:,:,t_idx, -1].transpose(1, 2).reshape(-1, num_feat)
            coral_loss += self.coral_loss_unit(sampled_st_s_, sampled_st_t_)
        """
        # apply coral on st-t features
        """
        for st_s_, st_t_, t_s_, t_t_ in zip(st_s, st_t, tonly_s, tonly_t):
            num_feat = st_s_.shape[1]
            s_idx = torch.randint(0, st_s_.shape[2], size=(16,), device=self.device)
            t_idx = torch.randint(0, st_t_.shape[2], size=(16,), device=self.device)
            sampled_st_s_ = (st_s_ - t_s_)[:,:,s_idx, -1].transpose(1, 2).reshape(-1, num_feat)
            sampled_st_t_ = (st_t_ - t_t_)[:,:,t_idx, -1].transpose(1, 2).reshape(-1, num_feat)
            coral_loss += self.coral_loss_unit(sampled_st_s_, sampled_st_t_)
        """
        # apply coral on st - reconstructed st feature
        
        for st_s_, st_t_, t_s_, t_t_, recons in zip(st_s, st_t, tonly_s, tonly_t, self.reconstructors):
            num_feat = st_s_.shape[1]
            s_idx = torch.randint(0, st_s_.shape[2], size=(16,), device=self.device)
            t_idx = torch.randint(0, st_t_.shape[2], size=(16,), device=self.device)
            sampled_st_s_ = (st_s_ - recons(t_s_))[:,:,s_idx,-1].transpose(1, 2).reshape(-1, num_feat)
            sampled_st_t_ = (st_t_ - recons(t_t_))[:,:,t_idx,-1].transpose(1, 2).reshape(-1, num_feat)
            coral_loss += self.coral_loss_unit(sampled_st_s_, sampled_st_t_)
        

        # for st_s_, st_t_ in zip(st_s, st_t):
            # print("st_s_", st_s_.shape) # 64, 32, num_node, seq_len
            # print("st_t_", st_t_.shape)
        # for t_s_, t_t_ in zip(tonly_s, tonly_t):
            # print("t_s_", t_s_.shape) # 64, 32, num_node, seq_len, same as st_s, st_s
            # print("t_t_", t_t_.shape)

        loss = mae_loss + degree_loss * self.degree_reg + coral_loss * self.coral_reg
        loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(list(self.gwnet_model.parameters())+list(self.dglv2.parameters()), self.clip)
        self.optimizer_s.step()
        rmse = util.masked_rmse(out, ys, 0.0)
        mape = util.masked_mape(out, ys, 0.0)
        # print(out.shape)

        # train reconstructor
        recons_losses = []
        for st_s_, st_t_, t_s_, t_t_, recons in zip(st_s, st_t, tonly_s, tonly_t, self.reconstructors):
            st_s_ = st_s_.detach()
            st_t_ = st_t_.detach()
            t_s_ = t_s_.detach()
            t_t_ = t_t_.detach()
            st_s_recons = recons(t_s_)
            st_t_recons = recons(t_t_)
            recons_losses.append((st_s_recons - st_s_).pow(2).mean())
            recons_losses.append((st_t_recons - st_t_).pow(2).mean())
        recons_loss = sum(recons_losses)
        self.recons_opt.zero_grad()
        recons_loss.backward()
        self.recons_opt.step()
        
        return mae_loss.item(), rmse.item(), mape.item(), degree_loss.item(), coral_loss.item(), recons_loss.item()

    def source_eval(self, xs, ys, long_xs, return_val = False):
        # dglv2
        self.dglv2.eval()
        self.gwnet_model.eval()
        
        with torch.no_grad():
            xs = xs.to(self.device)
            ys = ys.to(self.device)
            long_xs = long_xs.to(self.device)

            # dglv2
            bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.dglv2(long_xs, tsformer=self.tsformer, compute_hidden=False, domain='s')
            if self.args.adaptadj:
                adp = self.adp_s()
                out = self.gwnet_model(xs[:,:,:,:2], adp=adp, sampled_adj=sampled_adj)
            else:
                out = self.gwnet_model(xs[:,:,:,:2], adp=None, sampled_adj=sampled_adj)
            ys = ys[:,:,:,0].transpose(-2,-1)
            out = self.scaler_s.inverse_transform(out)
            ys = self.scaler_s.inverse_transform(ys)
            loss = util.masked_mae(out, ys, 0.0)
            rmse = util.masked_rmse(out, ys, 0.0)
            mape = util.masked_mape(out, ys, 0.0)
        if return_val:
            return out, ys  
        else:
            return loss.item(), rmse.item(), mape.item()

    def fine_tune(self, xt, yt, long_xt, use_long=False):
        # dglv2
        self.dglv2.train()
        self.gwnet_model.train()
        self.optimizer_t.zero_grad()
        xt = xt.to(self.device)
        yt = yt.to(self.device)
        long_xt = long_xt.to(self.device)
        # print('long_xt', long_xt.shape)

        
        # dglv2
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.dglv2(long_xt, tsformer=self.tsformer, compute_hidden=True, domain='t')
        if self.args.adaptadj:
            adp = self.adp_t()
            out = self.gwnet_model(xt[:,:,:,:2], adp=adp, sampled_adj=sampled_adj, hidden_states=hidden_states[:,:,-1,:]) ## (batch_size, num_node, 12)
        else:
            out = self.gwnet_model(xt[:,:,:,:2], adp=None, sampled_adj=sampled_adj, hidden_states=hidden_states[:,:,-1,:])
        yt = yt[:,:,:,0].transpose(-2,-1)
        out = self.scaler_t.inverse_transform(out)
        yt = self.scaler_t.inverse_transform(yt)
        mae_loss = util.masked_mae(out, yt, 0.0)
        degree_loss = sampled_adj.mean().pow(2)
        loss = mae_loss + degree_loss * self.degree_reg
        loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(list(self.gwnet_model.parameters())+list(self.dglv2.parameters()), self.clip)
        self.optimizer_t.step()
        rmse = util.masked_rmse(out, yt, 0.0)
        mape = util.masked_mape(out, yt, 0.0)
        # print(out.shape)
        return mae_loss.item(), rmse.item(), mape.item(), degree_loss.item()

    def target_eval(self, xt, yt, long_xt=None, return_val=False, use_long=False):

        self.dglv2.eval()
        self.gwnet_model.eval()
        with torch.no_grad():
            xt = xt.to(self.device)
            yt = yt.to(self.device)
            long_xt = long_xt.to(self.device)

            
            bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.dglv2(long_xt, tsformer=self.tsformer, compute_hidden=True, domain='t')
            if self.args.adaptadj:
                adp = self.adp_t()
                out = self.gwnet_model(xt[:,:,:,:2], adp=adp, sampled_adj=sampled_adj,hidden_states=hidden_states[:,:,-1,:]) ## (batch_size, num_node, 12)

            else:
                out = self.gwnet_model(xt[:,:,:,:2], adp=None, sampled_adj=sampled_adj,hidden_states=hidden_states[:,:,-1,:])
            yt = yt[:,:,:,0].transpose(-2,-1)
            out = self.scaler_t.inverse_transform(out)
            yt = self.scaler_t.inverse_transform(yt)
            loss = util.masked_mae(out, yt, 0.0)
            rmse = util.masked_rmse(out, yt, 0.0)
            mape = util.masked_mape(out, yt, 0.0)
        if return_val:
            return out, yt
        else:
            return loss.item(), rmse.item(), mape.item()
