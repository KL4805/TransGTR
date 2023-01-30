import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from util import *
import time
import copy

class DistilTSFormer(nn.Module):
    # This class implements a TSFormer which is used for distillation. 
    # The major difference is that this class uses absolute time embedding (diw, tid), while the original TSFormer uses relative time embedding
    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, mask_ratio, encoder_depth, decoder_depth, mode='pre-train'):
        super().__init__()
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio

        self.selected_feature = 0
        # norm layers
        # this is a potential for improvement: for sequential tasks, we should use batch norm
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        # encoder specific modules
        
        self.patch_embedding = PatchEmbedding(self.patch_size, in_channel, embed_dim, None)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        self.diw_embedding = nn.Parameter(torch.randn(7, self.embed_dim), requires_grad=True)
        # 5-minute embedding, which is tested to cause overfitting
        ## self.tid_embedding = nn.Parameter(torch.randn(288, self.embed_dim), requires_grad=True)
        # hourly embedding
        ## self.tid_embedding = nn.Parameter(torch.randn(24, self.embed_dim), requires_grad=True)
        self.mask = MaskGenerator(num_token, mask_ratio)
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()
        self.diw_tid_idx = torch.LongTensor(list(range(int(patch_size/2), num_token * patch_size, patch_size)))
        # print(self.diw_tid_idx, len(self.diw_tid_idx))

    def encoding(self, history_data, mask=True, index=None):

        t0 = time.time() 
        # Encoding process of the DistilTSFormer:
        # Input embedding, plus tid, diw, patchify, mask, Transformer layers
        # long_term_history: with shape [Batch, num_node, feat, seq_len]
        # where L = num_patch * patch_len
        batch_size, num_node, _ ,_ = history_data.shape
        # print('history_data.shape', history_data.shape)

        # 5-minute embedding
        ## tid = (history_data[:,:,1,:] * 288).long()

        # hour embedding
        tid = (history_data[:,:,1,:] * 24).long()

        diw = (history_data[:,:,2,:]).long()
        # tid: batch_size, num_node, seq_len
        # print('tid', tid.shape)
        # print('diw', diw.shape)

        patches = self.patch_embedding(history_data[:,:,:self.in_channel,:])
        patches = patches.transpose(-1, -2) 
        # print('patches', patches.shape)
        # patches: (batch_size, num_node, num_token, embed_dim)
        actual_num_token = int(patches.shape[2])
        # print(actual_num_token)
        diw_embedding = self.diw_embedding[diw[:,:,self.diw_tid_idx[:actual_num_token]],:]
        # print("diw_embedding", diw_embedding.shape)
        ## tid_embedding = self.tid_embedding[tid[:,:,self.diw_tid_idx],:]
        patches += diw_embedding
        ## patches += tid_embedding
        if index is None:
            patches = self.positional_encoding(patches)
        else:
            patches = self.positional_encoding(patches, index=list(range(index[0], index[1])))


        if mask:
            unmasked_token_index, masked_token_index = self.mask(actual_num_token)
            encoder_input = patches[:,:,unmasked_token_index,:]
        else:
            unmasked_token_index, masked_token_index=None, None
            encoder_input=patches
        ## print("encoder_input", encoder_input.shape)
        # encoding through transformer
        
        hidden_states_unmasked = self.encoder(encoder_input)
        # print('hidden_states_unmasked', hidden_states_unmasked.shape)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_node, -1, self.embed_dim)
        # print('hidden_states_unmasked after norm', hidden_states_unmasked.shape)

        # If there is mask, we also need to pass the absolute diw/tid to the decoder
        # such that the decoder knows the absolute position. 
        if mask:
            masked_diw_tid_idx = self.diw_tid_idx[masked_token_index]
            # print(masked_token_index[:20])
            # print(masked_diw_tid_idx[:20])
            masked_token_tid = tid[:,:,masked_diw_tid_idx]
            masked_token_diw = diw[:,:,masked_diw_tid_idx]
            # print(masked_token_tid[:20])
            # print(masked_token_diw[:20])

        else:
            masked_token_diw, masked_token_tid = None, None

        return hidden_states_unmasked, unmasked_token_index, masked_token_index, masked_token_diw, masked_token_tid
    
    def decoding(self, hidden_states_unmasked, masked_token_index, masked_token_diw, masked_token_tid):
        # Decoding of the TSFormer
        # args: hidden_states_unmasked: hidden states with shape (batch_size, num_node, num_unmasked, feat)
        # masked_token_index: (list)
        batch_size, num_nodes, _, _ = hidden_states_unmasked.shape

        # encoder2decoder
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)
        
        # mask tokens
        # add positional encoding
        masked_diw = self.diw_embedding[masked_token_diw, :]
        ## masked_tid = self.tid_embedding[masked_token_tid, :]
        masked_token = self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), self.embed_dim)
        # print('masked_token', masked_token.shape)
        # print('masked_tid', masked_tid.shape)
        ## hidden_states_masked = masked_diw + masked_tid + masked_token
        hidden_states_masked = masked_diw + masked_token
        ## hidden_states_masked = masked_token
        hidden_states_masked = self.positional_encoding(hidden_states_masked, index=masked_token_index)

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)

        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        # get reconstructed masked tokens and corresponding ground truth to compute loss
        # args: reconstruction_full: reconstructed full tokens
        
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:,:,len(unmasked_token_index):,:]
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)
        # print("reconstruction_masked_tokens", reconstruction_masked_tokens.shape)
        # print("real_value_full",real_value_full.shape)
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)
        label_full = label_full[:,:,:,0,:].transpose(1, 2)
        # print("label-full", label_full.shape)
        label_masked_tokens = label_full[:,:,masked_token_index,:].contiguous()
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)
        return reconstruction_masked_tokens, label_masked_tokens
        

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.diw_embedding, -.02, .02)
        ## nn.init.uniform_(self.tid_embedding, -.02, .02)
        nn.init.uniform_(self.positional_encoding.position_embedding, -0.02, 0.02)
        # mask token
        nn.init.trunc_normal_(self.mask_token, std=0.02) 

    def forward(self, history_data, mask=True, index=None):
        # forward pass of the DistilFormer
        # mode: 'mask' for doing masked autoencoding training, 'unmask' for doing inference without mask. 
        # args: history_data: (Batch, seq_len, num_node, feat), where feat=3, 
        history_data = history_data.permute(0, 2, 3, 1)
        # now, history_data is of shape (batch, num_node, feat, seq_len)
        batch_size, num_node, _, seq_len = history_data.shape
        if mask:
            hidden_states_unmasked, unmasked_token_index, masked_token_index, masked_token_diw, \
                masked_token_tid = self.encoding(history_data,mask=True)
            ## print("%.4fs Finish Encoding" % (time.time() - t1))
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index, masked_token_diw, masked_token_tid)
            ## print("%.4fs Finish decoding" % (time.time() - t1))
            reconstruction_masked_tokens, label_masked_tokens = \
                self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)
            ## print("%.4fs finish forward" % (time.time() - t1))
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _, _, _ = self.encoding(history_data, mask=False, index=index)
            return hidden_states_full




class TSFormer(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, mask_ratio, encoder_depth, decoder_depth, 
        mode="pre-train", distiler=False):

        super().__init__()        
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.distiler = distiler

        self.selected_feature = 0
        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # encoder specific modules
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        self.mask = MaskGenerator(num_token, mask_ratio)
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # decoder specific modules
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim)
        
        # mask token initialization
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # regression layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()
        if self.distiler:
            self.distiler_unit = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(), 
                nn.Linear(self.embed_dim, self.embed_dim)
            )


    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        # mask token
        nn.init.trunc_normal_(self.mask_token, std=0.02) 


    def encoding(self, long_term_history, mask=True): 
        """Encoding process of TSFormer: patchify, positional encoding, mask, Transformer layers.

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """
        t0 = time.time() 
        batch_size, num_nodes, _, _ = long_term_history.shape
        # patchify and embed input
        patches = self.patch_embedding(long_term_history)     # B, N, d, P
        patches = patches.transpose(-1, -2)         # B, N, P, d
        ## print("finish patchify %.4fs" % (time.time() - t0))
        # positional embedding
        patches = self.positional_encoding(patches) # batch, num_node, num_patch, num_feat(96)
        ## print("finish position encoding %.4fs" % (time.time() - t0))
        actual_num_token = patches.shape[2]
        # print('actual_num_token', actual_num_token)


        # mask
        if mask:
            unmasked_token_index, masked_token_index = self.mask(actual_num_token)
            encoder_input = patches[:, :, unmasked_token_index, :]
            # print("patches before mask", patches.shape) (8, 207, 168, 96)
            #print('patches after mask', encoder_input.shape) (8, 207, 42, 96), only 1/4 of the unmasked. 
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches


        # encoding
        ## print('encoder_input', encoder_input.shape)
        hidden_states_unmasked = self.encoder(encoder_input)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
        # print('hidden_states_unmasked', hidden_states_unmasked.shape) (8, 207, 42, 96), the (-1) dimension is the number of unmasked tokens
        ## print("finish attention encoding %.4fs" % (time.time() - t0))
        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, masked_token_index):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        batch_size, num_nodes, _, _ = hidden_states_unmasked.shape

        # encoder 2 decoder layer, linear layer at the last dimension
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)

        # add mask tokens
        hidden_states_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1]),
            index=masked_token_index
            )
        # mask_token is all zeros. We only feed in positional encoding
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d
        # (8, 207, 168, 96)

        # decoding
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)
        # print("decoder output", hidden_states_full.shape)
        # (8, 207, 168, 96)

        # prediction (reconstruction)
        # output layer is a nn.Linear, from hidden_dim (96) to patch_size (12)
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))
        # reconstruction_full: (8, 207, 168, 12)
        # print('reconstruction_full', reconstruction_full.shape)

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
        # masked tokens are in the first (# unmasked index) dimension
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N
        # print('reconstruction_masked_tokens', reconstruction_masked_tokens.shape)
        # (8, 1512, 207), 1512 = 168 * 12 * 0.75
        
        # print("real_value_full", real_value_full.shape) (8, 207, 1, 2016)
        # reshape to (8, 2016, 207, 1)
        # unfold(1, patch_size, patch_size) means that the unfolding happend on dimension 1, size is patch size, step is patch size
        # returns a tensor of (batch, num_patch, num_node, 1, patch_size)
        # essentially divides the tensor using a sliding window
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)
        label_full = label_full[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L, (8, 207, 168, 12)
        # print('label_full', label_full.shape)
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d, only extract masked tokens
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N
        # (8, 1512, 207)

        return reconstruction_masked_tokens, label_masked_tokens


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """feed forward of the TSFormer.
            TSFormer has two modes: the pre-training mode and the forecasting mode,
                                    which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            history_data (torch.Tensor): very long-term historical time series with shape B, L * P, N, 1.

        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N, 1]
                torch.Tensor: the ground truth of the masked tokens. Shape [B, L * P * r, N, 1]
                dict: data for plotting.
            forecasting:
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, 1].
        """
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        # print("TSFormer history data", history_data.shape), 8, 207, 1, 2016
        # feed forward
        if self.mode == "pre-train":
            # encoding
            t1 = time.time()
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            ## print("%.4fs finish encoding" % (time.time() - t1))
            # decoding
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index)
            ## print("%.4fs finish decoding" % (time.time() - t1))
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)
            ## print("%.4fs finish forward" % (time.time() - t1))
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full


class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim     # embed 96
        self.input_embedding = nn.Conv2d(
                                        in_channel,
                                        embed_dim,
                                        kernel_size=(self.len_patch, 1),
                                        stride=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """

        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1) # B, N, C, L, 1, (8, 207, 1, 2016, 1)
        # B*N,  C, L, 1
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1) # (8*207, 1, 2016, 1)
        # B*N,  d, L/P, 1
        output = self.input_embedding(long_term_history) # the input embedding conv2d actually does fully connected, len(patch)-> embed dim
        # norm
        output = self.norm_layer(output)
        # print('patch output', output.shape)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P, (8, 207, embed_dim, num_patch(168))
        assert output.shape[-1] == len_time_series / self.len_patch
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)

    def forward(self, input_data, index=None, abs_idx=None):
        ## if index is not None:
        ##     print(index[0], index[-1])
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.view(batch_size*num_nodes, num_patches, num_feat) # (8 * 207, 168, 96)
        # print('input_data', input_data.shape)
        # positional encoding
        if index is None:
            # actually goes this way
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
            # print('pe', pe.shape) # 1, 168, 96
            # 96 is the embedding dim
        else:
            pe = self.position_embedding[index].unsqueeze(0)
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        # reshape
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        # reshape to (8, 207, 168, 96)
        return input_data


class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        # print("mask_ratio", self.mask_ratio) 
        # mask_ratio = 0.75
        # mask 0.75 of the patches
        self.sort = True

    def uniform_rand(self, actual_num_token=0):
        if actual_num_token == 0:
            actual_num_token = self.num_tokens
        mask = list(range(int(actual_num_token)))
        random.shuffle(mask)
        mask_len = int(actual_num_token * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self, actual_num_token=0):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand(actual_num_token)
        return self.unmasked_tokens, self.masked_tokens





class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        # print("hidden_dim", hidden_dim)
        # print('mlp_ratio', mlp_ratio)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        ## print("hidden_dim * mlp_ratio", hidden_dim * mlp_ratio)
        ## print("hidden_dim", hidden_dim)
        ## print('mlp_ratio', mlp_ratio)
        # mlp_ratio*hidden_dim is the dimension of the feedforward network (hidden layer)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        # sqrt(d)
        src = src * math.sqrt(self.d_model)
        src = src.view(B*N, L, D)
        src = src.transpose(0, 1)
        # src shape is taken as (batch, seq_length, feature dimension)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)
        # reshape back to (B, N, L, D)
        # print("transformer encoder output", output.shape)
        # (8, 207, 42, 96)
        return output

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A = A.to(x.device)
        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes, embed_dim=10):
        super().__init__()
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(embed_dim, num_nodes), requires_grad=True)
        self.num_nodes=num_nodes
    
    def forward(self):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        return adp


class GraphWaveNet(nn.Module):
    """
        Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling.
        Link: https://arxiv.org/abs/1906.00121
        Ref Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    """

    def __init__(self, supports, dropout=0.3, adaptadj=True, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, **kwargs):
        """
            kindly note that although there is a 'supports' parameter, we will not use the prior graph if there is a learned dependency graph.
            Details can be found in the feed forward function.
        """
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.adaptadj = adaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        # self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.fc_his = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 2
        if supports is not None:
            self.supports_len += len(supports)

        if adaptadj:
            # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
            # self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
            self.supports_len +=1

        print("support_len", self.supports_len)
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                # self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

        self.receptive_field = receptive_field

    def _calculate_random_walk_matrix(self, adj_mx):
        if len(adj_mx.shape) == 3:
            B, N, N = adj_mx.shape

            adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).expand(B, N, N).to(adj_mx.device)
            d = torch.sum(adj_mx, 2)
            d_inv = 1. / d
            d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(adj_mx.device), d_inv)
            d_mat_inv = torch.diag_embed(d_inv)
            random_walk_mx = torch.bmm(d_mat_inv, adj_mx)
        else:
            N, N = adj_mx.shape
            adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).to(adj_mx.device)
            d = torch.sum(adj_mx, 1)
            d_inv = 1. / d
            d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(adj_mx.device), d_inv)
            d_mat_inv = torch.diag_embed(d_inv)
            # print("d_mat_inv", d_mat_inv.shape)
            # print('adj_mx', adj_mx.shape)
            random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, input, hidden_states=None, sampled_adj=None, adp=None, return_st = False):
        """feed forward of Graph WaveNet.

        Args:
            input (torch.Tensor): input history MTS with shape [B, L, N, C].
            His (torch.Tensor): the output of TSFormer of the last patch (segment) with shape [B, N, d].
            adj (torch.Tensor): the learned discrete dependency graph with shape [B, N, N].

        Returns:
            torch.Tensor: prediction with shape [B, N, L]
        """
        
        # reshape input: [B, L, N, C] -> [B, C, N, L]
        input = input.transpose(1, 3)
        # feed forward
        input = nn.functional.pad(input,(1,0,0,0))

        input = input[:, :2, :, :]
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0
        new_supports = copy.deepcopy(self.supports)
        # print('sampled_adj', sampled_adj)
        if sampled_adj is not None:
            # print("Sampled adj", sampled_adj.shape) (batch, 207, 207)
            # Why? 
            # ====== if use learned adjacency matrix, then reset the self.supports ===== #
            new_supports += [self._calculate_random_walk_matrix(sampled_adj)]
            new_supports += [self._calculate_random_walk_matrix(sampled_adj.transpose(-1, -2))]

        # calculate the current adaptive adj matrix

        if self.adaptadj:
            # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            if new_supports is None:
                new_supports=[adp]
            else:
                new_supports += [adp]
        # print('new_supports', len(new_supports))

        # print("length of supports", len(new_supports))
        # 3 adjacency matrices, with the adaptive one in GWNet
        # print("len_supports", len(new_supports))
        # print("len(supports)", len(self.supports))
        # WaveNet layers
        temporal_only = []
        spatio_temporal = []
        
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            if i != self.blocks * self.layers:
                temporal_only.append(x)

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)
            if i != self.blocks * self.layers:
                spatio_temporal.append(x)
            ## if self.gcn_bool and self.supports is not None:
            ##     if self.adaptadj:
            ##         x = self.gconv[i](x, new_supports)
            ##     else:
            ##         x = self.gconv[i](x,self.supports)
            ## else:
            ##     x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            # x = self.bn[i](x)
        # print("input long term state", hidden_states.shape) (32, 207, 96)
        
        if hidden_states is not None:
            hidden_states = self.fc_his(hidden_states)        # B, N, D
            hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
            # print("hidden state added to gwnet output", hidden_states.shape) #(32, 256, 207, 1)
            # print('hidden_states mean', hidden_states.mean())
            # print('skip mean', skip.mean())
            skip = skip + hidden_states
            ## skip = hidden_states
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # reshape output: [B, P, N, 1] -> [B, N, P]
        x = x.squeeze(-1).transpose(1, 2)
        if not return_st:
            return x
        else:
            return x, temporal_only, spatio_temporal

def sample_gumbel(shape, eps=1e-20, device=None):
    uniform = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(uniform + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """

    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

class DiscreteGraphLearningV2(nn.Module):
    # This module implements structure learning based on
    # fixed tsformer and trainable link predictor
    # this module should combine both source and target data
    # as the input feature space is now the same (tsformer outputs)
    def __init__(self, device, source_name, target_name, k, input_seq_len=2016, output_seq_len=12, data_number=0, tsformer_type='TSFormer'):
        super().__init__()

        self.device = device 
        self.k = k
        self.tsformer_type = tsformer_type
        self.num_nodes = {'METR-LA':207, 'PEMSD7M':228, 'PEMS-BAY':325, 'HKTSM':608}
        self.train_length = {'METR-LA':23990,'PEMSD7M':8870, 'PEMS-BAY':36482, 'HKTSM':21152}
        self.input_seq_len = input_seq_len
        self.num_nodes_s = self.num_nodes[source_name]
        self.num_nodes_t = self.num_nodes[target_name]
        self.train_length_s = self.train_length[source_name]
        self.train_length_t = self.train_length[target_name]
        if self.tsformer_type == 'TSFormer':
            self.node_feats_s = torch.from_numpy(load_pkl('../data/' + source_name + '/data_in12_out12.pkl')['processed_data']).float()[:self.train_length_s,:,:1]
            self.node_feats_t = torch.from_numpy(load_pkl('../data/' + target_name + '/data_in12_out12.pkl')['processed_data']).float()[:self.train_length_t,:,:1]
        elif self.tsformer_type == 'DistilFormer':
            # include diw and tid
            self.node_feats_s = torch.from_numpy(load_pkl('../data/' + source_name + '/data_in12_out12.pkl')['processed_data']).float()[:self.train_length_s,:,:]
            self.node_feats_t = torch.from_numpy(load_pkl('../data/' + target_name + '/data_in12_out12.pkl')['processed_data']).float()[:self.train_length_t,:,:]
        # print('self.node_feats_s', self.node_feats_s.shape)
        # print('self.node_feats_t', self.node_feats_t.shape) # (seq_len, num_node)
        if data_number != 0:
            self.node_feats_t = self.node_feats_t[-data_number*288:,:]
        
        # process data for tsformer input
        # tsformer input: (batch, seq_len, num_node, channel)
        self.sl_batch_s = []
        self.sl_batch_t = []
        max_slices = 10
        for i in range(int(self.node_feats_s.shape[0]/self.input_seq_len)):
            start_index = self.node_feats_s.shape[0] - self.input_seq_len * (i+1)
            end_index = self.node_feats_s.shape[0] - self.input_seq_len * i
            # start_index = -(i+1) * self.input_seq_len - 1
            # end_index = -i * self.input_seq_len - 1
            self.sl_batch_s.append(self.node_feats_s[start_index:end_index, :,:])
            if i == max_slices-1:
                break
        for i in range(int(self.node_feats_t.shape[0]/self.input_seq_len)):
            start_index = self.node_feats_t.shape[0] - self.input_seq_len * (i+1)
            end_index = self.node_feats_t.shape[0] - self.input_seq_len * i
            # start_index = -(i+1) * self.input_seq_len - 1
            # end_index = -i * self.input_seq_len - 1
            self.sl_batch_t.append(self.node_feats_t[start_index:end_index, :,:])
            if i == max_slices-1:
                break
        self.sl_batch_s = torch.stack(self.sl_batch_s, dim=0).to(device)
        self.sl_batch_t = torch.stack(self.sl_batch_t, dim=0).to(device)
        # print('sl_batch_s', self.sl_batch_s.shape) # (num_slice, seq_len, num_node, channel)
        print('sl_batch_t', self.sl_batch_t.shape)

        # define link predictor
        self.embedding_dim=96
        self.conv1 = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=12, stride=2)
        self.conv2 = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=12, stride=2)
        ## self.conv3 = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=12, stride=2)
        ## self.conv4 = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=12, stride=2)
        if data_number == 7:
            embedding_expansion=68
        else:
            embedding_expansion=20
        self.fc_cat = nn.Linear(self.embedding_dim*embedding_expansion, self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim, 2)
    
    def forward(self, long_term_history, tsformer, compute_hidden=True, domain='t'):
        # args:
        # long_term_history: batch_size, seq_len, num_node, channel
        # tsformer: the tsformer to compute node features
        
        
        batch_size, _, num_nodes, _ = long_term_history.shape
        # generate node features for structure learning
        if domain == 's':
            if self.tsformer_type == 'TSFormer':
                node_feat = tsformer(self.sl_batch_s)
            elif self.tsformer_type == 'DistilFormer':
                node_feat = tsformer(self.sl_batch_s, mask=False)
        elif domain == 't':
            if self.tsformer_type == 'TSFormer':
                node_feat = tsformer(self.sl_batch_t)
            elif self.tsformer_type == 'DistilFormer':
                node_feat = tsformer(self.sl_batch_t, mask=False)
        # print('node_feat', node_feat.shape) # num_slices, num_node, num_token, embed_dim 
        sl_batch_size, _, num_token, embed_dim = node_feat.shape
        ### node_feat = node_feat.mean(0)[:,-1,:] # last token, this is tested better than mean over all tokens
        ## node_feat = node_feat.mean(0).mean(1) # mean over all tokens
        # node_feat: num_node, embed_dim
        node_feat = node_feat.view(sl_batch_size * num_nodes, num_token, embed_dim).transpose(-2, -1)
        # print("node_feat", node_feat.shape)
        # node_feat: (batch_size * num_nodes, embed_dim, num_token(seq_len))
        node_feat = F.relu(self.conv1(node_feat))
        # print('node_feat', node_feat.shape) # seq len 61
        node_feat = F.relu(self.conv2(node_feat))
        # print('node_feat', node_feat.shape) # seq len 50
        ### node_feat = F.relu(self.conv3(node_feat))
        # print('node_feat', node_feat.shape) # seq len 20
        ### node_feat = F.relu(self.conv4(node_feat))
        # print('node_feat', node_feat.shape) # seq len 5
        node_feat = node_feat.view(sl_batch_size, num_nodes, self.embedding_dim, -1).contiguous().mean(0).view(num_nodes, -1)
        # print('node_feat', node_feat.shape)
        off_diag = np.ones([num_nodes, num_nodes])
        rec_idx = torch.LongTensor(np.where(off_diag)[0]).to(self.device)
        send_idx = torch.LongTensor(np.where(off_diag)[1]).to(self.device)
        senders = node_feat[send_idx, :]
        receivers = node_feat[rec_idx, :]
        x = torch.cat([senders, receivers], dim=1)
        x = F.relu(self.fc_cat(x))
        bernoulli_unnorm = self.fc_out(x)

        sampled_adj = gumbel_softmax(bernoulli_unnorm, temperature=0.5, hard=True)
        sampled_adj = sampled_adj[..., 0].clone().reshape(num_nodes, -1)
        mask = torch.eye(num_nodes, num_nodes).bool().to(self.device)
        sampled_adj.masked_fill_(mask, 0)

        if compute_hidden:
            # hidden_states=tsformer(long_term_history[..., [0]])
            if self.tsformer_type == 'TSFormer':
                hidden_states=tsformer(long_term_history[...,[0]])
            elif self.tsformer_type == 'DistilFormer':
                # print("long_term_history", long_term_history.shape)
                hidden_states=tsformer(long_term_history, mask=False)
                # print('hidden_states', hidden_states.shape)
        else:
            hidden_states=None
        # at present, we ignore the knn prior graph
        adj_knn=None

        return bernoulli_unnorm, hidden_states, adj_knn, sampled_adj


class CORAL_loss(nn.Module):
    # implementation copied from DomainBed, 
    # https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
    def __init__(self):
        super(CORAL_loss, self).__init__()
    
    def forward(self, source, target):
        batch_size = source.shape[0]
        mean_s = source.mean(0, keepdims=True)
        mean_t = target.mean(0, keepdims=True)
        cent_s = source-mean_s
        cent_t = target-mean_t
        cova_s = (cent_s.t() @ cent_s) / (batch_size - 1)
        cova_t = (cent_t.t() @ cent_t) / (batch_size - 1)
        mean_diff = (mean_s - mean_t).pow(2).mean()
        cova_diff = (cova_s - cova_t).pow(2).mean()

        return mean_diff + cova_diff

class FCReconstructor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=(1, 1))
    
    def forward(self, input_feat):
        x = self.conv1(input_feat)
        x = F.relu(x)
        x = self.conv2(x)
        return x
