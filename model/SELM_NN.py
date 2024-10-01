import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import pandas as pd
import math
from itertools import chain
import itertools
import random
import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.nn.utils.prune as prune
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, enc_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._enc_dim = enc_dim
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        
        self.to_embedding = nn.Sequential(
            nn.Linear(1, int(self._embedding_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self._embedding_dim/2), self._embedding_dim)
        )
        
        self.to_unembedding = nn.Sequential(
            nn.Linear(self._embedding_dim, int(self._embedding_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self._embedding_dim/2), 1)
        )
        
        self.to_cls = nn.Sequential(
            #Set cluster number
            nn.Linear(self._embedding_dim * self._enc_dim, 10), 
            nn.Sigmoid()
        )

    def forward(self, inputs):
        #Get cls before quantization
        # cls = self.to_cls(inputs)
        
        # convert inputs from B, gen_dim -> B, 1, enc_dim -> B, code_embedding, enc_dim 
        inputs = rearrange(inputs, 'b (d l) -> b d l', d=1)
        inputs = inputs.contiguous()
        inputs = inputs.permute(0, 2, 1).contiguous()   ##'b d l -> b l d', d = 1
        # input_shape = inputs.shape
        
        # Flatten input
        inputs = self.to_embedding(inputs).contiguous()   ##'b l 1 -> b l d', d = embedding_dim
        flat_input = rearrange(inputs, 'b l d -> (b l) d')
        # flat_input = self.to_embedding(flat_input)
        
        
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Get the encoding that has the min distance
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Convert to one-hot encodings
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize the latents and unflatten
        quantized_latent = torch.matmul(encodings, self._embedding.weight)  ##(b l) d
        quantized_latent = rearrange(quantized_latent, '(b l) d -> b l d', l = self._enc_dim)  ##(b l) d -> b l d
        # quantized_latent = self.to_unembedding(quantized_latent)        ##b l d -> b l 1
        
        cls_quantized = rearrange(quantized_latent, 'b l d -> b (l d)') 
        cls = self.to_cls(cls_quantized)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized_latent.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized_latent, inputs.detach())
        
        vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        #### Consider to improve it
        quantized_latent = inputs + (quantized_latent - inputs).detach()  ##b l d
        quantized_latent = rearrange(quantized_latent, 'b l d -> b (l d)')  ##3-tensor -> 2-tensor
        quantized_latent = quantized_latent.contiguous()
    
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        #return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        return vq_loss, quantized_latent, perplexity, encodings, cls

class ResidualStack(nn.Module):
    def __init__(self, encoder_dim):
        super(ResidualStack, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(encoder_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, encoder_dim),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_dim)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class Encoder(nn.Module):
    def __init__(self, input_size, enc_lay1, enc_dim, dropout):
        super(Encoder, self).__init__()

        self.lay1 = nn.Sequential(
            nn.Linear(input_size, enc_lay1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lay2 = nn.Sequential(
            nn.Linear(enc_lay1, enc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self._residual_stack = ResidualStack(enc_dim)

    def forward(self, inputs):
        x = self.lay1(inputs)
        x = self.lay2(x)
        x = self._residual_stack(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_size, enc_dim, enc_lay1, dropout):
        super(Decoder, self).__init__()
        
        self.pre_layer = nn.Sequential(
            nn.Linear(embedding_dim*enc_dim, enc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._residual_stack = ResidualStack(enc_dim)
        
        self.lay1 = nn.Sequential(
            nn.Linear(enc_dim, enc_lay1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lay2 = nn.Sequential(
            nn.Linear(enc_lay1, input_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, inputs):
        x = self.pre_layer(inputs)
        x = self._residual_stack(x)
        x = self.lay1(x)
        x = self.lay2(x)
        return x


class SELM(nn.Module):
    def __init__(self, input_size, enc_dim, enc_lay1, num_embeddings, embedding_dim, commitment_cost, dropout, decay=0):
        super(SELM, self).__init__()
        
        self._encoder = Encoder(input_size, enc_lay1, enc_dim, dropout)

        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, enc_dim, commitment_cost)
            
        self._decoder = Decoder(embedding_dim, input_size, enc_dim, enc_lay1, dropout)

    def forward(self, x):
        z = self._encoder(x)
        loss, quantized, perplexity, _, cls = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity, quantized, cls, z