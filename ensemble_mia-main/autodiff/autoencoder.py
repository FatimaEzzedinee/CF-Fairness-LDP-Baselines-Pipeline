import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
#import process_edited as pce
from .process_GQ import DataFrameParser

import tqdm
import gc
import random
import pandas as pd
import matplotlib.pyplot as plt

def _make_mlp_layers(num_units):
    layers = nn.ModuleList([
        nn.Linear(in_features, out_features)
        for in_features, out_features in zip(num_units, num_units[1:])
    ])
    return layers

class DeapStack(nn.Module):
    ''' Simple MLP body. '''
    def __init__(self,  n_bins, n_cats, n_nums, cards, in_features, hidden_size, bottleneck_size, num_layers, device='cuda'):
        super().__init__()       
        self.device = device
        
        encoder_layers = num_layers >> 1
        decoder_layers = num_layers - encoder_layers - 1
        self.encoders = _make_mlp_layers([in_features] + [hidden_size] * encoder_layers)
        self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.decoders = _make_mlp_layers([bottleneck_size] + [hidden_size] * decoder_layers)

        self.n_bins = n_bins
        self.n_cats = n_cats       
        self.n_nums = n_nums

        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
        self.cats_linears = nn.ModuleList([nn.Linear(hidden_size, card) for card in cards])
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None
        
        # Move model to device
        self.to(device)
               
    def forward_pass(self, x):
        for encoder_layer in self.encoders:
            x = F.relu(encoder_layer(x))
        x = b = self.bottleneck(x)
        for decoder_layer in self.decoders:
            x = F.relu(decoder_layer(x))
        return [b, x]

    def forward(self, x):
        # Ensure input is on correct device
        x = x.to(self.device)
        outputs = dict()
        
        num_min_values, _ = torch.min(x[:,self.n_bins+self.n_cats:self.n_bins+self.n_cats+self.n_nums], dim=0)
        num_max_values, _ = torch.max(x[:,self.n_bins+self.n_cats:self.n_bins+self.n_cats+self.n_nums], dim=0)
        
        decoder_output = self.forward_pass(x)[1]
        
        if self.bins_linear:
            outputs['bins'] = self.bins_linear(decoder_output)

        if self.cats_linears:
            outputs['cats'] = [linear(decoder_output) for linear in self.cats_linears]            
            
        if self.nums_linear:
            before_threshold = self.nums_linear(decoder_output)
            outputs['nums'] = before_threshold
            
            for col in range(len(num_min_values)):
                outputs['nums'][:,col] = torch.where(before_threshold[:,col] < num_min_values[col], num_min_values[col], before_threshold[:,col])     
                outputs['nums'][:,col] = torch.where(before_threshold[:,col] > num_max_values[col], num_max_values[col], before_threshold[:,col]) 
                                                     
        return outputs

    def featurize(self, x):
        x = x.to(self.device)
        return self.forward_pass(x)[0]
    
    def decoder(self, latent_feature, num_min_values, num_max_values):
        # Ensure all inputs are on correct device
        latent_feature = latent_feature.to(self.device)
        num_min_values = num_min_values.to(self.device)
        num_max_values = num_max_values.to(self.device)
        
        decoded_outputs = dict()
        x = latent_feature

        for layer in self.decoders:
            x = F.relu(layer(x))
        last_hidden_layer = x
        
        if self.bins_linear:
            decoded_outputs['bins'] = self.bins_linear(last_hidden_layer)

        if self.cats_linears:
            decoded_outputs['cats'] = [linear(last_hidden_layer) for linear in self.cats_linears]            
            
        if self.nums_linear:
            d_before_threshold = self.nums_linear(last_hidden_layer)
            decoded_outputs['nums'] = d_before_threshold
            
            for col in range(len(num_min_values)):
                decoded_outputs['nums'][:,col] = torch.where(d_before_threshold[:,col] < num_min_values[col], num_min_values[col], d_before_threshold[:,col])     
                decoded_outputs['nums'][:,col] = torch.where(d_before_threshold[:,col] > num_max_values[col], num_max_values[col], d_before_threshold[:,col]) 
                
        return decoded_outputs

    
def auto_loss(inputs, reconstruction, n_bins, n_nums, n_cats, cards, device='cuda'):
    """ Calculating the loss for DAE network.
        BCE for masks and reconstruction of binary inputs.
        CE for categoricals.
        MSE for numericals.
        reconstruction loss is weighted average of mean reduction of loss per datatype.
        mask loss is mean reduced.
        final loss is weighted sum of reconstruction loss and mask loss.
    """
    # Ensure inputs are on correct device
    inputs = inputs.to(device)
    
    bins = inputs[:,0:n_bins]
    cats = inputs[:,n_bins:n_bins+n_cats]
    nums = inputs[:,n_bins+n_cats:n_bins+n_cats+n_nums]
    
    reconstruction_losses = dict()
        
    if 'bins' in reconstruction:
        reconstruction_losses['bins'] = F.binary_cross_entropy_with_logits(reconstruction['bins'], bins)

    if 'cats' in reconstruction:
        cats_losses = []
        for i in range(len(reconstruction['cats'])):
            cats_losses.append(F.cross_entropy(reconstruction['cats'][i], cats[:, i].long()))
        reconstruction_losses['cats'] = torch.stack(cats_losses).mean()        
        
    if 'nums' in reconstruction:
        reconstruction_losses['nums'] = F.mse_loss(reconstruction['nums'], nums)
        
    reconstruction_loss = torch.stack(list(reconstruction_losses.values())).mean()
    return reconstruction_loss


def sigmoid_threshold(logits, device='cuda'):
    sigmoid_output = torch.sigmoid(logits)
    threshold_output = torch.where(sigmoid_output > 0.5, 
                                   torch.tensor(1, device=device), 
                                   torch.tensor(0, device=device))
    return threshold_output

def softmax_with_max(predictions):
    # Applying softmax function
    probabilities = F.softmax(predictions, dim=1)
    
    # Getting the index of the maximum element
    max_indices = torch.argmax(probabilities, dim=1)
    
    return max_indices

def train_autoencoder(df, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold, device='cuda'):
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Training autoencoder on device: {device}")
    
    parser = DataFrameParser().fit(df, threshold)
    data = parser.transform()
    data = torch.tensor(data.astype('float32'), device=device)  # Move data to device immediately
    print(f"Data shape: {data.shape}, device: {data.device}")

    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']; n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']; cards = datatype_info['cards']
    
    print(f"Data types - bins: {n_bins}, cats: {n_cats}, nums: {n_nums}")

    DS = DeapStack(n_bins, n_cats, n_nums, cards, data.shape[1], 
                   hidden_size=hidden_size, bottleneck_size=data.shape[1], 
                   num_layers=num_layers, device=device)
    
    print(f"Model created and moved to device: {next(DS.parameters()).device}")

    optimizer = Adam(DS.parameters(), lr=lr, weight_decay=weight_decay)

    tqdm_epoch = tqdm.tqdm(range(n_epochs))

    losses = []
    all_indices = list(range(data.shape[0]))

    for epoch in tqdm_epoch:
        batch_indices = random.sample(all_indices, min(batch_size, len(all_indices)))
        batch_data = data[batch_indices,:]  # Data is already on device

        output = DS(batch_data)

        l2_loss = auto_loss(batch_data, output, n_bins, n_nums, n_cats, cards, device)
        optimizer.zero_grad()
        l2_loss.backward()
        optimizer.step()

        # Memory cleanup
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        # Print the training loss over the epoch.
        losses.append(l2_loss.item())

        tqdm_epoch.set_description('Average Loss: {:5f}'.format(l2_loss.item()))
    
    # Calculate min/max values on device
    num_min_values, _ = torch.min(data[:,n_bins+n_cats:n_bins+n_cats+n_nums], dim=0)
    num_max_values, _ = torch.max(data[:,n_bins+n_cats:n_bins+n_cats+n_nums], dim=0)

    print(f"Min/max values computed on device: {num_min_values.device}")

    # Get latent features
    latent_features = DS.featurize(data)
    print(f"Latent features shape: {latent_features.shape}, device: {latent_features.device}")
    
    # Test decoder
    output = DS.decoder(latent_features, num_min_values, num_max_values)
    print(f"Decoder test completed")

    # Create a wrapper for the decoder that maintains device consistency
    class DeviceAwareDecoder:
        def __init__(self, decoder, device):
            self.decoder = decoder
            self.device = device
            
        def __call__(self, latent_feature, num_min_values, num_max_values):
            # Ensure all inputs are on the correct device
            latent_feature = latent_feature.to(self.device)
            num_min_values = num_min_values.to(self.device)
            num_max_values = num_max_values.to(self.device)
            return self.decoder(latent_feature, num_min_values, num_max_values)
        
        def to(self, device):
            # Allow moving the decoder to different devices
            self.device = device
            return self

    device_aware_decoder = DeviceAwareDecoder(DS.decoder, device)

    return (device_aware_decoder, latent_features, num_min_values, num_max_values)