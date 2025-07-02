import torch

import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision.models import swin_v2_b
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from oracle.trainer import Trainer
from oracle.tester import Tester

swin_v2_b_output_dim = 1000

# Template for the Hierarchical Classifier
class Hierarchical_classifier(nn.Module, Trainer, Tester):

    def __init__(self, output_dim):

        nn.Module.__init__(self)
        self.n_nodes = output_dim

    def predict_conditional_probabilities(self, batch):
        
        logits = self.forward(batch)
        conditional_probabilities = F.softmax(logits).detach()
        print(conditional_probabilities.shape)
        return conditional_probabilities
    
    def predict_conditional_probabilities_df(self, batch):

        level_order_nodes = self.one_hot_encoder.categories_[0]
        conditional_probabilities = self.predict_conditional_probabilities(batch)
        df = pd.DataFrame(conditional_probabilities, columns=level_order_nodes)
        return df
    
    def get_latent_space_embeddings(self, batch):

        raise NotImplementedError
    
# Base version of the classifier which only uses the Light curve image
class ORACLE2_lite_swin(Hierarchical_classifier):
        
    def __init__(self, output_dim):

        super(ORACLE2_lite_swin, self).__init__(output_dim)

        # TODO: Think about what weights we want to initialize the transformer with.
        self.swin = torch.hub.load("pytorch/vision", "swin_v2_t", weights="DEFAULT", progress=False)

        # Make sure all parameters in Swin are trainable
        for param in self.swin.parameters():
            param.requires_grad = True

        # Additional layers for classification
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(64, self.n_nodes),
        )
    
    def forward(self, batch):
        
        swin_output = self.swin(batch['lc_plot'])
        logits = self.fc(swin_output)
        return logits

class ORACLE2_pro_swin(Hierarchical_classifier):

    def __init__(self, output_dim):

        super(ORACLE2_pro_swin, self).__init__(output_dim)


        # TODO: Think about what weights we want to initialize the transformer with.
        self.swin_lc = torch.hub.load("pytorch/vision", "swin_v2_t", weights="DEFAULT", progress=False)

        # Make sure all parameters in Swin are trainable
        for param in self.swin_lc.parameters():
            param.requires_grad = True

        # TODO: Think about what weights we want to initialize the transformer with.
        self.swin_postage = torch.hub.load("pytorch/vision", "swin_v2_t", weights="DEFAULT", progress=False)

        # Make sure all parameters in Swin are trainable
        for param in self.swin_postage.parameters():
            param.requires_grad = True

        # Additional layers for classification
        self.fc = nn.Sequential(
            nn.Linear(2000, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(64, self.n_nodes),
        )

    def forward(self, batch):
        
        swin_lc_output = self.swin_lc(batch['lc_plot'])
        swin_postage_output = self.swin_lc(batch['postage_stamp'])
        combined_output = torch.concat((swin_lc_output, swin_postage_output), dim=1)
        logits = self.fc(combined_output)
        return logits
    
class ORACLE1(Hierarchical_classifier):

    def __init__(self, output_dim,  ts_feature_dim=5, latent_space_dim=64, static_feature_dim=18):

        super(ORACLE1, self).__init__(output_dim)

        self.latent_space_dim = latent_space_dim
        self.ts_feature_dim = ts_feature_dim
        self.static_feature_dim = static_feature_dim

        # recurrent backbone
        self.gru = nn.GRU(input_size=ts_feature_dim, hidden_size=100, num_layers=2, batch_first=True)

        # post‐GRU dense on time‐series path
        self.dense1 = nn.Linear(100, 100)

        # dense on static path
        self.dense2 = nn.Linear(static_feature_dim, 10)

        # merge & head
        self.dense3 = nn.Linear(100 + 10, 100)
        self.dense4 = nn.Linear(100, self.latent_space_dim)

        self.fc_out = nn.Linear(self.latent_space_dim, self.n_nodes)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, batch):

        x_ts = batch['ts'] # (batch_size, seq_len, n_ts_features)
        lengths = batch['length'] # (batch_size)
        x_static = batch['static'] # (batch_size, n_static_features)

        # Pack the padded time series data. the lengths vector lets the GRU know the true lengths of each TS, so it can ignore padding
        packed = pack_padded_sequence(x_ts, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Recurrent backbone
        h0 = torch.zeros(2, x_ts.shape[0], 100).to(x_ts.device)
        _, hidden = self.gru(packed, h0)

        # Take the last output of the GRU
        gru_out = hidden[-1] # (batch_size, hidden_size)

        # Post-GRU dense on time-series path
        dense1 = self.dense1(gru_out)
        dense1 = self.tanh(dense1)

        # Dense on static path
        dense2 = self.dense2(x_static)
        dense2 = self.tanh(dense2)

        # Merge & head
        x = torch.cat((dense1, dense2), dim=1)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        x = self.relu(x)
        logits = self.fc_out(x)

        return logits

class ORACLE1_lite(Hierarchical_classifier):

    def __init__(self, output_dim, ts_feature_dim=5, latent_space_dim=64):

        super(ORACLE1_lite,  self).__init__(output_dim)

        self.latent_space_dim = latent_space_dim
        self.ts_feature_dim = ts_feature_dim

        # recurrent backbone
        self.gru = nn.GRU(input_size=ts_feature_dim, hidden_size=100, num_layers=2, batch_first=True)

        # post‐GRU dense on time‐series path
        self.dense1 = nn.Linear(100, 100)

        # merge & head
        self.dense2 = nn.Linear(100, self.latent_space_dim)

        self.fc_out = nn.Linear(self.latent_space_dim, self.n_nodes)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, batch):

        x_ts = batch['ts'] # (batch_size, seq_len, n_ts_features)
        lengths = batch['length'] # (batch_size)

        # Pack the padded time series data. the lengths vector lets the GRU know the true lengths of each TS, so it can ignore padding
        packed = pack_padded_sequence(x_ts, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Recurrent backbone
        h0 = torch.zeros(2, x_ts.shape[0], 100).to(x_ts.device)
        _, hidden = self.gru(packed, h0)

        # Take the last output of the GRU
        gru_out = hidden[-1] # (batch_size, hidden_size)

        # Post-GRU dense on time-series path
        dense1 = self.dense1(gru_out)
        dense1 = self.tanh(dense1)

        # Merge & head
        x = self.dense2(dense1)
        x = self.relu(x)
        logits = self.fc_out(x)

        return logits

if __name__ == '__main__':

    config = {
        "layer1_neurons": 512,
        "layer1_dropout": 0.3,
        "layer2_neurons": 128,
        "layer2_dropout": 0.2,
    }
    
    model = ORACLE2_lite_swin(config, 6)
    model.eval()

    x = torch.rand(10, 3, 256, 256)

    print(model.predict_conditional_probabilities_df(x))
    print(model.predict_class_probabilities_df(x))

    print(model.state_dict())