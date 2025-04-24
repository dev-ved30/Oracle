import torch

import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision.models import swin_v2_b
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from oracle.taxonomies import Taxonomy, ORACLE_Taxonomy

swin_v2_b_output_dim = 1000

# Template for the Hierarchical Classifier
class Hierarchical_classifier(nn.Module):

    def __init__(self, taxonomy: Taxonomy):
        super(Hierarchical_classifier, self).__init__()

        self.taxonomy = taxonomy
        self.n_nodes = len(taxonomy.get_level_order_traversal())

    def get_conditional_probabilities(self, x):
        
        logits = self.forward(x)
        conditional_probabilities = self.taxonomy.get_conditional_probabilities(logits).detach().numpy()
        return conditional_probabilities

    def get_class_probabilities(self, x):

        conditional_probabilities = self.get_conditional_probabilities(x)
        class_probabilities = self.taxonomy.get_class_probabilities(conditional_probabilities)
        return class_probabilities
    
    def get_conditional_probabilities_df(self, x):

        level_order_nodes = self.taxonomy.get_level_order_traversal()
        conditional_probabilities = self.get_conditional_probabilities(x)
        df = pd.DataFrame(conditional_probabilities, columns=level_order_nodes)
        return df
    
    def get_class_probabilities_df(self, x):

        level_order_nodes = self.taxonomy.get_level_order_traversal()
        class_probabilities = self.get_class_probabilities(x)
        df = pd.DataFrame(class_probabilities, columns=level_order_nodes)
        return df
    
# TODO: Come up with a catchier name for this class.
    
# Base version of the classifier which only uses the Light curve image
class Light_curve_classifier(Hierarchical_classifier):
        
    def __init__(self, config, taxonomy: Taxonomy):

        super(Light_curve_classifier, self).__init__(taxonomy)

        # TODO: Think about what weights we want to initialize the transformer with.
        self.swin = swin_v2_b()

        # Additional layers for classification
        self.fc = nn.Sequential(
            nn.Linear(swin_v2_b_output_dim, config['layer1_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['layer1_dropout']),
            nn.Linear(config['layer1_neurons'], config['layer2_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['layer2_dropout']),
            nn.Linear(config['layer2_neurons'], self.n_nodes),
        )
    
    def forward(self, x):
        
        transformer_output = self.swin(x)
        logits = self.fc(transformer_output)
        return logits
    
# TODO: Think about multi modal implementation.
class Multi_modal_classifier(Hierarchical_classifier):

    def __init__(self, config, taxonomy: Taxonomy):

        super(Multi_modal_classifier, self).__init__(taxonomy)


    def forward(self, x):
        
        raise NotImplementedError
    
class ORACLE_1(Hierarchical_classifier):

    def __init__(self, taxonomy: Taxonomy):

        super(ORACLE_1, self).__init__(taxonomy)

        self.latent_space_dim = 64
        ts_feature_dim = 5
        static_feature_dim = 18

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

    def forward(self, x):

        x_ts = x['ts_data'] # (batch_size, seq_len, n_ts_features)
        x_static = x['static_data'] # (batch_size, n_static_features)
        lengths = x['lengths'] # (batch_size)

        packed = pack_padded_sequence(x_ts, lengths, batch_first=True, enforce_sorted=False)

        # Recurrent backbone
        gru_out, hidden = self.gru(packed)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)

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
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        logits = self.fc_out(x)

        return logits

if __name__ == '__main__':

    taxonomy = ORACLE_Taxonomy()
    config = {
        "layer1_neurons": 512,
        "layer1_dropout": 0.3,
        "layer2_neurons": 128,
        "layer2_dropout": 0.2,
    }
    
    model = Light_curve_classifier(config, taxonomy)
    model.eval()

    x = torch.rand(10, 3, 256, 256)

    print(model.get_conditional_probabilities_df(x))
    print(model.get_class_probabilities_df(x))

    print(model.state_dict())