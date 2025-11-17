"""
Top-level module for defining various neural network architectures for hierarchical classification.
"""
import torch
import timm

import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision.models import swin_v2_b
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from oracle.taxonomies import Taxonomy, ORACLE_Taxonomy
from oracle.trainer import Trainer
from oracle.tester import Tester

swin_v2_b_output_dim = 1000

# Template for the Hierarchical Classifier
class Hierarchical_classifier(nn.Module, Trainer, Tester):
    """
    Base class for hierarchical classification architectures."""

    def __init__(self, taxonomy: Taxonomy):
        """
        Initializes an instance of the architecture.
        Args:
            taxonomy (Taxonomy): An instance of the Taxonomy class representing the hierarchical taxonomy structure.
        Attributes:
            taxonomy (Taxonomy): The taxonomy instance provided as an argument.
            n_nodes (int): The number of nodes obtained by performing a level order traversal on the taxonomy.
        """

        nn.Module.__init__(self)
        self.taxonomy = taxonomy
        self.n_nodes = len(taxonomy.get_level_order_traversal())

    def predict_conditional_probabilities(self, batch):
        """
        Compute conditional probabilities from the model's output.
        This method performs a forward pass using the given batch of inputs to obtain the logits.
        It then utilizes the taxonomy's get_conditional_probabilities method to convert these logits
        into conditional probabilities. The resulting tensor is detached from the computation graph
        and returned.

        Parameters:
            batch (dictionary): A batch of input data to be fed into the model. The exact format and type of the batch
                depend on the requirements of the forward method.

        Returns:
            torch.tensor: A tensor representing the conditional probabilities computed from the model's logits.
        """
        
        logits = self.forward(batch)
        conditional_probabilities = self.taxonomy.get_conditional_probabilities(logits).detach()
        return conditional_probabilities

    def predict_class_probabilities(self, batch):
        """
        Predicts the class probabilities for a given batch of data.
        This method first computes the conditional probabilities for the batch using
        the 'predict_conditional_probabilities' method. It then leverages the taxonomy to
        convert these conditional probabilities into final class probabilities.

        Parameters:
            batch (dictionary): The input data batch for which class probabilities are to be predicted.

        Returns:
            torch.tensor: The computed class probabilities derived from the input batch.
        """

        conditional_probabilities = self.predict_conditional_probabilities(batch)
        class_probabilities = self.taxonomy.get_class_probabilities(conditional_probabilities)
        return class_probabilities
    
    def predict_conditional_probabilities_df(self, batch):
        """
        Predict conditional probabilities for a batch and return them as a pandas DataFrame.

        This method retrieves a level-order traversal of nodes from the taxonomy, computes the conditional
        probabilities for the given batch using the predict_conditional_probabilities method, and then
        constructs a pandas DataFrame with the probabilities. The DataFrame's columns are named using the
        node order from the taxonomy's level-order traversal.

        Parameters:
            batch (dictionary): The input data batch for which the conditional probabilities are to be predicted.

        Returns:
            pandas.DataFrame: A DataFrame containing the predicted conditional probabilities with columns
            corresponding to the level-order nodes.
        """

        level_order_nodes = self.taxonomy.get_level_order_traversal()
        conditional_probabilities = self.predict_conditional_probabilities(batch)
        df = pd.DataFrame(conditional_probabilities, columns=level_order_nodes)
        return df
    
    def predict_class_probabilities_df(self, batch):
        """
        Predict class probabilities for a given batch and return the results in a DataFrame.

        This method performs the following steps:
        1. Retrieves the list of taxonomy nodes ordered by level using the taxonomy's `get_level_order_traversal` method.
        2. Computes the class probabilities for the input batch via the `predict_class_probabilities` method.
        3. Constructs and returns a pandas DataFrame with the computed probabilities, where each column corresponds to a taxonomy node in level order.

        Parameters:
             batch (dictionary): The input data batch on which to perform predictions. The expected format
                and type of `batch` depend on the implementation details of the prediction model.

        Returns:
             pandas.DataFrame: A DataFrame with columns representing taxonomy nodes and each cell containing
            the predicted class probability for the corresponding node.
        """

        level_order_nodes = self.taxonomy.get_level_order_traversal()
        class_probabilities = self.predict_class_probabilities(batch)
        df = pd.DataFrame(class_probabilities, columns=level_order_nodes)
        return df
    
    def get_latent_space_embeddings(self, batch):
        """
        Compute the latent space embeddings for a given batch of data.

        This method should be implemented by subclasses to transform the input batch into a latent representation.
        The exact nature of the embedding (e.g., dimensionality, transformation mechanism) depends on the specific architecture.

        Parameters:
            batch (dictionary): A batch of input data on which to compute the latent embeddings. The expected format and type should
                be defined by the implementing subclass.

        Returns:
            Any: The latent space embeddings corresponding to the provided batch. The structure and type of the embeddings
            is determined by the subclass implementation.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError("get_latent_space_embeddings function not implemented. This is only available for specific architectures.")
    
    def predict_full_scores(self, table):

        raise NotImplementedError("predict_full_scores function not implemented. This is only available for pretrained models.")

    def embed(self, table):
        """
        Embeds the provided table into the model's latent space.
        Parameters:
            table : object
                The table data to be embedded. The expected type and structure of this data
                should be compatible with the model’s input requirements.
        Raises:
            NotImplementedError:
                This method is not implemented by default and is intended for pretrained models only.
        """
        
        raise NotImplementedError("embed function not implemented. This is only available for pretrained models.")

    def score(self, table):
        """
        Compute hierarchical scores for the input table.
        Predicts scores for all taxonomy nodes using self.predict_full_scores(), then
        groups those scores by taxonomy depth and returns a mapping from depth levels
        to DataFrames containing the corresponding node scores. 

        Parameters:
            table (astropy.table.Table): Input observations/features to be scored.

        Returns:
            dict[int, pandas.DataFrame]: A mapping from taxonomy depth level to a
                DataFrame of predicted scores for nodes at that level. Each DataFrame
                is a subset of the full prediction DataFrame containing only the
                columns for the nodes at that depth.

        Raises:
            KeyError: 
                If expected node columns (from self.taxonomy.get_nodes_by_depth()) are not present in the DataFrame returned by predict_full_scores().
        """
        full_df = self.predict_full_scores(table)
        nodes_by_depth = self.taxonomy.get_nodes_by_depth()
        scores_by_depth = {}

        for level in nodes_by_depth:
            scores_by_depth[level] = full_df[nodes_by_depth[level]]
        
        return scores_by_depth
    
    def predict(self, table):
        """
        Predict the label at each hierarchical level for the table.

        Parameters:
            table (astropy.table.Table): Input data containing one or more rows.

        Returns:
            dict: Mapping from hierarchical level (as returned by self.score) to the predicted class
                label. For each level, self.score(table) is expected to return a
                pandas.DataFrame of shape (n_samples, n_classes) with class labels as columns; the
                predicted label is the column with the highest score for the first sample.

        Raises:
            Any exceptions raised by self.score or by numpy operations (e.g., if the score DataFrame is empty) will be propagated.
        """
        scores_by_depth = self.score(table)
        preds_by_depth = {}

        for level in scores_by_depth:
            level_classes = scores_by_depth[level].columns
            preds_by_depth[level] = level_classes[np.argmax(scores_by_depth[level].to_numpy(), axis=1)][0]
        
        return preds_by_depth
    
class GRU(Hierarchical_classifier):
    """GRU-based neural network architecture for hierarchical classification."""

    def __init__(self, taxonomy: Taxonomy, ts_feature_dim=5):
        """
        Initialize the GRU-based neural network architecture.

        This constructor sets up the neural network by initializing its recurrent backbone,
        dense layers, merge-head, and activation functions.

        Parameters:
            taxonomy (Taxonomy): An instance of the Taxonomy class used for defining the network's structure.
            ts_feature_dim (int, optional): The dimensionality of the input time-series features. Defaults to 5.

        Attributes:
            ts_feature_dim (int): Stores the time-series feature dimension.
            output_dim (int): The output dimension, derived from the number of nodes (self.n_nodes).
            gru (nn.GRU): The GRU layer that serves as the recurrent backbone, configured with a hidden size of 100,
                           2 layers, and batch_first=True.
            dense1 (nn.Linear): A fully-connected layer mapping from 100 units (GRU output) to 64 units.
            dense2 (nn.Linear): A fully-connected layer mapping from 64 to 32 units for merging features.
            dense3 (nn.Linear): A fully-connected layer mapping from 32 to 16 units for further dimensionality reduction.
            fc_out (nn.Linear): The final output layer mapping from 16 units to the output dimension.
            tanh (nn.Tanh): The hyperbolic tangent activation function.
            relu (nn.ReLU): The ReLU activation function.
        """

        super(GRU, self).__init__(taxonomy)

        self.ts_feature_dim = ts_feature_dim
        self.output_dim = self.n_nodes

        # recurrent backbone
        self.gru = nn.GRU(input_size=ts_feature_dim, hidden_size=100, num_layers=2, batch_first=True)

        # post‐GRU dense on time‐series path
        self.dense1 = nn.Linear(100, 64)

        # merge & head
        self.dense2 = nn.Linear(64, 32)

        self.dense3 = nn.Linear(32, 16)

        self.fc_out = nn.Linear(16, self.output_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def get_latent_space_embeddings(self, batch):
        """
        Compute latent space embeddings for a batch of time-series data.

        Parameters:
            batch (dict): A dictionary containing:
                1. 'ts' (torch.Tensor): Input time-series data of shape (batch_size, seq_len, n_ts_features).
                2.'length' (torch.Tensor or list[int]): Sequence lengths indicating the valid lengths of each time-series in the batch.

        Returns:
            torch.Tensor: The latent space embeddings resulting from the network's forward pass.
        """
        
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
        dense2 = self.dense2(dense1)
        dense2 = self.tanh(dense2)

        x = self.dense3(dense2)
        return x

    def forward(self, batch):
        """
        Compute the logits for a given input batch through the model.

        This method processes the input batch by first obtaining its latent space embeddings, 
        applying a ReLU activation, and then projecting the result to logits using a fully 
        connected output layer.

        Parameters:
            batch(dict): Input data batch. 
            
        Returns:
            logits(torch.Tensor): The computed output logits after the final linear transformation.
        """

        # Get the latent space embedding
        x = self.get_latent_space_embeddings(batch)

        # Final step to produce logits
        x = self.relu(x)
        logits = self.fc_out(x)

        return logits
    
class GRU_MD(Hierarchical_classifier):
    """GRU-based neural network architecture with multi-dimensional static features for hierarchical classification."""

    def __init__(self, taxonomy: Taxonomy, ts_feature_dim=5, static_feature_dim=30):
        """
        Initialize the GRU_MD architecture.

        Args:
            taxonomy (Taxonomy): An instance representing the hierarchical taxonomy. It is used to determine the number
                of output nodes.
            ts_feature_dim (int, optional): The dimensionality of the time-series input features. Defaults to 5.
            static_feature_dim (int, optional): The dimensionality of the static input features. Defaults to 30.

        Attributes:
            ts_feature_dim (int): The time-series feature dimension.
            static_feature_dim (int): The static feature dimension.
            output_dim (int): The number of nodes (i.e., the output dimensionality), derived from the taxonomy.
            gru (nn.GRU): A GRU layer that processes the time-series data with an input size of ts_feature_dim, a hidden size of 100,
                2 layers, and batch-first setting enabled.
            dense1 (nn.Linear): A linear layer that processes the output of the GRU.
            dense2 (nn.Linear): A linear layer to process the static input features.
            dense3 (nn.Linear): A linear layer that merges the outputs from the GRU path and the static feature path.
            dense4 (nn.Linear): The first dense layer in the post-merge head, reducing dimensionality.
            dense5 (nn.Linear): The second dense layer in the head.
            dense6 (nn.Linear): The third dense layer in the head, further reducing dimensionality.
            fc_out (nn.Linear): The final fully connected layer that outputs a tensor of shape corresponding to output_dim.
            tanh (nn.Tanh): Tanh activation function.
            relu (nn.ReLU): ReLU activation function.
        """

        super(GRU_MD, self).__init__(taxonomy)

        self.ts_feature_dim = ts_feature_dim
        self.static_feature_dim = static_feature_dim
        self.output_dim = self.n_nodes

        # recurrent backbone
        self.gru = nn.GRU(input_size=ts_feature_dim, hidden_size=100, num_layers=2, batch_first=True)

        # post‐GRU dense on time‐series path
        self.dense1 = nn.Linear(100, 100)

        # dense on static path
        self.dense2 = nn.Linear(static_feature_dim, 30)

        # merge & head
        self.dense3 = nn.Linear(100 + 30, 100)
        self.dense4 = nn.Linear(100, 64)
        self.dense5 = nn.Linear(64, 32)
        self.dense6 = nn.Linear(32, 16)

        self.fc_out = nn.Linear(16, self.output_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    
    def get_latent_space_embeddings(self, batch):
        """
        Generates the latent space embedding for a given batch of data by processing both time series and static features.

        Parameters:
            batch (dict): A dictionary containing the following keys:
                1. 'ts' (torch.Tensor): Time series data of shape (batch_size, seq_len, n_ts_features). 
                2. 'length' (torch.Tensor): Tensor containing the true lengths for each time series in the batch (shape: (batch_size)).
                3. 'static' (torch.Tensor): Static features of shape (batch_size, n_static_features).

        Returns:
            torch.Tensor: The latent space embedding computed by merging the processed time series and static features after passing them through respective dense layers and nonlinear activations. 
        """

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
        x = self.tanh(x)

        x = self.dense5(x)
        x = self.tanh(x)

        x = self.dense6(x)

        return x
    
    def forward(self, batch):
        """
        Computes the forward pass through the network module.
        This method performs the following steps:
            - Extracts latent space embeddings from the input batch using self.get_latent_space_embeddings.
            - Applies a ReLU activation to the latent embeddings.
            - Computes the logits by passing the activated embeddings through a fully connected layer (self.fc_out).

        Parameters:
            batch: Input data (e.g., tensors or other data structures) used to compute the latent embeddings.
            
        Returns:
            Tensor: Logits produced by the network.
        """
        
        # Get the latent space embedding
        x = self.get_latent_space_embeddings(batch)

        # Final step to produce logits
        x = self.relu(x)
        logits = self.fc_out(x)

        return logits
    
class GRU_MD_MM(Hierarchical_classifier):
    """GRU-based neural network architecture with multi-dimensional static features and multi-modal inputs for hierarchical classification."""
    
    def __init__(self, output_dim, ts_feature_dim=5, static_feature_dim=18):
        """
        Initialize the GRU_MD_MM model architecture.

        This method sets up the network components including:
            - A Swin transformer backbone for feature extraction with all parameters set to trainable.
            - A GRU-based recurrent module for processing time series data.
            - Dense layers for post-GRU processing of time series features.
            - A dense layer for processing static features.
            - A series of fully connected layers to merge the extracted features from the time series, static, and Swin transformer paths.
            - A final linear layer to output the predictions with a dimension specified by output_dim.
            - Activation functions (ReLU and Tanh) to introduce non-linearity in the network.

        Parameters:
            output_dim (int): The dimension of the final output.
            ts_feature_dim (int, optional): The dimension of the time series features per timestep. Defaults to 5.
            static_feature_dim (int, optional): The dimension of the static features. Defaults to 18.
        """

        super(GRU_MD_MM, self).__init__()

        self.ts_feature_dim = ts_feature_dim
        self.static_feature_dim = static_feature_dim
        self.output_dim = output_dim

        # TODO: Think about what weights we want to initialize the transformer with.
        self.swin_postage = torch.hub.load("pytorch/vision", "swin_v2_t", weights="DEFAULT", progress=False)

        # Make sure all parameters in Swin are trainable
        for param in self.swin_postage.parameters():
            param.requires_grad = True

        # recurrent backbone
        self.gru = nn.GRU(input_size=ts_feature_dim, hidden_size=100, num_layers=2, batch_first=True)

        # post‐GRU dense on time‐series path
        self.dense1 = nn.Linear(100, 100)

        # dense on static path
        self.dense2 = nn.Linear(static_feature_dim, 10)

        self.swin_fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
        )

        # merge & head
        self.dense3 = nn.Linear(100 + 10 + 64, 100)
        self.dense4 = nn.Linear(100, 64)
        self.dense5 = nn.Linear(64, 32)
        self.dense6 = nn.Linear(32, 16)

        self.fc_out = nn.Linear(64, self.output_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    
    def get_latent_space_embeddings(self, batch):
        """
        Compute latent space embeddings from the given batch of inputs.

        Processes time-series, static, and postage stamp inputs to produce a latent representation.
        The function performs the following steps:
            - Packs the padded time-series data using the sequence lengths provided.
            - Processes the time-series data with a bidirectional GRU and extracts its last hidden state.
            - Processes the postage stamp data through a Swin-based module and a fully connected layer.
            - Transforms the GRU and static features through separate dense layers with non-linear activation.
            - Concatenates the processed time-series, static, and postage stamp features.
            - Passes the merged representation through additional dense layers with activation functions to yield the final latent space embedding.

        Parameters:
            batch (dict): A dictionary containing:
                1. "ts" (torch.Tensor): Time series data of shape (batch_size, seq_len, n_ts_features).
                2. "length" (torch.Tensor): Lengths of each time series in the batch (batch_size, ).
                3. "static" (torch.Tensor): Static features of shape (batch_size, n_static_features).
                4. "postage_stamp" (torch.Tensor): Postage stamp data for additional processing.

        Returns:
            torch.Tensor: The latent space embeddings computed from the combined features.
        """

        x_ts = batch['ts'] # (batch_size, seq_len, n_ts_features)
        lengths = batch['length'] # (batch_size)
        x_static = batch['static'] # (batch_size, n_static_features)
        x_postage = batch['postage_stamp']

        # Pack the padded time series data. the lengths vector lets the GRU know the true lengths of each TS, so it can ignore padding
        packed = pack_padded_sequence(x_ts, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Recurrent backbone
        h0 = torch.zeros(2, x_ts.shape[0], 100).to(x_ts.device)
        _, hidden = self.gru(packed, h0)

        # Take the last output of the GRU
        gru_out = hidden[-1] # (batch_size, hidden_size)

        swin_out = self.swin_postage(x_postage)
        swin_fc_out = self.swin_fc(swin_out)

        # Post-GRU dense on time-series path
        dense1 = self.dense1(gru_out)
        dense1 = self.tanh(dense1)

        # Dense on static path
        dense2 = self.dense2(x_static)
        dense2 = self.tanh(dense2)

        # Merge & head
        x = torch.cat((dense1, dense2, swin_fc_out), dim=1)
        x = self.dense3(x)
        x = self.relu(x)

        x = self.dense4(x)
        x = self.tanh(x)

        x = self.dense5(x)
        x = self.tanh(x)

        x = self.dense6(x)

        return x
    
    def forward(self, batch):
        """
        Compute the network's output logits from an input batch.
        This method first extracts latent space embeddings from the batch, applies a ReLU activation to introduce non-linearity, and then computes the final logits using a fully connected output layer.
        
        Parameters:
            batch: The input data batch from which latent embeddings are derived.
        
        Returns:
            torch.Tensor: The computed logits after processing the latent embeddings through the ReLU activation and linear layer.
        """
        
        # Get the latent space embedding
        x = self.get_latent_space_embeddings(batch)

        # Final step to produce logits
        x = self.relu(x)
        logits = self.fc_out(x)

        return logits
    
class MaxViT(Hierarchical_classifier):

    def __init__(self, taxonomy: Taxonomy):

        super(MaxViT, self).__init__(taxonomy)

        self.output_dim = self.n_nodes

        model_kind = "maxvit_tiny_rw_224.sw_in1k"
        self.image_size = 224
        self.maxvit = timm.create_model(model_kind, pretrained=True)

        self.maxvit.head = nn.Sequential(
            self.maxvit.head.global_pool,
            nn.Linear(self.maxvit.head.in_features, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.final_out = nn.Sequential(
            nn.GELU(),
            nn.Linear(128, self.output_dim),
        )


    def get_latent_space_embeddings(self, batch):

        input_data = batch['postage_stamp']

        if input_data.shape[-1] != self.image_size or input_data.shape[-2] != self.image_size:
            input_data = torch.nn.functional.interpolate(
                input_data,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        return self.maxvit(input_data)

    def forward(self, batch) -> torch.Tensor:

        # Get the latent space embedding
        x = self.get_latent_space_embeddings(batch)

        # Final step to produce logits
        logits = self.final_out(x)

        return logits
    
class GRU_MD_Improved(Hierarchical_classifier):
    """
    Improved GRU-based neural network architecture with multi-dimensional static features for hierarchical classification.

    Key differences vs original:
        - Bidirectional GRU with attention pooling over time (gives richer sequence summary).
        - LayerNorm / BatchNorm and Dropout for regularisation & stability.
        - Residual connections in the MLP head.
        - Configurable hidden sizes and dropout.
        - GELU activations in the head for smoother gradients.
    """
    def __init__(self,
                 taxonomy: Taxonomy,
                 ts_feature_dim: int = 5,
                 static_feature_dim: int = 30,
                 gru_hidden: int = 128,
                 gru_layers: int = 2,
                 dropout: float = 0.2):
        """
        Args:
            taxonomy (Taxonomy): hierarchical taxonomy (used to determine output dim via self.n_nodes).
            ts_feature_dim (int): dimensionality of time-series features.
            static_feature_dim (int): dimensionality of static features.
            gru_hidden (int): hidden size of the (per-direction) GRU.
            gru_layers (int): number of GRU layers.
            dropout (float): dropout probability for regularisation.
        """
        super(GRU_MD_Improved, self).__init__(taxonomy)

        self.ts_feature_dim = ts_feature_dim
        self.static_feature_dim = static_feature_dim
        self.output_dim = self.n_nodes

        # recurrent backbone: bidirectional for richer encoding
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.num_directions = 2  # bidirectional
        self.gru = nn.GRU(input_size=ts_feature_dim,
                          hidden_size=gru_hidden,
                          num_layers=gru_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)

        # attention pooling on top of per-timestep outputs
        # attention: score = v^T tanh(W h_t + b)
        self.attn_W = nn.Linear(gru_hidden * self.num_directions, gru_hidden, bias=True)
        self.attn_v = nn.Linear(gru_hidden, 1, bias=False)

        # post-GRU dense on time-series path
        self.ts_proj = nn.Linear(gru_hidden * self.num_directions, 128)
        self.ts_ln = nn.LayerNorm(128)
        self.ts_dropout = nn.Dropout(dropout)

        # dense on static path
        self.static_proj = nn.Linear(static_feature_dim, 64)
        self.static_bn = nn.BatchNorm1d(64)
        self.static_dropout = nn.Dropout(dropout)

        # merge & head with residual blocks
        merge_in = 128 + 64
        self.merge_proj = nn.Linear(merge_in, 128)

        # head (residual MLP blocks)
        self.head_fc1 = nn.Linear(128, 128)
        self.head_fc2 = nn.Linear(128, 64)
        self.head_fc3 = nn.Linear(64, 32)

        # small bottleneck before output
        self.latent_proj = nn.Linear(32, 16)

        self.fc_out = nn.Linear(16, self.output_dim)

        # activations
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        # dropout and layernorm in head
        self.head_dropout = nn.Dropout(dropout)
        self.head_ln1 = nn.LayerNorm(128)
        self.head_ln2 = nn.LayerNorm(64)

        # init weights
        self._init_weights()

    def _init_weights(self):
        """Simple weight initialization to help training stability."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def _attention_pool(self, packed_outputs, lengths):
        """
        Attention pooling over time.
        packed_outputs: output from GRU (PackedSequence)
        lengths: tensor of true lengths (batch,)
        returns: (batch, hidden_size * num_directions)
        """
        from torch.nn.utils.rnn import pad_packed_sequence
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)  # (batch, max_seq, feat)
        # compute attention scores
        attn_hidden = torch.tanh(self.attn_W(outputs))  # (batch, max_seq, gru_hidden)
        attn_scores = self.attn_v(attn_hidden).squeeze(-1)  # (batch, max_seq)
        # mask padding positions
        max_len = outputs.size(1)
        device = outputs.device
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)  # True for padding
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, max_seq, 1)
        context = torch.sum(attn_weights * outputs, dim=1)  # (batch, feat)
        return context

    def get_latent_space_embeddings(self, batch):
        """
        Generates the latent embedding for a batch (time-series + static).
        batch keys:
            'ts'     -> (batch, seq_len, n_ts_features)
            'length' -> (batch,)
            'static' -> (batch, n_static_features)
        Returns:
            torch.Tensor of shape (batch, 16) -- the latent representation before final classifier.
        """
        x_ts = batch['ts']                  # (batch, seq_len, ts_features)
        lengths = batch['length']           # (batch,)
        x_static = batch['static']          # (batch, static_features)

        # pack padded sequences so GRU ignores padding
        packed = pack_padded_sequence(x_ts, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # initialize h0 properly: (num_layers * num_directions, batch, hidden_size)
        batch_size = x_ts.shape[0]
        h0 = torch.zeros(self.gru_layers * self.num_directions, batch_size, self.gru_hidden, device=x_ts.device)

        # GRU returns PackedSequence for outputs when input is packed
        packed_outputs, hidden = self.gru(packed, h0)

        # attention pooling over time using the packed_outputs
        seq_repr = self._attention_pool(packed_outputs, lengths)  # (batch, gru_hidden * num_directions)

        # ts path projection
        ts = self.ts_proj(seq_repr)
        ts = self.ts_ln(ts)
        ts = self.gelu(ts)
        ts = self.ts_dropout(ts)

        # static path
        static = self.static_proj(x_static)  # (batch, 64)
        # batchnorm expects (batch, features). If batch==1, BN behaves strangely; keep as-is.
        static = self.static_bn(static)
        static = self.relu(static)
        static = self.static_dropout(static)

        # merge
        x = torch.cat((ts, static), dim=1)
        x = self.merge_proj(x)
        x = self.gelu(x)

        # head with residual connection
        residual = x
        x = self.head_fc1(x)
        x = self.head_ln1(x)
        x = self.gelu(x)
        x = self.head_dropout(x)

        x = self.head_fc2(x)
        x = self.head_ln2(x)
        x = self.gelu(x)
        x = self.head_dropout(x)

        # add residual (project if needed)
        if residual.size(1) == x.size(1):
            x = x + residual
        else:
            # project residual to match dim
            proj_res = nn.Linear(residual.size(1), x.size(1)).to(x.device)
            x = x + proj_res(residual)

        x = self.head_fc3(x)
        x = self.gelu(x)

        # final latent projection
        latent = self.latent_proj(x)  # (batch, 16)

        return latent

    def forward(self, batch):
        """
        Forward pass: get latent embedding and compute logits.
        """
        x = self.get_latent_space_embeddings(batch)
        x = self.relu(x)
        logits = self.fc_out(x)
        return logits


if __name__ == '__main__':

    batch_size = 10

    taxonomy = ORACLE_Taxonomy()
    
    model = GRU(taxonomy)
    model.eval()

    x = {
        'ts': torch.rand(batch_size, 256, 5),
        'length': torch.from_numpy(np.array([256]*batch_size)),
        'postage_stamp': torch.rand(batch_size, 3, 252, 252)
    }

    print(model.predict_conditional_probabilities_df(x))
    print(model.predict_class_probabilities_df(x))

    model = MaxViT(taxonomy)
    print(model.predict_conditional_probabilities_df(x))
    print(model.predict_class_probabilities_df(x))