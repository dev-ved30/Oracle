import torch

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

if __name__ == '__main__':

    taxonomy = ORACLE_Taxonomy()
    
    model = GRU(taxonomy)
    model.eval()

    x = {
        'ts': torch.rand(10, 256, 5),
        'length': torch.from_numpy(np.array([256]*10))
    }

    print(model.predict_conditional_probabilities_df(x))
    print(model.predict_class_probabilities_df(x))

    print(model.state_dict())