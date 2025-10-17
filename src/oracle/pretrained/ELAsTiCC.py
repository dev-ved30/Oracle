import torch

from pathlib import Path
from astropy.table import Table

from oracle.presets import GRU, GRU_MD
from oracle.taxonomies import ORACLE_Taxonomy
from oracle.custom_datasets.ELAsTiCC import *

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

here = Path(__file__).resolve().parent
default_oracle1_elasticc_model_path = str(here.parent.parent.parent / "models" / 'ELAsTiCC' / 'vocal-bird-160')
default_oracle1_elasticc_lite_model_path = str(here.parent.parent.parent / "models" / 'ELAsTiCC-lite' / 'revived-star-159')

def augment_table(table):
    """
    Augments an astronomical observation table by cleaning and transforming its data.
    This function performs several modifications on the input table:
        - Creates a copy of the table to avoid modifying the original.
        - Removes rows where the 'PHOTFLAG' column indicates saturation (using a bitmask with 1024).
        - Reassigns the 'PHOTFLAG' column values:
              * Sets to 1 for detections (when the bitmask with 4096 is non-zero).
              * Sets to 0 for non-detections.
        - Converts band labels in the 'BAND' column to mean wavelengths using the
          'LSST_passband_to_wavelengths' mapping.
        - Normalizes time data by subtracting the Modified Julian Date (MJD) of the first
          detection from all 'MJD' entries.
        - Reorders the columns based on the predefined list 'time_dependent_feature_list'.
        - Iterates over the metadata ('meta') of the table and replaces any values that
          match entries in 'missing_data_flags' with 'flag_value'.

    Parameters:
        table : Astropy Table
            An astropy table containing the columns
            'PHOTFLAG', 'BAND', 'MJD', and a 'meta' attribute. The table is expected to
            adhere to the structure required by the function.
    
    Returns:
        Table-like object
            A new, augmented table with the applied cleaning, conversion, and reordering
            of columns and metadata.
    """

    table = table.copy()

    # Drop the saturations
    saturation_mask =  (np.array(table['PHOTFLAG']) & 1024) == 0 
    table = table[saturation_mask]

    # Change the phot flag to 1 for detections and 0 for non detections
    table['PHOTFLAG'] = np.where(np.array(table['PHOTFLAG']) & 4096 != 0, 1, 0)

    # Convert band labels to mean wavelengths 
    table['BAND'] = [LSST_passband_to_wavelengths[pb] for pb in table['BAND']]

    # Subtract away time of first detection
    table['MJD'] -= table['MJD'][np.where(table['PHOTFLAG']==1)[0][0]]

    # Reorder the columns
    table = table[time_dependent_feature_list]

    for k in table.meta:
        if table.meta[k] in missing_data_flags:
            table.meta[k] = flag_value

    return table

class ORACLE1_ELAsTiCC(GRU_MD):
    """
    ORACLE1_ELAsTiCC is a model class that inherits from GRU_MD designed to load a pre-trained
    ELAsTiCC model and perform predictions on time series data augmented with static features.
    The model uses a hierarchical taxonomy to output predictions at multiple levels of granularity.

    Attributes:
        taxonomy (ORACLE_Taxonomy): An instance of the taxonomy used to structure the class labels.
        ts_feature_dim (int): Dimensionality of the time series input features.
        static_feature_dim (int): Dimensionality of the static input features.
        model_dir (str): Directory path where the model weights are stored.
    """

    def __init__(self, model_dir=default_oracle1_elasticc_model_path):
        """
        Initialize the ELAsTiCC model instance.

        This constructor sets up the model by initializing the taxonomy, time series (TS) feature dimension,
        and static feature dimension. After calling the base class initializer with these parameters,
        it loads the model weights from a specified directory.

        Parameters:
            model_dir (str): Path to the directory containing model weights. Defaults to the
                             pre-defined oracle1 elasticc model path.
        """

        self.taxonomy = ORACLE_Taxonomy()
        self.ts_feature_dim=5
        self.static_feature_dim=18

        super().__init__(taxonomy=self.taxonomy, ts_feature_dim=self.ts_feature_dim, static_feature_dim=self.static_feature_dim)

        self.model_dir = model_dir
        print(f'Loading model weights from {self.model_dir}')
        self.load_state_dict(torch.load(f'{self.model_dir}/best_model_f1.pth', map_location=device), strict=False)

    def make_batch(self, table):
        """
        Create a batch from the input table.

        Parameters:
            table (astropy.table.Table): Input data containing one or more rows.

        Returns:
            dict: A dictionary containing the batch data.
        """
        table = augment_table(table)

        x_ts = np.vstack([table[col] for col in table.colnames]).T.astype(np.float32)
        x_ts = torch.from_numpy(np.expand_dims(x_ts, axis=0))
        
        x_static = []
        for k in time_independent_feature_list:
            x_static.append(table.meta[k])
        x_static = torch.from_numpy(np.expand_dims(x_static, axis=0).astype(np.float32))

        length = torch.from_numpy(np.array([len(table)]).astype(np.float32))

        batch = {
            'ts': x_ts,
            'static': x_static,
            'length': length
        }

        return batch

    def predict_full_scores(self, table):
        """
        Predict class probability scores for a single time-series table.

        Prepares a single observation table for the model by calling augment_table and
        converting time-dependent and time-independent features into torch tensors.
        The input table must contain the columns 'FLUX', 'FLUXERR', 'BAND', 'MJD', and
        'PHOTFLAG', and must provide metadata entries for the keys listed in
        time_independent_feature_list.

        Parameters:
            table (astropy.table.Table): Astropy Table containing time-series data and metadata.

        Returns:
            pd.DataFrame: A DataFrame containing class probability scores for each class in the taxonomy.

        Raises:
            KeyError
                If one or more keys in time_independent_feature_list are missing from table.meta.
            ValueError
                If time-series columns have inconsistent lengths or if the table is empty in a way that the downstream model cannot handle.

        Note:
            - Numeric inputs are converted to numpy.float32 and then to torch tensors.
            - The produced batch has the following keys and tensor shapes:
            - 'ts'     : torch.FloatTensor, shape (1, T, D)
            - 'static' : torch.FloatTensor, shape (1, S)
            - 'length' : torch.FloatTensor, shape (1,)
            - augment_table(table) is called and its return value is used; the original
              table may be replaced by the augmented one.
        """
        batch = self.make_batch(table)
        return self.predict_class_probabilities_df(batch)
    
    def score(self, table):
        """
        Compute hierarchical scores for the input table.
        Predicts scores for all taxonomy nodes using self.predict_full_scores(), then
        groups those scores by taxonomy depth and returns a mapping from depth levels
        to DataFrames containing the corresponding node scores. Taxonomy levels 0
        and 3 are removed because they are irrelevant in the current taxonomy.

        Parameters:
            table (astropy.table.Table): Input observations/features to be scored.

        Returns:
            dict[int, pandas.DataFrame]: A mapping from taxonomy depth level to a
            DataFrame of predicted scores for nodes at that level. Each DataFrame
            is a subset of the full prediction DataFrame containing only the
            columns for the nodes at that depth.

        Raises:
            KeyError: 
                If expected node columns (from self.taxonomy.get_nodes_by_depth())
                are not present in the DataFrame returned by predict_full_scores().

        Note:
            - This is very similar to the model used for the original ORACLE paper.
        """

        scores_by_depth = super().score(table)

        # Remove unused levels
        # print(scores_by_depth.keys())  # --- IGNORE ---
        scores_by_depth.pop(0, None)
        scores_by_depth.pop(3, None)
        
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

        preds_by_depth = super().predict(table)

        # Remove unused levels
        preds_by_depth.pop(0, None)
        preds_by_depth.pop(3, None)

        return preds_by_depth
    
    def embed(self, table):
        """
        Embed a table into its latent space representation.

        Parameters:
            table: The input data (e.g., a table or structured data) to be embedded. The exact format
                   is expected to be compatible with the make_batch method.

        Returns:
            numpy.ndarray: A NumPy array containing the latent space embeddings corresponding to the input table.
        """
        batch = self.make_batch(table)
        return self.get_latent_space_embeddings(batch).detach().numpy()
    

class ORACLE1_ELAsTiCC_lite(GRU):
    """
    Predict class probability scores for a single time-series table.

    Prepares a single observation table for the model by calling augment_table and
    converting time-dependent features into torch tensors.The input table must contain
    the columns 'FLUX', 'FLUXERR', 'BAND', 'MJD', and 'PHOTFLAG'. This model does not
    require additional metadata entries.

    Parameters:
        table (astropy.table.Table): Astropy Table containing time-series data.

    Returns:
        pd.DataFrame: A DataFrame containing class probability scores for each class in the taxonomy.

    Raises:
        ValueError
            If time-series columns have inconsistent lengths or if the table is empty in a way that the downstream model cannot handle.

    Note:
        - This model does not use static features, and can classify using the light curve alone.
        - Numeric inputs are converted to numpy.float32 and then to torch tensors.
        - The produced batch has the following keys and tensor shapes:
        - 'ts'     : torch.FloatTensor, shape (1, T, D)
        - 'length' : torch.FloatTensor, shape (1,)
    """

    def __init__(self, model_dir=default_oracle1_elasticc_lite_model_path):
        """
        Initialize the ELAsTiCC-lite model instance.

        This constructor sets up the model by initializing the taxonomy and time series (TS) feature dimension.
        After calling the base class initializer with these parameters, it loads the model weights from a specified directory.

        Parameters:
            model_dir (str): Path to the directory containing model weights. Defaults to the
                             pre-defined oracle1 elasticc-lite model path.
        """
        self.taxonomy = ORACLE_Taxonomy()
        self.ts_feature_dim=5

        super().__init__(taxonomy=self.taxonomy, ts_feature_dim=self.ts_feature_dim)

        self.model_dir = model_dir
        print(f'Loading model weights from {self.model_dir}')
        self.load_state_dict(torch.load(f'{self.model_dir}/best_model_f1.pth', map_location=device), strict=False)

    def make_batch(self, table):
        """
        Create a batch from the input table.

        Parameters:
            table (astropy.table.Table): Input data containing one or more rows.

        Returns:
            dict: A dictionary containing the batch data.
        """
        table = augment_table(table)

        x_ts = np.vstack([table[col] for col in table.colnames]).T.astype(np.float32)
        x_ts = torch.from_numpy(np.expand_dims(x_ts, axis=0))
    
        length = torch.from_numpy(np.array([len(table)]).astype(np.float32))

        batch = {
            'ts': x_ts,
            'length': length
        }

        return batch

    def predict_full_scores(self, table):
        """
        Predict class probability scores for a single time-series table.

        Prepares a single observation table for the model by calling augment_table and
        converting time-dependent and time-independent features into torch tensors.
        The input table must contain the columns 'FLUX', 'FLUXERR', 'BAND', 'MJD', and
        'PHOTFLAG'.

        Parameters:
            table (astropy.table.Table): Astropy Table containing time-series data.

        Returns:
            pd.DataFrame: A DataFrame containing class probability scores for each class in the taxonomy.

        Raises:
            ValueError
                If time-series columns have inconsistent lengths or if the table is empty in a way that the downstream model cannot handle.

        Note:
            - Numeric inputs are converted to numpy.float32 and then to torch tensors.
            - The produced batch has the following keys and tensor shapes:
            - 'ts'     : torch.FloatTensor, shape (1, T, D)
            - 'length' : torch.FloatTensor, shape (1,)
        """
        batch = self.make_batch(table)
        return self.predict_class_probabilities_df(batch)
    
    def score(self, table):
        """
        Compute hierarchical scores for the input table.
        Predicts scores for all taxonomy nodes using self.predict_full_scores(), then
        groups those scores by taxonomy depth and returns a mapping from depth levels
        to DataFrames containing the corresponding node scores. Taxonomy levels 0
        and 3 are removed because they are irrelevant in the current taxonomy.

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

        Note:
            - This is very similar to the model used for the original ORACLE paper.
        """
        scores_by_depth = super().score(table)

        # Remove unused levels
        scores_by_depth.pop(0, None)
        scores_by_depth.pop(3, None)
        
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
        preds_by_depth = super().predict(table)

        # Remove unused levels
        preds_by_depth.pop(0, None)
        preds_by_depth.pop(3, None)

        return preds_by_depth
    
    def embed(self, table):
        """
        Embed a table into its latent space representation.

        Parameters:
            table: The input data (e.g., a table or structured data) to be embedded. The exact format
                   is expected to be compatible with the make_batch method.

        Returns:
            numpy.ndarray: A NumPy array containing the latent space embeddings corresponding to the input table.
        """
        batch = self.make_batch(table)
        return self.get_latent_space_embeddings(batch).detach().numpy()
    

if __name__=='__main__':

    table = Table.read('notebooks/AGN_17032813.ecsv')

    model = ORACLE1_ELAsTiCC()
    model.score(table)
    model.predict(table)
    print(model.predict(table))

    model = ORACLE1_ELAsTiCC_lite()
    model.score(table)
    model.predict(table)
    print(model.predict(table))