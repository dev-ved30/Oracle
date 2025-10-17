import torch

from pathlib import Path
from astropy.table import Table

from oracle.presets import GRU, GRU_MD
from oracle.taxonomies import BTS_Taxonomy
from oracle.custom_datasets.BTS import *

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

here = Path(__file__).resolve().parent
default_oracle1_BTS_model_path = str(here.parent.parent.parent / "models" / 'BTS' / 'lemon-spaceship-252')


def augment_table(table):

    table = table.copy()

    # Convert filter ids to mean wavelengths 
    table['fid'] = [ZTF_fid_to_wavelengths[pb] for pb in table['fid']]

    # Subtract away time of first detection
    table['jd'] -= min(table['jd'])

    # Reorder the columns
    lc_table = table[time_dependent_feature_list]
    lc_table['flag'] = 1

    # Get the final time-independent features
    static_table = table[time_independent_feature_list]

    return lc_table, static_table

class ORACLE1_BTS(GRU_MD):

    def __init__(self, model_dir=default_oracle1_BTS_model_path):

        self.taxonomy = BTS_Taxonomy()
        self.ts_feature_dim=5
        self.static_feature_dim=30

        super().__init__(taxonomy=self.taxonomy, ts_feature_dim=self.ts_feature_dim, static_feature_dim=self.static_feature_dim)

        self.model_dir = model_dir
        print(f'Loading model weights from {self.model_dir}')
        self.load_state_dict(torch.load(f'{self.model_dir}/best_model_f1.pth', map_location=device), strict=False)

    def predict_full_scores(self, table):

        ts_table, static_table = augment_table(table)

        x_ts = np.vstack([ts_table[col] for col in ts_table.colnames]).T.astype(np.float32)
        x_ts = torch.from_numpy(np.expand_dims(x_ts, axis=0))
        
        x_static = static_table.to_pandas().to_numpy().astype(np.float32)[-1, :]
        x_static = torch.from_numpy(np.expand_dims(x_static, axis=0).astype(np.float32))

        md_list = []
        for f in meta_data_feature_list:
            md_list.append(ts_table.meta[f])
        md_list = torch.from_numpy(np.expand_dims(md_list, axis=0).astype(np.float32))

        x_static = torch.cat([x_static, md_list], dim=1)        
        length = torch.from_numpy(np.array([len(table)]).astype(np.float32))

        batch = {
            'ts': x_ts,
            'static': x_static,
            'length': length
        }

        return self.predict_class_probabilities_df(batch)
    
    def score(self, table):
        """
        Compute hierarchical scores for the input table.
        Predicts scores for all taxonomy nodes using self.predict_full_scores(), then
        groups those scores by taxonomy depth and returns a mapping from depth levels
        to DataFrames containing the corresponding node scores. Taxonomy levels 0 are 
        removed because they are irrelevant in the current taxonomy.

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
        scores_by_depth.pop(0, None)
        scores_by_depth.pop(2, None)
        
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
        preds_by_depth.pop(2, None)

        return preds_by_depth

if __name__=='__main__':

    table = Table.read('notebooks/fake_SN.ecsv')

    model = ORACLE1_BTS()
    print(model.predict(table))
    print(model.score(table))
