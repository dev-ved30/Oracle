import torch

from pathlib import Path
from astropy.table import Table

from oracle.presets import GRU, GRU_MD
from oracle.taxonomies import ORACLE_Taxonomy, BTS_Taxonomy
from oracle.custom_datasets.ELAsTiCC import *

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

here = Path(__file__).resolve().parent
default_oracle1_elasticc_model_path = str(here.parent.parent.parent / "models" / 'ELAsTiCC' / 'vocal-bird-160')
default_oracle1_elasticc_lite_model_path = str(here.parent.parent.parent / "models" / 'ELAsTiCC-lite' / 'revived-star-159')

def augment_table(table):

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

    def __init__(self, model_dir=default_oracle1_elasticc_model_path):

        self.taxonomy = ORACLE_Taxonomy()
        self.ts_feature_dim=5
        self.static_feature_dim=18

        super().__init__(taxonomy=self.taxonomy, ts_feature_dim=self.ts_feature_dim, static_feature_dim=self.static_feature_dim)

        self.model_dir = model_dir
        print(f'Loading model weights from {self.model_dir}')
        self.load_state_dict(torch.load(f'{self.model_dir}/best_model_f1.pth', map_location=device), strict=False)

    def predict_full_scores(self, table):

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

        return self.predict_class_probabilities_df(batch)
    
    def score(self, table):

        full_df = self.predict_full_scores(table)
        nodes_by_depth = self.taxonomy.get_nodes_by_depth()

        # Remove unused levels
        nodes_by_depth.pop(0)
        nodes_by_depth.pop(3)

        scores_by_depth = {}

        for level in nodes_by_depth:
            scores_by_depth[level] = full_df[nodes_by_depth[level]]
        
        return scores_by_depth
    
    def predict(self, table):

        scores_by_depth = self.score(table)
        preds_by_depth = {}

        for level in scores_by_depth:
            level_classes = scores_by_depth[level].columns
            preds_by_depth[level] = level_classes[np.argmax(scores_by_depth[level].to_numpy(), axis=1)][0]
        
        return preds_by_depth

class ORACLE1_ELAsTiCC_lite(GRU):

    def __init__(self, model_dir=default_oracle1_elasticc_lite_model_path):

        self.taxonomy = ORACLE_Taxonomy()
        self.ts_feature_dim=5

        super().__init__(taxonomy=self.taxonomy, ts_feature_dim=self.ts_feature_dim)

        self.model_dir = model_dir
        print(f'Loading model weights from {self.model_dir}')
        self.load_state_dict(torch.load(f'{self.model_dir}/best_model_f1.pth', map_location=device), strict=False)

    def predict_full_scores(self, table):

        table = augment_table(table)

        x_ts = np.vstack([table[col] for col in table.colnames]).T.astype(np.float32)
        x_ts = torch.from_numpy(np.expand_dims(x_ts, axis=0))
    
        length = torch.from_numpy(np.array([len(table)]).astype(np.float32))

        batch = {
            'ts': x_ts,
            'length': length
        }

        return self.predict_class_probabilities_df(batch)
    
    def score(self, table):

        full_df = self.predict_full_scores(table)
        nodes_by_depth = self.taxonomy.get_nodes_by_depth()

        # Remove unused levels
        nodes_by_depth.pop(0)
        nodes_by_depth.pop(3)

        scores_by_depth = {}

        for level in nodes_by_depth:
            scores_by_depth[level] = full_df[nodes_by_depth[level]]
        
        return scores_by_depth
    
    def predict(self, table):

        scores_by_depth = self.score(table)
        preds_by_depth = {}

        for level in scores_by_depth:
            level_classes = scores_by_depth[level].columns
            preds_by_depth[level] = level_classes[np.argmax(scores_by_depth[level].to_numpy(), axis=1)][0]
        
        return preds_by_depth

if __name__=='__main__':

    table = Table.read('/Users/vedshah/Documents/Research/NU-Miller/Projects/Hierarchical-VT/notebooks/AGN_17032813.ecsv')

    model = ORACLE1_ELAsTiCC()
    model.score(table)
    model.predict(table)

    model = ORACLE1_ELAsTiCC_lite()
    model.score(table)
    model.predict(table)