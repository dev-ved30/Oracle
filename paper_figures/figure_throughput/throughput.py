import time 

import matplotlib.pyplot as plt
import numpy as np

from oracle.architectures import GRU_MD_Improved, GRU_MD_MM_Improved, GRU_Improved, ConvNeXt
from oracle.taxonomies import BTS_Taxonomy, ORACLE_Taxonomy

import os
import psutil

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # especially relevant on macOS

import torch
torch.set_num_threads(8)

print(torch.get_num_threads())
print(torch.get_num_interop_threads())

taxonomy = BTS_Taxonomy()
elasticc_taxonomy = ORACLE_Taxonomy()



bts_oracle2 = GRU_MD_Improved(taxonomy)
bts_oracle2_omni = GRU_MD_MM_Improved(taxonomy, image_model_dir=None, lc_md_model_dir=None)
bts_oracle2_lite = GRU_Improved(taxonomy)

elasticc_oracle2 = GRU_MD_Improved(elasticc_taxonomy, static_feature_dim=18)
elasticc_oracle2_lite = GRU_Improved(elasticc_taxonomy)

image_backbone = ConvNeXt(taxonomy)

bts_oracle2_lite.eval()
bts_oracle2.eval()
bts_oracle2_omni.eval()
elasticc_oracle2.eval()
elasticc_oracle2_lite.eval()
image_backbone.eval()

F1_scores_max = {
    "Oracle-2 Lite (BTS)": 0.59,
    "Oracle-2 (BTS)": 0.70,
    "Oracle-2 Omni (BTS)": 0.73,
    "Oracle-2 Lite (ELAsTiCC)": 0.83,
    "Oracle-2 (ELAsTiCC)": 0.88,
}

F1_scores_min = {
    "Oracle-2 Lite (BTS)": 0.19,
    "Oracle-2 (BTS)": 0.49,
    "Oracle-2 Omni (BTS)": 0.59,
    "Oracle-2 Lite (ELAsTiCC)": 0.54,
    "Oracle-2 (ELAsTiCC)": 0.67,
}

models = {
    "Oracle-2 Lite (BTS)": bts_oracle2_lite,
    "Oracle-2 (BTS)": bts_oracle2,
    "Oracle-2 Omni (BTS)": bts_oracle2_omni,
    "Image Backbone (BTS)": image_backbone,
    "Oracle-2 Lite (ELAsTiCC)": elasticc_oracle2_lite,
    "Oracle-2 (ELAsTiCC)": elasticc_oracle2,
}

colors = {
    'Oracle-2 Lite (BTS)': '#994882',
    'Oracle-2 (BTS)': '#008080',
    'Oracle-2 Omni (BTS)': '#FF6645',
    'Oracle-2 Lite (ELAsTiCC)': "crimson",
    'Oracle-2 (ELAsTiCC)': "steelblue",
}

markers = {
    'Oracle-2 Lite (BTS)': 's',
    'Oracle-2 (BTS)': 'D',
    'Oracle-2 Omni (BTS)': 'o',
    'Oracle-2 Lite (ELAsTiCC)': "X",
    'Oracle-2 (ELAsTiCC)': "*",
}

mean_times = {}
std_times = {}
mean_throughput = {}
std_throughput = {}
parameter_counts = {}

for m in models:
    parameter_counts[m] = sum(p.numel() for p in models[m].parameters())


# Benchmarking the throughput of the two models
num_iterations = 100
batch_size = 1


for m in models:

    if 'BTS' in m:
        static_dim = 30
        sequence_length = 41
    else:
        static_dim = 18
        sequence_length = 174

    batch = {
        "ts": torch.randn(batch_size, sequence_length, 5, dtype=torch.float32),  # (batch_size, seq_len, num_features)
        "static": torch.randn(batch_size, static_dim, dtype=torch.float32),
        "length": torch.from_numpy(np.array([sequence_length] * batch_size)),  # (batch_size,)
        "postage_stamp": torch.randn(batch_size, 3, 63, 63, dtype=torch.float32)  # (batch_size, num_channels, height, width)
    }


    model = models[m].eval()

    with torch.inference_mode():

        times = []
        throughput = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            model(batch)
            end_time = time.perf_counter()
            times.append((end_time - start_time))
            throughput.append(1 / (end_time - start_time))
        mean_times[m] = np.mean(times)
        std_times[m] = np.std(times)
        mean_throughput[m] = np.mean(throughput)
        std_throughput[m] = np.std(throughput)

# make a table of the results
import pandas as pd

results_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Mean Inference Time (s)': [mean_times[m] for m in models],
    'Inference Time Std (s)': [std_times[m] for m in models],
    'Mean Throughput (samples/s)': [mean_throughput[m] for m in models],
    'Throughput Std (samples/s)': [std_throughput[m] for m in models],
    'Num Parameters': [parameter_counts[m] for m in models]
})

# make the formatted table with the mean and std in the same column
results_df['Inference Time (s)'] = results_df.apply(lambda row: f"{row['Mean Inference Time (s)']:.4f} ± {row['Inference Time Std (s)']:.3f}", axis=1)
results_df['Throughput (samples/s)'] = results_df.apply(lambda row: f"{row['Mean Throughput (samples/s)']:.2f} ± {row['Throughput Std (samples/s)']:.2f}", axis=1)
results_df.drop(columns=['Mean Inference Time (s)', 'Inference Time Std (s)', 'Mean Throughput (samples/s)', 'Throughput Std (samples/s)'], inplace=True)


print(results_df)

print(results_df.to_latex(index=False))