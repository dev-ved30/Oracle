import time 
import torch
import numpy as np
import pandas as pd

from oracle.architectures import GRU_MD_Improved, GRU_MD_MM_Improved, GRU_Improved, ConvNeXt
from oracle.taxonomies import BTS_Taxonomy, ORACLE_Taxonomy


def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark model throughput")
    parser.add_argument("model_choice", type=str, help="Model to benchmark")
    return parser.parse_args()


def main(model_choice):

    taxonomy = BTS_Taxonomy()
    elasticc_taxonomy = ORACLE_Taxonomy()
    
    if model_choice == "bts_oracle2":
        model = GRU_MD_Improved(taxonomy)
    elif model_choice == "bts_oracle2_omni":
        model = GRU_MD_MM_Improved(taxonomy, image_model_dir=None, lc_md_model_dir=None)
    elif model_choice == "bts_oracle2_lite":
        model = GRU_Improved(taxonomy)
    elif model_choice == "elasticc_oracle2":
        model = GRU_MD_Improved(elasticc_taxonomy, static_feature_dim=18)
    elif model_choice == "elasticc_oracle2_lite":
        model = GRU_Improved(elasticc_taxonomy)
    elif model_choice == "image_backbone":
        model = ConvNeXt(taxonomy)
    else:
        raise ValueError("Invalid model choice")
    
    model.to('cpu')
    model.eval()

    models = {
        model_choice: model,
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

        if 'bts' in m:
            static_dim = 30
            sequence_length = 41
        else:
            static_dim = 18
            sequence_length = 174

        batch = {
            "ts": torch.randn(batch_size, sequence_length, 5, dtype=torch.float32).to('cpu'),  # (batch_size, seq_len, num_features)
            "static": torch.randn(batch_size, static_dim, dtype=torch.float32).to('cpu'),
            "length": torch.from_numpy(np.array([sequence_length] * batch_size)).to('cpu'),  # (batch_size,)
        }

        if m == "image_backbone" or m == "bts_oracle2_omni":
            batch["postage_stamp"] = torch.randn(batch_size, 3, 63, 63, dtype=torch.float32).to('cpu')  # (batch_size, num_channels, height, width)


        model = models[m].eval()

        with torch.inference_mode():

            times = []
            throughput = []
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                model(batch)
                end_time = time.perf_counter()
                times.append((end_time - start_time))
                throughput.append(batch_size / (end_time - start_time))
            mean_times[m] = np.mean(times)
            std_times[m] = np.std(times)
            mean_throughput[m] = np.mean(throughput)
            std_throughput[m] = np.std(throughput)

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

if __name__ == "__main__":

    args = argparse()
    main(args.model_choice)
    