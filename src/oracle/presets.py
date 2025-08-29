import torch

from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from oracle.architectures import *
from oracle.custom_datasets.BTS import *
from oracle.custom_datasets.ZTF_sims import *
from oracle.taxonomies import BTS_Taxonomy, ORACLE_Taxonomy

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

def get_class_weights(labels):

    classes, counts = np.unique(labels, return_counts=True)
    weights = {}

    for c, n in zip(classes, counts):
        weights[c] = 1/n

    return weights

def get_model(model_choice):

    if model_choice == "BTS-lite":
        model = GRU(6)
    elif model_choice == "BTS":
        model = GRU_plus_MD(BTS_Taxonomy(), static_feature_dim=30)
    elif model_choice == "BTS_MM":
        model = GRU_MM(6, static_feature_dim=30)
    elif model_choice == "BTS_full_lc":
        model = GRU_plus_MD(6, static_feature_dim=30)
    elif model_choice == "ZTF_Sims-lite":
        model = GRU(6)
    return model


def get_train_loader(model_choice, batch_size, max_n_per_class, excluded_classes=[]):

    # Keep generator on the CPU
    generator = torch.Generator(device=device)

    if model_choice == "BTS-lite":

        # Load the training set
        transform = partial(truncate_BTS_light_curve_by_days_since_trigger, add_jitter=True)
        train_dataset = BTS_LC_Dataset(BTS_train_parquet_path, max_n_per_class=max_n_per_class, transform=transform, excluded_classes=excluded_classes)
        collate_fn = custom_collate_BTS

    elif model_choice == "BTS":

        # Load the training set
        transform = partial(truncate_BTS_light_curve_by_days_since_trigger, add_jitter=True)
        train_dataset = BTS_LC_Dataset(BTS_train_parquet_path, max_n_per_class=max_n_per_class, transform=transform, excluded_classes=excluded_classes)
        collate_fn = custom_collate_BTS

    elif model_choice == "BTS_MM":

        # Load the training set
        transform = partial(truncate_BTS_light_curve_by_days_since_trigger, add_jitter=True)
        train_dataset = BTS_LC_Dataset(BTS_train_parquet_path, max_n_per_class=max_n_per_class, transform=transform, excluded_classes=excluded_classes, include_postage_stamps=True)
        collate_fn = custom_collate_BTS

    elif model_choice == "BTS_full_lc":

        # Load the training set
        train_dataset = BTS_LC_Dataset(BTS_train_parquet_path, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
        collate_fn = custom_collate_BTS

    elif model_choice == "ZTF_Sims-lite":

        # Load the training set
        train_dataset = ZTF_SIM_LC_Dataset(ZTF_sim_train_parquet_path, include_lc_plots=False, transform=truncate_ZTF_SIM_light_curve_fractionally, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
        collate_fn = custom_collate_ZTF_SIM


    train_labels = train_dataset.get_all_labels()

    class_weights = get_class_weights(train_labels)
    train_weights = torch.from_numpy(np.array([class_weights[x] for x in train_labels]))
    sampler = WeightedRandomSampler(train_weights, len(train_weights))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=generator)

    return train_dataloader, train_labels


def get_val_loader(model_choice, batch_size, val_truncation_days, excluded_classes=[]):

    # Keep generator on the CPU
    generator = torch.Generator(device=device)

    if model_choice == "BTS-lite":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(BTS_LC_Dataset(BTS_val_parquet_path, transform=transform, excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        val_dataloader = DataLoader(concatenated_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_BTS, generator=generator)

    elif model_choice == "BTS":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(BTS_LC_Dataset(BTS_val_parquet_path, transform=transform,  excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        val_dataloader = DataLoader(concatenated_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_BTS, generator=generator)

    elif model_choice == "BTS_MM":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(BTS_LC_Dataset(BTS_val_parquet_path, transform=transform,  excluded_classes=excluded_classes, include_postage_stamps=True))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        val_dataloader = DataLoader(concatenated_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_BTS, generator=generator)

    elif model_choice == "BTS_full_lc":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(BTS_LC_Dataset(BTS_val_parquet_path, transform=transform, excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        val_dataloader = DataLoader(concatenated_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_BTS, generator=generator)


    elif model_choice == "ZTF_Sims-lite":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_ZTF_SIM_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(ZTF_SIM_LC_Dataset(ZTF_sim_val_parquet_path, transform=transform, excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        val_dataloader = DataLoader(concatenated_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_ZTF_SIM, generator=generator)

    val_labels = val_dataset[0].get_all_labels()
    return val_dataloader, val_labels

def get_test_loaders(model_choice, batch_size, max_n_per_class, days_list, excluded_classes=[], mapper=None):

    # Keep generator on the CPU
    generator = torch.Generator(device=device)

    test_loaders = []

    if model_choice == "BTS-lite":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
            test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)
            test_loaders.append(test_dataloader)

    elif model_choice == "BTS":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes, mapper=mapper)
            test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)
            test_loaders.append(test_dataloader)

    elif model_choice == "BTS_full_lc":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class,  excluded_classes=excluded_classes)
            test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)
            test_loaders.append(test_dataloader)

    elif model_choice == "BTS_MM":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            if 'Anomaly' not in excluded_classes:

                train_dataset_anomalies = BTS_LC_Dataset(BTS_train_parquet_path, include_postage_stamps=True, max_n_per_class=max_n_per_class,  excluded_classes=['SN-Ia','SN-Ib/c','SN-II','SLSN-I','CV','AGN'])
                train_dataset_anomalies.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)

                test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_postage_stamps=True, max_n_per_class=max_n_per_class,  excluded_classes=excluded_classes)
                test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)

                test_dataloader = DataLoader(ConcatDataset([train_dataset_anomalies, test_dataset]), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)

            else:

                test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_postage_stamps=True, max_n_per_class=max_n_per_class,  excluded_classes=excluded_classes)
                test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)

                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)



            test_loaders.append(test_dataloader)

    elif model_choice == "ZTF_Sims-lite":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = ZTF_SIM_LC_Dataset(ZTF_sim_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
            test_dataset.transform = partial(truncate_ZTF_SIM_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ZTF_SIM, generator=generator)
            test_loaders.append(test_dataloader)

    return test_loaders

