"""
Top-level module for defining presets and utility functions for model selection,
data loading, and training configurations in the ORACLE project."""
import os
import torch

from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from oracle.architectures import *
from oracle.custom_datasets.BTS import *
from oracle.custom_datasets.ZTF_sims import *
from oracle.custom_datasets.ELAsTiCC import *  
from oracle.taxonomies import BTS_Taxonomy, ORACLE_Taxonomy

device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)

# Dataloader parameters for training and validation
shuffle = True
num_workers = 4
pin_memory = False
prefetch_factor = 2
persistent_workers = False

def worker_init_fn(worker_id):
    """Ensure proper random seeding in each worker process."""
    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_class_weights(labels):
    """
    Calculates the weights for each class using inverse frequency weighting.

    Parameters:
        labels (array-like): An iterable of labels from which unique classes and their counts are derived.

    Returns:
        dict: A dictionary where keys are the unique class labels and values are their corresponding weights (1/count).
    """

    classes, counts = np.unique(labels, return_counts=True)
    weights = {}

    for c, n in zip(classes, counts):
        weights[c] = 1/n

    return weights

def get_model(model_choice):
    """
    Retrieves and instantiates a model based on the provided model choice.

    Parameters:
        model_choice (str): A string identifier for the desired model configuration.
            Valid options and their corresponding behaviors are:
                - "BTS-lite": Uses BTS_Taxonomy to instantiate a GRU model.
                - "BTS": Uses BTS_Taxonomy to instantiate a GRU_MD model with a static feature dimension of 30.
                - "ZTF_Sims-lite": Uses BTS_Taxonomy to instantiate a GRU model.
                - "ELAsTiCC-lite": Uses ORACLE_Taxonomy to instantiate a GRU model.
                - "ELAsTiCC": Uses ORACLE_Taxonomy to instantiate a GRU_MD model with a static feature dimension of 18.

    Returns:
        The model instance associated with the given model_choice.
    """

    if model_choice == "BTS-lite":
        taxonomy = BTS_Taxonomy()
        model = GRU(taxonomy)
    elif model_choice == "BTS":
        taxonomy = BTS_Taxonomy()
        model = GRU_MD(taxonomy, static_feature_dim=30)
    elif model_choice == "ZTF_Sims-lite":
        taxonomy = BTS_Taxonomy()
        model = GRU(taxonomy)
    elif model_choice == "ELAsTiCC-lite":
        taxonomy = ORACLE_Taxonomy()
        model = GRU(taxonomy)
    elif model_choice == "ELAsTiCC":
        taxonomy = ORACLE_Taxonomy()
        model = GRU_MD(taxonomy, static_feature_dim=18)
    elif model_choice == "ELAsTiCCv2":
        taxonomy = ORACLE_Taxonomy()
        model = GRU_MD_Improved(taxonomy, static_feature_dim=18)
    elif model_choice == "BTSv2":
        taxonomy = BTS_Taxonomy()
        model = GRU_MD_Improved(taxonomy, static_feature_dim=30)
    elif model_choice == "BTSv2-pro":
        taxonomy = BTS_Taxonomy()
        model = GRU_MD_MM_Improved(taxonomy)
    elif model_choice == "BTSv2_PSonly":
        taxonomy = BTS_Taxonomy()
        model = ConvNeXt(taxonomy)
    return model

def get_train_loader(model_choice, batch_size, max_n_per_class, gamma, excluded_classes=[]):
    """
    Generates a DataLoader for training based on the provided model choice and dataset configuration.

    Parameters:
        model_choice (str): The identifier for the model and corresponding dataset. Supported values include:
                            "BTS-lite", "BTS", "ZTF_Sims-lite", "ELAsTiCC-lite", and "ELAsTiCC".
        batch_size (int): The number of samples per batch to load.
        max_n_per_class (int): The maximum number of samples to include per class in the dataset.
        excluded_classes (list, optional): A list of class identifiers to exclude from the dataset. 
                                           Defaults to an empty list.

    Returns:
        tuple: A tuple containing:
               - DataLoader(torch.utils.data.DataLoader): A DataLoader instance for iterating through the training dataset.
               - list(list): A list of all labels present in the training dataset.
    """

    # Keep generator on the CPU
    torch.set_default_device('cpu')
    generator = torch.Generator(device='cpu')

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

    elif model_choice == "ZTF_Sims-lite":

        # Load the training set
        train_dataset = ZTF_SIM_LC_Dataset(ZTF_sim_train_parquet_path, include_lc_plots=False, transform=truncate_ZTF_SIM_light_curve_fractionally, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
        collate_fn = custom_collate_ZTF_SIM

    elif model_choice == "ELAsTiCC-lite":

        # Load the training set
        train_dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_train_parquet_path, include_lc_plots=False, transform=truncate_ELAsTiCC_light_curve_by_days_since_trigger, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
        collate_fn = custom_collate_ELAsTiCC

    elif model_choice == "ELAsTiCC":

        # Load the training set
        train_dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_train_parquet_path, include_lc_plots=False, transform=truncate_ELAsTiCC_light_curve_by_days_since_trigger, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
        collate_fn = custom_collate_ELAsTiCC

    elif model_choice == "ELAsTiCCv2":

        # Load the training set
        train_dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_train_parquet_path, include_lc_plots=False, transform=truncate_ELAsTiCC_light_curve_by_days_since_trigger, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
        collate_fn = custom_collate_ELAsTiCC

    elif model_choice == "BTSv2":

        # Load the training set
        transform = partial(truncate_BTS_light_curve_by_days_since_trigger, add_jitter=True)
        train_dataset = BTS_LC_Dataset(BTS_train_parquet_path, max_n_per_class=max_n_per_class, transform=transform,  img_transform=augment_panstarss, excluded_classes=excluded_classes)
        collate_fn = custom_collate_BTS

    elif model_choice == "BTSv2-pro":

        # Load the training set
        transform = partial(truncate_BTS_light_curve_by_days_since_trigger, add_jitter=True)
        train_dataset = BTS_LC_Dataset(BTS_train_parquet_path, max_n_per_class=max_n_per_class, include_PS_images=True, transform=transform, excluded_classes=excluded_classes)
        collate_fn = custom_collate_BTS
    
    elif model_choice == "BTSv2_PSonly":

        # Load the training set
        transform = partial(truncate_BTS_light_curve_by_days_since_trigger, add_jitter=True)
        train_dataset = BTS_LC_Dataset(BTS_train_parquet_path, include_PS_images=True, max_n_per_class=max_n_per_class, transform=transform, excluded_classes=excluded_classes)
        collate_fn = custom_collate_BTS

    train_labels = train_dataset.get_all_labels()

    class_weights = get_class_weights(train_labels)
    train_weights = torch.from_numpy(np.array([class_weights[x] for x in train_labels]))
    sampler = WeightedRandomSampler(train_weights**(1-gamma), len(train_weights))
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  collate_fn=collate_fn, 
                                  generator=generator, 
                                  pin_memory=pin_memory, 
                                  num_workers=num_workers, 
                                  prefetch_factor=prefetch_factor, 
                                  persistent_workers=persistent_workers, 
                                  worker_init_fn=worker_init_fn)
    
    return train_dataloader, train_labels

def get_val_loader(model_choice, batch_size, val_truncation_days, max_n_per_class, excluded_classes=[]):
    """
    Creates a DataLoader for the validation dataset along with its corresponding labels based on the specified model choice.

    This function selects a dataset and transformation based on the model_choice provided and then concatenates 
    the datasets (one for each truncation day specified in val_truncation_days). It returns a DataLoader constructed 
    with the concatenated dataset and the set of all labels retrieved from the first dataset instance.

    Parameters:
        model_choice (str):
            The name of the model variant to use. Valid options include:
            "BTS-lite", "BTS", "ZTF_Sims-lite", "ELAsTiCC-lite", "ELAsTiCC".
        batch_size (int):
            The number of samples per batch in the returned DataLoader.
        val_truncation_days (list):
            A list of days used to truncate the light curves; each day corresponds to a transformation
            applied to the dataset.
        max_n_per_class (int): 
            The maximum number of samples to include per class in the dataset.
        excluded_classes (list, optional):
            A list of classes to be excluded from the dataset. Defaults to an empty list.

    Returns:
        tuple: A tuple containing:
            - DataLoader: The DataLoader for the concatenated validation dataset with the specified batch size and collate function, constructed with a CPU-based torch.Generator.
            - list: A list of validation labels obtained from the first dataset in the list via get_all_labels().
    """

    # Keep generator on the CPU
    torch.set_default_device('cpu')
    generator = torch.Generator(device='cpu')

    if model_choice == "BTS-lite":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(BTS_LC_Dataset(BTS_val_parquet_path, transform=transform, excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        collate_func = custom_collate_BTS

    elif model_choice == "BTS":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(BTS_LC_Dataset(BTS_val_parquet_path, transform=transform,  excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        collate_func = custom_collate_BTS

    elif model_choice == "ZTF_Sims-lite":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_ZTF_SIM_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(ZTF_SIM_LC_Dataset(ZTF_sim_val_parquet_path, transform=transform, excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        collate_func = custom_collate_ZTF_SIM

    elif model_choice == "ELAsTiCC-lite":

        # Load the training set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_ELAsTiCC_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(ELAsTiCC_LC_Dataset(ELAsTiCC_val_parquet_path, max_n_per_class, transform=transform, excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        collate_func = custom_collate_ELAsTiCC

    elif model_choice == "ELAsTiCC":

        # Load the training set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_ELAsTiCC_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(ELAsTiCC_LC_Dataset(ELAsTiCC_val_parquet_path, max_n_per_class, transform=transform, excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        collate_func = custom_collate_ELAsTiCC

    elif model_choice == "ELAsTiCCv2":

        # Load the training set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_ELAsTiCC_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(ELAsTiCC_LC_Dataset(ELAsTiCC_val_parquet_path, max_n_per_class, transform=transform, excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        collate_func = custom_collate_ELAsTiCC

    elif model_choice == "BTSv2":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(BTS_LC_Dataset(BTS_val_parquet_path, transform=transform,  excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        collate_func = custom_collate_BTS

    elif model_choice == "BTSv2-pro":

        # Load the validation set
        val_dataset = []
        for d in val_truncation_days:
            transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(BTS_LC_Dataset(BTS_val_parquet_path, transform=transform,  include_PS_images=True, excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        collate_func = custom_collate_BTS

    elif model_choice == "BTSv2_PSonly":

        # Load the validation set
        val_dataset = []
        for d in [1024]:
            transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            val_dataset.append(BTS_LC_Dataset(BTS_val_parquet_path, include_PS_images=True, transform=transform,  excluded_classes=excluded_classes))
        concatenated_val_dataset = ConcatDataset(val_dataset)
        collate_func = custom_collate_BTS

    val_dataloader = DataLoader(concatenated_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_func, generator=generator, pin_memory=pin_memory, num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, worker_init_fn=worker_init_fn)
    val_labels = val_dataset[0].get_all_labels()
    return val_dataloader, val_labels

def get_test_loaders(model_choice, batch_size, max_n_per_class, days_list, excluded_classes=[], mapper=None):
    """
    Generates and returns a list of test DataLoaders configured according to the specified model type and parameters.
    Parameters:
        model_choice (str): Specifies which model and corresponding dataset to use. Supported values include
            "BTS-lite", "BTS", "ZTF_Sims-lite", "ELAsTiCC-lite", and "ELAsTiCC".
        batch_size (int): The size of batches to generate from each DataLoader.
        max_n_per_class (int): The maximum number of samples to include per class in the dataset.
        days_list (list): A list of day values used to dynamically configure a transformation (truncating light curves)
            applied to the test dataset.
        excluded_classes (list, optional): A list of class labels to exclude from the dataset. Defaults to an empty list.
        mapper (dict, optional): An optional dictionary used to map or modify dataset labels/structure (only used for the
            "BTS" model). Defaults to None.
    Returns:
        List[DataLoader]: A list of DataLoader objects, each configured with a custom transformation based on a corresponding
        day from days_list.

    Note:
        Each DataLoader is constructed for a specific truncation of the light curve based on the day value.
        The generator for DataLoader shuffling is explicitly created on the CPU.
    """

    # Keep generator on the CPU
    torch.set_default_device('cpu')
    generator = torch.Generator(device='cpu')

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

    elif model_choice == "ZTF_Sims-lite":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = ZTF_SIM_LC_Dataset(ZTF_sim_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
            test_dataset.transform = partial(truncate_ZTF_SIM_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ZTF_SIM, generator=generator)
            test_loaders.append(test_dataloader)

    elif model_choice == "ELAsTiCC-lite":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
            test_dataset.transform = partial(truncate_ELAsTiCC_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ELAsTiCC, generator=generator)
            test_loaders.append(test_dataloader)

    elif model_choice == "ELAsTiCC":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
            test_dataset.transform = partial(truncate_ELAsTiCC_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ELAsTiCC, generator=generator)
            test_loaders.append(test_dataloader)

    elif model_choice == "ELAsTiCCv2":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes)
            test_dataset.transform = partial(truncate_ELAsTiCC_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ELAsTiCC, generator=generator)
            test_loaders.append(test_dataloader)

    elif model_choice == "BTSv2":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes, mapper=mapper)
            test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)
            test_loaders.append(test_dataloader)


    elif model_choice == "BTSv2-pro":

        for d in days_list:
            
            # Set the custom transform and recreate dataloader
            test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_PS_images=True, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes, mapper=mapper)
            test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)
            test_loaders.append(test_dataloader)


    elif model_choice == "BTSv2_PSonly":

        # make sure the val set is not loaded many times for the image model.
        for d in [1024]:
            
            # Set the custom transform and recreate dataloader
            test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_PS_images=True, max_n_per_class=max_n_per_class, excluded_classes=excluded_classes, mapper=mapper)
            test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)
            test_loaders.append(test_dataloader)

    return test_loaders

