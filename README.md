# ORACLE

[![DOI](https://img.shields.io/badge/astro.IM-2501.01496-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2501.01496) [![Docs](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://dev-ved30.github.io/Oracle/) [![DOI](https://zenodo.org/badge/doi/10.48550/arXiv.2501.01496.svg)](https://ui.adsabs.harvard.edu/abs/2025arXiv250101496S/abstract) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.15099699.svg)](https://zenodo.org/records/15099699)

> [!NOTE]  
> This is a complete rewrite of the original Oracle codebase using PyTorch, and is meant to supersede it. If you are looking for the original repository, you can find it [here](https://github.com/uiucsn/Astro-ORACLE/tree/main).

<!-- > [!WARNING]  
> ⚠️ This is a warning banner. Be careful! -->

<p align="center">
  <img src="figures/logo.jpeg" width="500" />
</p>


We present ORACLE, the first hierarchical deep-learning model for real-time, context-aware classification of transient and variable astrophysical phenomena. ORACLE is a recurrent neural network with Gated Recurrent Units (GRUs), and has been trained using a custom hierarchical cross-entropy loss function to provide high-confidence classifications along an observationally-driven taxonomy with as little as a single photometric observation. Contextual information for each object, including host galaxy photometric redshift, offset, ellipticity and brightness, is concatenated to the light curve embedding and used to make a final prediction.

For more information, please read the our paper - https://ui.adsabs.harvard.edu/abs/2025arXiv250101496S/abstract

If you use any of this code in your own work, please cite the associated paper and software using the references in the [CITATION.cff](CITATION.cff) file.


## Installation:

Please refer to the documentation [here](https://dev-ved30.github.io/Oracle/) for installation instructions.

# Classification Taxonomy

There is no universally correct classification taxonomy - however we want to build something that is able to best serve real world science cases. For obvious reasons, the leaf nodes need to be the true class of the object however what we decide for nodes higher up in the taxonomy is ultimately determined by the science case. 

For this work, we are implementing a hierarchy that will be of interest to the TVS (Transient and Variable star) community since it overlaps well with the classes of the elasticc data set. The exact taxonomy used is shown below:

![](figures/HC_taxonomy.png)

A trap we wanted to avoid was mixing different "metaphors" for classification. For instance, we decided against using `Galactic vs Extra galactic` classification since we would be mixing a spatial distinction with temporal ones (like `Periodic vs Non periodic`). This makes the problem trickier since some objects, like Cepheids, can be both galactic and extragalactic which would result in an artificial inflation in the number of leaf nodes without adding much value to the science case.

*Note:* There is some inconsistency around the Mira/LPV object in the elasticc documentation however we have confirmed that this object was removed for the elasticc2 data set that we use in this work.

# Machine learning architecture

![](figures/model_arch.png)

Once again, we have much more detailed discussion in the paper.

# Found Bugs?
We appreciate any support in the form of bug reports in order to provide the best possible experience. Bugs can be reported in the `Issues` tab.

# Contributing
If you like what we're doing and want to contribute to the project, please start by reaching out to the developer. We are happy to accept suggestions in the Issues tab. 

# Some cool results:

Overall model performance:


![](figures/f1-performance.jpg)

First at the root,

![](figures/level_1_cf_days.gif)
![](figures/level_1_roc_days.gif)

At the next level in the hierarchy

![](figures/level_2_cf_days.gif)
![](figures/level_2_roc_days.gif)

And finally, at the leaf...

![](figures/leaf_cf_days.gif)
![](figures/leaf_roc_days.gif)

## References:
* ORACLE - https://arxiv.org/abs/2501.01496
* HXE Loss Function - https://arxiv.org/abs/1912.09393
* WHXE Loss Function - https://arxiv.org/abs/2312.02266
* VT Model - https://arxiv.org/abs/2502.20479

