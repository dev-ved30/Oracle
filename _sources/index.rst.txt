.. oracle documentation master file, created by
   sphinx-quickstart on Wed Oct 15 01:30:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation
====================

.. image:: ../figures/logo.jpeg
   :alt: Oracle Project Banner
   :align: center
   :width: 80%


**Oracle** is a modular and extensible framework for hierarchical classification of astronomical transients.  
It provides a set of tools for model training, evaluation, and inference on large-scale time-domain survey data.


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   installation
   quickstart
   api
   contributing
   citation
   changelog


1) Installation
---------------

Oracle is a pip installable package and was developed on `python 3.11`. 
I recommend creating a new environment for every project. 
If you are using conda, you can do this using

.. code-block:: bash

   conda create --name oracle python=3.11
   conda activate oracle

Then, install Oracle via pip: [Not available yet]

.. code-block:: bash

   pip install astro-oracle

If you would like to install the version of the repository up on git, please use:

.. code-block:: bash

   pip install git+https://github.com/dev-ved30/Oracle.git

If you intend to develop or modify the code, clone the repository and 
install in editable mode:

.. code-block:: bash

   git clone https://github.com/dev-ved30/Oracle.git
   cd Oracle
   pip install -e .

This should install all required dependencies.

---

2) Quick Start
--------------

Once installed, you can classify light curves using a pretrained model:

**For LSST (via ELAsTiCC)**

.. code-block:: python

   from astropy.table import Table
   from oracle.pretrained.ELAsTiCC import ORACLE1_ELAsTiCC, ORACLE1_ELAsTiCC_lite

   table = Table.read('notebooks/AGN_17032813.ecsv')

   # If you have access to the both the light curve and contextual features.
   model = ORACLE1_ELAsTiCC()
   model.score(table) # This provides the scores at each level of the hierarchy
   model.predict(table) # This provides the predicted class for each level of the hierarchy
   model.predict(table)  # This provide the latent space embeddings for each source

   # If you only have access to the light curve features, use the lite version instead. 
   model = ORACLE1_ELAsTiCC_lite()
   model.score(table) 
   model.predict(table) 
   model.embed(table)

**For BTS**

.. code-block:: python

   from astropy.table import Table
   from oracle.pretrained.BTS import ORACLE1_BTS

   table = Table.read('notebooks/fake_SN.ecsv')

 # If you have access to the both the light curve and contextual features.
   model = ORACLE1_BTS() 
   model.predict(table) # This provides the scores at each level of the hierarchy
   model.score(table) # This provides the predicted class for each level of the hierarchy
   model.embed(table)  # This provide the latent space embeddings for each source

For more examples, see the ``notebooks`` directory in the repository.

If you wish to train models from scratch, you can use the `oracle-train` CLI tool.

.. code-block:: bash

   $ oracle-train -h

This will display the help message:

.. code-block:: none

   Using cpu device
   usage: oracle-train [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--max_n_per_class MAX_N_PER_CLASS] [--alpha ALPHA] [--gamma GAMMA] [--dir DIR]
                     [--load_weights LOAD_WEIGHTS]
                     {BTS-lite,BTS,ZTF_Sims-lite,ELAsTiCC,ELAsTiCC-lite}

   positional arguments:
   {BTS-lite,BTS,ZTF_Sims-lite,ELAsTiCC,ELAsTiCC-lite}
                           Type of model to train.

   options:
   -h, --help            show this help message and exit
   --num_epochs NUM_EPOCHS
                           Number of epochs to train the model for.
   --batch_size BATCH_SIZE
                           Batch size used for training.
   --lr LR               Learning rate used for training.
   --max_n_per_class MAX_N_PER_CLASS
                           Maximum number of samples for any class. This allows for balancing of datasets.
   --alpha ALPHA         Alpha value used for the loss function. See Villar et al. (2024) for more information. [https://arxiv.org/abs/2312.02266]
   --gamma GAMMA         Exponent for the training weights.
   --dir DIR             Directory for saving the models and best model during training.
   --load_weights LOAD_WEIGHTS
                           Path to model which should be loaded before training stars.

Then, to test the model, you can use the `oracle-test` CLI tool.

.. code-block:: bash

   $ oracle-test -h

This will display the help message:

.. code-block:: none

   Using cpu device
   usage: oracle-test [-h] [--batch_size BATCH_SIZE] [--max_n_per_class MAX_N_PER_CLASS] dir

   positional arguments:
   dir                   Directory for saved model.

   options:
   -h, --help            show this help message and exit
   --batch_size BATCH_SIZE
                           Batch size used for test.
   --max_n_per_class MAX_N_PER_CLASS
                           Maximum number of samples for any class. This allows for balancing of datasets.

If you wish to add your own models, I recommend adding to the functions implemented in 'src/oracle/presets.py'. 
The training code is flexible enough to accommodate new model architectures and training procedures.
---

3) References
-------------

If you use this software in your research, please cite:

- Shah et al. (2025)  
  *ORACLE: A Real-Time, Hierarchical, Deep-Learning Photometric Classifier for the LSST.*  
  `[ADS Link] <https://ui.adsabs.harvard.edu/abs/2025arXiv250101496S/abstract>`_
- Shah et al. (2025)
  *ORACLE: A family of real-time, hierarchical classifiers for transient and variable phenomena in LSST alert streams.*
  `[Zenodo Link] <https://doi.org/10.5281/zenodo.15328166>`_

GitHub Repository:
~~~~~~~~~~~~~~~~~~
`https://github.com/dev-ved30/Oracle# <https://github.com/dev-ved30/Oracle#>`_

---

License & Disclosure
--------------------

Please see the `LICENSE <https://github.com/dev-ved30/Oracle/blob/main/LICENSE>`_ file for more details.


.. warning::
   The docstrings for this project were generated, in part, with the help of Large Language Models (LLMs)
   and vetted by the developer(s). While efforts have been made to ensure accuracy,
   if you find any discrepancies or errors, please report them via the
   `GitHub issue tracker <https://github.com/dev-ved30/Oracle/issues>`_.

---