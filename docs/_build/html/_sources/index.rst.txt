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

   conda create --name oracle2 python=3.11
   conda activate oracle2

Then, install Oracle via pip:

.. code-block:: bash

   pip install

If you intend to develop or modify the code, clone the repository and 
install in editable mode:

.. code-block:: bash

   git clone https://github.com/dev-ved30/Hierarchical-VT.git
   cd Hierarchical-VT
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

   model = ORACLE1_ELAsTiCC()
   model.score(table)
   model.predict(table)
   model.predict(table)

   model = ORACLE1_ELAsTiCC_lite()
   model.score(table)
   model.predict(table)
   model.predict(table)

**For BTS**

.. code-block:: python

   from astropy.table import Table
   from oracle.pretrained.BTS import ORACLE1_BTS

   table = Table.read('notebooks/fake_SN.ecsv')

   model = ORACLE1_BTS()
   model.predict(table)
   model.score(table)
   model.embed(table)

The output is a hierarchical class probability distribution for each level of the taxonomy.

For more examples, see the ``notebooks`` directory in the repository.

---

3) References
-------------

If you use this software in your research, please cite:

- Shah et al. (2025)  
  *ORACLE: A Real-Time, Hierarchical, Deep-Learning Photometric Classifier for the LSST*  
  `ADS Link <https://ui.adsabs.harvard.edu/abs/2025arXiv250101496S/abstract>`_

- Shah et al. (2025)
   *ORACLE- A family of real-time, hierarchical classifiers for transient and variable phenomena in LSST alert streams.*
   `Zenodo Link <https://doi.org/10.5281/zenodo.15328166>`_
   
GitHub Repository:
~~~~~~~~~~~~~~~~~~
`https://github.com/dev-ved30/Hierarchical-VT <https://github.com/dev-ved30/Hierarchical-VT>`_

---

License & Disclosure
--------------------

This software is made available for **research purposes only**.  
By using this software, you agree to cite the above paper in any publications that make use of it.

.. warning::
   The docstrings for this project were generated, in part, with the help of Large Language Models (LLMs)
   and vetted by the developer(s). While efforts have been made to ensure accuracy,
   if you find any discrepancies or errors, please report them via the
   `GitHub issue tracker <https://github.com/dev-ved30/Hierarchical-VT/issues>`_.

---

Acknowledgments
---------------

Oracle is developed as part of ongoing research in time-domain astronomy.
We thank the transient astronomy community for providing open datasets and tools that
make projects like this possible.

