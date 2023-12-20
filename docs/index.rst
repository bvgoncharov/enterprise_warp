.. enterprise_warp documentation master file, created by
   sphinx-quickstart on Thu Jun 27 22:53:00 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to enterprise_warp's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   paramfile
   api

enterprise_warp
===============

enterprise_warp is a set of tools for pulsar timing analysis with `Enterprise <https://github.com/nanograv/enterprise/>`__, `Libstempo <https://github.com/vallis/libstempo/>`__ and `Tempo2 <https://bitbucket.org/psrsoft/tempo2>`__. It uses the `Bilby <https://git.ligo.org/lscsoft/bilby/>`__ to enable Bayesian inference for pulsar timing arrays using many different samplers and all other advantages of Bilby.

Look how easy it is to run from the command line:

.. code-block:: console

   $ python run_example_paramfile.py --prfile example_params/default_model_dynesty.dat --num 0

Here ``run_example_paramfile.py`` is the script available in ``examples/``. To view and analyze the results, run:

.. code-block:: console

   $ python -m enterprise_warp.results --result example_params/default_model_dynesty.dat --info 1 --corner 1

Where ``--result`` can be a parameter file or an output directory. The above command saves a corner plot to the output directory. Other command line options are related to noise files, Bayes factors, chain plots, etc. Please run ``python -m enterprise_warp.results -h`` to list available options.

Features
--------

- Configure your runs with parameter files instead of editing python scripts. This allows easier management of your data analysis progress. Keep track of your input and output directories.

- The code is optimized for parallel computing via the usage of command line arguments (i.e., for different pulsars)

- Simply choose any MCMC/Nested sampler that is available in Bilby: PTMCMCSampler, Dynesty, Nestle, PyPolyChord, PyMC3, ptmcee, and more. Evaluate evidence, perform parameter estimation, or directly compare models with product-space method and PTMCMCSampler.

- Accellerate your data analysis with MPI using PyPolychord sampler: distribute heavy calculations between ~10-100 cores of a supercomputer. The code has special options for MPI runs.

- Use noise model files with tailored noise models for each pulsar and common noise processes in all pulsars. Run analysis with individual pulsars or the whole pulsar timing array.

- Save time on setting up runs: all sampler keyword arguments, your custom noise models and priors are automatically recongized in parameter files.

- Easily add your own models, using several examples. Or load models from `enterprise_extensions <https://github.com/nanograv/enterprise_extensions/>`__ or other code.

- And more! 

Getting started
---------------

You should simply have the main running script which imports `enterprise_warp`. When running it, you should point it to a parameter file. If necessary, you can also have a custom noise model file `.py` with a parent class `StandardModels`, with custom models for signal and noise. It is then imported in the main running script. 

When starting your new project with `enterprise_warp`, you can copy/fork examples below:
- `PPTA DR2 noise analysis (2019-2020) <https://github.com/bvgoncharov/ppta_dr2_noise_analysis>`. Here, the main running script is `run_dr2.py` and custom noise models are in `ppta_dr2_models.py`.
- `Search for the gravitational wave background with PPTA DR2 (2020-2021) <https://github.com/bvgoncharov/correlated_noise_pta_2020>`. Here, the main running script is `run_analysis.py` and custom noise models are in `ppta_dr2_models.py`.

License
-------

The project is licensed under the MIT license.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
