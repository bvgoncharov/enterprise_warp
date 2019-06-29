==============
Parameter file
==============

To run Enterprise from a command line using a parameter file on a first pair of .par-.tim files in your data directory, please go to :code:`examples/` and run:

.. code-block:: console

   $ python run_example_paramfile.py --prfile example_paramfile.dat --num 0

Parameter file options
----------------------
- **{x} (a number in curly brackets)**: a separator, indicating that following parameters are only for model 'x'. If we specify more than one model, then Enterprise is launched in model comparison regime using :code:`class HyperModel` from `enterprise_extensions <https://github.com/stevertaylor/enterprise_extensions/>`__.
- **datadir**: directory with .par and .tim files
- **out**: output directory with Enterprise/Bilby results
- **overwrite**: option to overwrite overwrite an old Enterprise output
- **allpulsars**: whether to run analysis on all pulsars, or on a single pulsar (True/False)
- **noisefiles**: path to .json noise files if we are including fixed efac/equad
- **sampler**: in case of other sampler than PTMCMCSampler, we use Bilby

Next, we define unifrom prior ranges for parameters. If there is a negative number, i.e. "-1", parameter is taken as a constant from .par file for each observing backend. If there is a positive number, it is taken as an actual and constant parameter.

- **efacpr**: EFAC uniform prior range
- **efacsel**: EFAC selection
- **equadpr**: log10 EQUAD uniform prior range
- **equadsel**: EQUAD selection
- **erorrpr**: log10 ECORR uniform prior range
- **ecorrsel**: ECORR selection
- **sn_model**: model for red/spin noise, common for all systems and observing bands for each pulsars. Options: none, default.
- **sn_lgApr**: lg10 spin noise powerlaw amplitude 
- **sn_gpr**: spin noise powerlaw index
- **dm_model**: model for DM noise. Options: none, default.
- **dm_lgApr**: log10 DM-variation noise powerlaw amplitude
- **dm_gpr**: DM-variation noise powerlaw index

Selection options for white noise
=================================
Selections allow us to estimate a certain white noise parameter in different splitting regimes, or how to define a constant white noise parameter:

- **no_selection** - for a single pulsar, without splitting
- **by_backend** - for each observing backend (using backend flags)
- **nanograv_backends** - for each observing backend, but only for NANOGrav pulsars (only using NANOGrav flags)

Custom options and models for a parameter file
----------------------------------------------

One can add custom options for their own models in Enterprise by adding a python code with class that contains:

- dictionary, as an instruction for reading a parameter file
- function that takes in parameters from a dictionary, and outputs Enterprise signal object

An example can be found in :code:`examples/custom_models.py`
