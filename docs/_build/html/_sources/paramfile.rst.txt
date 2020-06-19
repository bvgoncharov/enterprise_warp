====================
Parameter file usage
====================

To run Enterprise from a command line using a parameter file on a first pair of .par-.tim files in your data directory, please go to :code:`examples/` and run:

.. code-block:: console

   $ python run_example_paramfile.py --prfile example_paramfile.dat --num 0

Parameter file options
----------------------
- **{x} (a number in curly brackets)**: a separator, indicating that following parameters are only for model 'x'. If we specify more than one model and choose ptmcmc sampeler, Enterprise is launched in model comparison mode using the product-space method and :code:`class HyperModel` from `enterprise_extensions <https://github.com/stevertaylor/enterprise_extensions/>`__.
- **datadir**: directory with .par and .tim files
- **out**: output directory with Enterprise/Bilby results
- **overwrite**: option to overwrite overwrite an old Enterprise output
- **allpulsars**: whether to run analysis on all pulsars, or on a single pulsar (True/False)
- **noisefiles**: path to .json noise files needed to fix white noise parameters
- **sampler**: choose ptmcmcsampler or any of the samplers compatible with Bilby

Parameter file also automatically recognizes:
- Priors. Default parameters of prior distributions are set in :code:`ModelParams` class or its child class where you specify your custom noise models.
- Sampler keyword arguments. I.e., dlogz. They should only be specified after the sampler.

