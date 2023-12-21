====================
Parameter file usage
====================

To run Enterprise from a command line using a parameter file on a first pair of .par-.tim files in your data directory, please go to :code:`examples/` and run:

.. code-block:: console

   $ python run_example_paramfile.py --prfile example_paramfile.dat --num 0

Parameter file options
----------------------
- **{x} (a number in curly brackets)**: a separator, indicating that the following parameters are only for model 'x'. If we specify more than one model and choose ``ptmcmcsampler`` as a sampler, enterprise is launched in a model comparison mode using the product-space method and :code:`class HyperModel` from `enterprise_extensions <https://github.com/stevertaylor/enterprise_extensions/>`__.
- **timing_package**: a keyword argument of ``enterprise.pulsar.Pulsar()``, a default option is ``tempo2``, another option is ``pint``.
- **paramfile_label**: a unique label for the output directory, associated with the given parameter file. The label inside a noise model file(s) is (are) also added to the output directory name.
- **datadir**: a directory with .par and .tim files, or a path to pickled pulsars. In case it is a directory, make sure to have only one .par and .tim file per pulsar, with the same base name.
- **out**: output directory with Enterprise/Bilby results.
- **array_analysis**: whether to run analysis on a pulsar timing array, or on a single pulsar (True for array, False for single pulsars).
- **noisefiles**: a path to .json noise files needed to fix white noise parameters. White noise parameters (EFAC, EQUAD, ECORR) are fixed in case you add lines ``efac: -1``, ``equad: -1``, and ``ecorr: -1`` to a parameter file. 
- **sampler**: choose ``ptmcmcsampler`` or any of the samplers compatible with Bilby. Also, you can add any argument of a sampler as a line in a parameter file (e.g., ``AMweight`` for ``ptmcmcsampler``), they are automatically recognized. 
- **noise_model_file**: a path to ``enterprise_warp`` json noise model files, one for each model (under {x)}. See examples. 
- **psrlist**: if provided, only pulsars with names from this text file will be analyzed. A file format is a column of pulsar names.
- **ssephem**: Solar System ephemeris model, the default one is DE436.
- **clock**: a clock argument for enterprise, it is passed to ``libstempo`` or ``pint`` (timing packages). At the moment, it is not supported for single-pulsar noise analysis (check the use of ``Pulsar()`` in ``enterprise_warp.py``). A default option is ``None``.
- **fref**: reference radio frequency for "chromatic" (e.g., DM) noise, used in ``enterprise_models.py`` and it can be accessed in your own child class of ``StandardModels``.

Somewhat less useful parameters: 
- **overwrite**: an option to overwrite an old Enterprise output. It is not maintained at the moment.
- **load_toa_filenames**: keep a list of raw TOA file names (first column in a .tim file, at least for PPTA data) in a variable ``self.filenames`` in a parameter file. It was used for advanced noise modelling. 
- **mcmc_covm_csv**: a MCMC covariance matrix from ``ptmcmcsampler``. The idea is to use a covariance matrix for single-pulsar noise analyses to speed up full-PTA analysis. However, it requires a modification to ``enterprise_extensions``, so it is currently not supported.
- ****

Parameter file also automatically recognizes:
- Priors. Default parameters of prior distributions are set in :code:`ModelParams` class or its child class where you specify your custom noise models.
- Sampler keyword arguments. I.e., dlogz. They should only be specified after the sampler.

