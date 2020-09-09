#!/bin/python

import numpy as np
import sys
import os
import inspect
import bilby
from enterprise_warp import enterprise_warp
from enterprise_warp import bilby_warp
from enterprise_extensions import model_utils

import custom_models

include_custom_models = True

opts = enterprise_warp.parse_commandline()
if include_custom_models: 
  custom = custom_models.CustomModels
else:
  custom = None

params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=custom)
pta = enterprise_warp.init_pta(params)

if params.sampler == 'ptmcmcsampler':
  if len(params.models)==1:
    sampler = model_utils.setup_sampler(pta[0], resume=False,
                                        outdir=params.output_dir)
    x0 = np.hstack(p.sample() for p in pta[0].params)
    sampler.sample(x0, params.nsamp, **params.sampler_kwargs)
  else:
    super_model = model_utils.HyperModel(pta)
    print('Super model parameters: ', super_model.params)
    sampler = super_model.setup_sampler(resume=False, outdir=params.output_dir)
    N = params.nsamp
    x0 = super_model.initial_sample()

    # Remove extra kwargs that Bilby took from PTSampler module, not ".sample"
    ptmcmc_sample_kwargs = inspect.getargspec(sampler.sample).args
    upd_sample_kwargs = {key: val for key, val in params.sampler_kwargs.items()\
                         if key in ptmcmc_sample_kwargs}
    del upd_sample_kwargs['Niter']
    del upd_sample_kwargs['p0']

    sampler.sample(x0, N, **upd_sample_kwargs)
else:
    priors = bilby_warp.get_bilby_prior_dict(pta[0])
    parameters = dict.fromkeys(priors.keys())
    likelihood = bilby_warp.PTABilbyLikelihood(pta[0],parameters)
    label = os.path.basename(os.path.normpath(params.out))
    if opts.mpi_regime != 1:
      bilby.run_sampler(likelihood=likelihood, priors=priors,
                        outdir=params.output_dir, label=params.label,
                        sampler=params.sampler, **params.sampler_kwargs)
    else:
      print('Preparations for the MPI run are complete - now set \
             opts.mpi_regime to 2 and enjoy the speed!')

