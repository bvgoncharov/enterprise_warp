"""
An example of the script to run enterprise_warp
"""

import numpy as np
import sys
sys.path.insert(0,'/work/boris.goncharov/dev_enterprise_warp/')
import os
import inspect
import bilby
from enterprise_warp import enterprise_warp
from enterprise_warp import bilby_warp
from enterprise_extensions import hypermodel

import custom_models

opts = enterprise_warp.parse_commandline()

# Adding custom models is optional:
custom = custom_models.CustomModels
#custom = None

params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=custom)
pta = enterprise_warp.init_pta(params)

if params.sampler == 'ptmcmcsampler':
    super_model = hypermodel.HyperModel(pta)
    print('Super model parameters: ', super_model.params)
    print('Output directory: ', params.output_dir)
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

    bilby.run_sampler(likelihood=likelihood, priors=priors,
                        outdir=params.output_dir, label=params.label,
                        sampler=params.sampler, **params.sampler_kwargs)


