#!/bin/python

import numpy as np
import sys
import os
import custom_models
import bilby
from enterprise_warp import enterprise_warp
from enterprise_warp import bilby_warp
from enterprise_extensions import model_utils

include_custom_models = False

opts = enterprise_warp.parse_commandline()
if include_custom_models: 
  custom = custom_models.CustomModels
else:
  custom = None

params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=custom)
pta = enterprise_warp.init_pta(params)

if params.sampler == 'PTMCMCSampler':
  if len(params.models)==1:
    sampler = model_utils.setup_sampler(pta[0], resume=False, outdir=params.output_dir)
    x0 = np.hstack(p.sample() for p in pta[0].params)
    sampler.sample(x0, params.nsamp, **params.sampler_kwargs)
  else:
    super_model = model_utils.HyperModel(pta)
    print('Super model parameters: ', super_model.params)
    sampler = super_model.setup_sampler(resume=False, outdir=params.output_dir)
    N = params.nsamp
    x0 = super_model.initial_sample()
    sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
else:
    priors = bilby_warp.get_bilby_prior_dict(pta[0])
    parameters = dict.fromkeys(priors.keys())
    likelihood = bilby_warp.PTABilbyLikelihood(pta[0],parameters)
    label = os.path.basename(os.path.normpath(params.out))
    if opts.mpi_regime != 1:
      bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=params.output_dir, label=params.label, sampler=params.sampler, **params.sampler_kwargs)
    else:
      print('Preparations for the MPI run are complete - now set \
             opts.mpi_regime to 2 and enjoy the speed!')

infofile = open(params.output_dir+'info.txt','w')
if params.allpulsars=='False':
  infofile.write('Name '+psrs[0].name+'\n')
  infofile.write('Tspan(sec) '+str(Tspan)+'\n')
  infofile.write('Ra(rad) '+str(raj)+'\n')
  infofile.write('Dec(rad) '+str(decj)+'\n')
  infofile.write('TOAerrMedian(sec) '+str(toaerrmed)+'\n')
  infofile.write('TOAerrMean(sec) '+str(toaerrmean)+'\n')
if params.sampler == 'nestle':
  infofile.write('Logz '+str(result.logz)+'\n')
  infofile.write('Logz_err '+str(result.logzerr)+'\n')
elif params.sampler == 'dynesty':
  infofile.write('Logz '+str(sampler.results.logz[-1])+'\n')
  infofile.write('Logz_err '+str(sampler.results.logzerr[-1])+'\n')
elif params.sampler == 'emcee':
  infofile.write('Logz '+str(sampler.thermodynamic_integration_log_evidence()[0])+'\n')
  infofile.write('Logz_err '+str(sampler.thermodynamic_integration_log_evidence()[1])+'\n')
#elif params.sampler == 'ptmcmc':
#  infofile.write('Neff '+str(sampler.Neff)+'\n')
infofile.close()

print('Finished: ',opts.num)
