#!/bin/python

import numpy as np
import sys
import os
import custom_models
import bilby
from enterprise_warp import enterprise_warp
from enterprise_warp import bilby_warp
from enterprise_extensions import model_utils

include_custom_models = True

opts = enterprise_warp.parse_commandline()
if include_custom_models: 
  custom = custom_models.CustomModels
else:
  custom = None
params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=custom)
params.init_pulsars()
#import ipdb; ipdb.set_trace()
#params.psrs[0].__dict__['custom_flag_bor'] = 'group'
#params.psrs[0].__dict__['flagval_bor'] = 'PDFB_10CM'
pta = enterprise_warp.init_pta(params)

#elif os.path.exists(params.directory+'info.txt'):
#    if params.overwrite=='False':
#        sys.exit(exit_message)
#    elif params.overwrite=='True':
#        os.remove(params.directory+'info.txt')
#        if os.path.exists(params.directory+'chain_1.txt'): os.remove(params.directory+'chain_1.txt')

if params.sampler == 'ptmcmc':
  if len(params.models)==1:
    sampler = model_utils.setup_sampler(pta[0], resume=False, outdir=params.directory)
    x0 = np.hstack(p.sample() for p in pta[0].params)
    sampler.sample(x0, params.nsamp, SCAMweight=30, AMweight=15, DEweight=50)
  else:
    super_model = model_utils.HyperModel(pta)
    print('Super model parameters: ', super_model.params)
    sampler = super_model.setup_sampler(resume=False, outdir=params.directory)
    N = params.nsamp
    x0 = super_model.initial_sample()
    sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
else:
    priors = bilby_warp.get_bilby_prior_dict(pta[0])
    parameters = dict.fromkeys(priors.keys())
    likelihood = bilby_warp.PTABilbyLikelihood(pta[0],parameters)
    label = os.path.basename(os.path.normpath(params.out))
    bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=params.directory, label=params.label, sampler=params.sampler, resume=True, nwalkers=500)

infofile = open(params.directory+'info.txt','w')
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
