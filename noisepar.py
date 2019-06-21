#!/bin/python

from __future__ import division

import numpy as np
from astropy.stats import LombScargle
import glob
import ipdb
import sys
import time
import hashlib
import pickle

from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals.selections import Selection
#from tests.enterprise_test_data import params.datadir

# Additional, by Boris:
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from corner import corner
import os
#from core import parse_commandline, init_pta, Params, paramstringcut, Nestle_PT
from core import *

from enterprise_extensions import models, model_utils

opts = parse_commandline()

params = Params(opts.prfile,opts=opts)

psrs = init_pulsars(params)

pta = dict.fromkeys(params.model_ids)

for ii, pp in params.models.items():

  # initialize PTA
  pta[ii] = init_pta(pp,Tspan,psrs,parfiles,timfiles,ii)

    # Print and write order of params for future use in noiseparPlot.py
    print('Params order: ', pta[ii].param_names)
    np.savetxt(directory+'/pars.txt', pta[ii].param_names, fmt='%s')

if not os.path.exists(directory):
    os.makedirs(directory)
elif os.path.exists(directory+'info.txt'):
    if params.overwrite=='False':
        sys.exit(exit_message)
    elif params.overwrite=='True':
        os.remove(directory+'info.txt')
        if os.path.exists(directory+'chain_1.txt'): os.remove(directory+'chain_1.txt')

if params.sampler == 'ptmcmc':
  if len(params.models)==1:
    sampler = model_utils.setup_sampler(pta[0], resume=False, outdir=directory)
    x0 = np.hstack(p.sample() for p in pta[0].params)
    sampler.sample(x0, pp.nsamp, SCAMweight=30, AMweight=15, DEweight=50)
  else:
    super_model = model_utils.HyperModel(pta)
    print('Super model parameters: ', super_model.params)
    sampler = super_model.setup_sampler(resume=False, outdir=directory)
    N = params.nsamp
    x0 = super_model.initial_sample()
    sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
else:
    priors = get_bilby_prior_dict(pta[0])
    parameters = dict.fromkeys(priors.keys())
    likelihood = PTABilbyLikelihood(pta[0],parameters)
    label = os.path.basename(os.path.normpath(params.out))
    bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=directory, label=params.label, sampler=params.sampler, resume=True, nwalkers=500)

infofile = open(directory+'info.txt','w')
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

print('Finished: ',opts.num,'/',len(parfiles))
