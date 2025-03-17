import numpy as np
import pandas as pd
import json
import glob
import os
import optparse
import warnings
import hashlib
import pickle

import enterprise.signals.parameter as parameter
from enterprise.signals import signal_base
import enterprise.signals.gp_signals as gp_signals
from enterprise.pulsar import Pulsar
import enterprise.constants as const
from enterprise_extensions import models
from .enterprise_models import StandardModels
import discovery as ds

try:
  from mpi4py import MPI
  process_rank = MPI.COMM_WORLD.Get_rank()
except:
  process_rank = 0

try:
  from bilby import sampler as bimpler
except:
  warnings.warn("Warning: failed to import bilby.sampler")


def init_pta_discovery(params_all):
  """
  Initiate enterprise signal models and enterprise.signals.signal_base.PTA.
  """
  ptas = dict.fromkeys(params_all.models)
  for ii, params in params_all.models.items():

    allpsr_model = params_all.noise_model_obj(psr=params_all.psrs,
                                              params=params)

    models = list()
    from_par_file = list()

# ====== NEW =======

#m_sep is now psr_model
#We will also change m_all to to pta_model
    psr_model_list = []
    #Adding single pulsar models
    #Loop over pulsars
    for pnum, psr in enumerate(params_all.psrs):
      psr_model = []
      psr_model += [psr.residuals]
      psr_model += [ds.makegp_timing(psr, svd=True)]

      singlepsr_model = params_all.noise_model_obj(psr=psr, params=params)
      #add noise models
      if psr.name in params.noisemodel.keys():
        noise_model_dict_psr = params.noisemodel[psr.name]
      else:
        noise_model_dict_psr = params.universal
      for psp, option in noise_model_dict_psr.items():
        psr_model += getattr(singlepsr_model, psp)(option=option)

      psr_model_list += [ds.PulsarLikelihood(psr_model)]

      #Adding common signals
      pta_model = []
      common_gp_list = []
      global_gp_list = []
      for psp, option in params.common_signals.items():
     # [TO-DO] Add other common signals: CW, bwm, ...
     # [TO-DO] Add "spin_noise" or "dm_noise" which is in "universal: {}"
     # to common_gp_list
        if psp == "gwb":
          if option["orf"] == "none":
            common_gp_list += [getattr(allpsr_model, psp)(option=option)]
          else:
            global_gp_list += [getattr(allpsr_model, psp)(option=option)]
        else:
          raise ValueError('Only gwb is supported as common signal when using Discovery as your package.')
          
  pta = ds.ArrayLikelihood(psr_model_list, commongp=common_gp_list, globalgp=global_gp_list)

  pta[ii] = pta

  return pta


