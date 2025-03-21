"""
An extension of enterprise_warp.py to discovery:
- Creates a discovery PTA object, with methods to compute likelihoods and priors.
"""

import warnings

import numpy as np

try:
  import discovery as ds
except Exception as ex:
  print(ex)
  warnings.warn("discovery is not available")

def init_pta_discovery(params_all):
  """
  Initiate discovery PTA object.
  """
  ptas = dict.fromkeys(params_all.models)

  # Loop over models in a parameter file: {0}, {1}, ...
  for ii, params in params_all.models.items():

    allpsr_model = params_all.noise_model_obj(psr=params_all.psrs,
                                              params=params)

    models = list()
    from_par_file = list()

    psr_model_list = []
    pta_model = []
    common_gp_list = []
    global_gp_list = []

    # Loop over pulsars
    for pnum, psr in enumerate(params_all.psrs):
      psr_model = []
      psr_model += [psr.residuals]
      psr_model += [ds.makegp_timing(psr, svd=True)]

      # Pulsar-specific noise models
      singlepsr_model = params_all.noise_model_obj(psr=psr, params=params)
      if psr.name in params.noisemodel.keys():
        noise_model_dict_psr = params.noisemodel[psr.name]
      else:
        noise_model_dict_psr = params.to_remaining_psrs
      for psp, option in {**noise_model_dict_psr,**params.to_each_psr}.items():
        psr_model += [getattr(singlepsr_model, psp)(option=option)]

      psr_model_list += [ds.PulsarLikelihood(psr_model)]

      # Common signals in all pulsars
      for psp, option in params.common_signals.items():
        if "common_gp" in psp:
          common_gp_list += [getattr(allpsr_model, psp)(option=option)]
        elif "global_gp" in psp:
          global_gp_list += [getattr(allpsr_model, psp)(option=option)]
        else:
          raise ValueError('Only common_gp and global_gp are supported as common signal when using Discovery as your package.')

  if not global_gp_list: global_gp_list = None
  if not common_gp_list: common_gp_list = None
  pta = ds.ArrayLikelihood(psr_model_list, commongp=common_gp_list, globalgp=global_gp_list)

  ptas[ii] = pta

  return ptas


