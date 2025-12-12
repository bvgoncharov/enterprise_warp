"""
An extension of enterprise_warp.py to discovery:
- Creates a discovery PTA object, with methods to compute likelihoods and priors.
"""

import warnings

import numpy as np

try:
  import discovery as ds
  from discovery import matrix as dmatrix
except Exception as ex:
  print(ex)
  warnings.warn("discovery is not available")
  dmatrix = None


def _make_singlepsr_commongp(psr_gps):
  """
  Merge per-pulsar VariableGP lists into a single commongp compatible with ArrayLikelihood.
  """
  if dmatrix is None:
    return None

  if not psr_gps or not all(psr_gps):
    return None

  compounds = [dmatrix.CompoundGP(gps) if len(gps) > 1 else gps[0] for gps in psr_gps]

  priors = [gp.Phi.getN for gp in compounds]
  ns = [gp.F.shape[1] for gp in compounds]
  nmax = max(ns)

  def prior(params):
    yp = dmatrix.jnp.full((len(priors), nmax), 1e-40)
    for ii, (p, width) in enumerate(zip(priors, ns)):
      yp = yp.at[ii, :width].set(p(params))
    return yp
  prior.params = sorted(set(par for p in priors for par in p.params))

  Fs = [np.pad(gp.F, [(0, 0), (0, nmax - width)]) for gp, width in zip(compounds, ns)]

  comm_gp = dmatrix.VariableGP(dmatrix.VectorNoiseMatrix1D_var(prior), Fs)
  comm_gp.index = {}
  return comm_gp

def init_pta_discovery(params_all):
  """
  Initiate discovery PTA object.
  """
  ptas = dict.fromkeys(params_all.models)

  # Loop over models in a parameter file: {0}, {1}, ...
  for ii, params in params_all.models.items():

    allpsr_model = params_all.noise_model_obj(psr=params_all.psrs,
                                              params=params)

    array_mode = bool(getattr(params, "discovery_array_mode", False))

    if not array_mode and params.common_signals:
      raise RuntimeError("Discovery common/global signals require discovery_array_mode=1.")

    psr_model_list = []
    psr_gp_terms = [] if array_mode else None

    # Loop over pulsars to build per-pulsar likelihood pieces
    for pnum, psr in enumerate(params_all.psrs):
      psr_terms = [psr.residuals, ds.makegp_timing(psr, svd=True)]
      gp_terms = []

      # Pulsar-specific noise models
      singlepsr_model = params_all.noise_model_obj(psr=psr, params=params)
      if psr.name in params.noisemodel.keys():
        noise_model_dict_psr = params.noisemodel[psr.name]
      else:
        noise_model_dict_psr = params.to_remaining_psrs
      for psp, option in {**noise_model_dict_psr,**params.to_each_psr}.items():
        component = getattr(singlepsr_model, psp)(option=option)
        if array_mode and dmatrix is not None and isinstance(component, dmatrix.VariableGP):
          gp_terms.append(component)
        else:
          psr_terms.append(component)

      psr_model_list += [ds.PulsarLikelihood(psr_terms)]
      if array_mode:
        psr_gp_terms.append(gp_terms)

    # Build common/global signals once, outside the pulsar loop
    common_gp_list = []
    global_gp_list = []
    for psp, option in params.common_signals.items():
      if "common_gp" in psp:
        common_gp_list += [getattr(allpsr_model, psp)(option=option)]
      elif "global_gp" in psp:
        global_gp_list += [getattr(allpsr_model, psp)(option=option)]
      else:
        raise ValueError('Only common_gp and global_gp are supported as common signal when using Discovery as your package. Add deterministic common signals as single-pulsar signals for discovery in model files.')

    commongp_terms = list(common_gp_list)
    if array_mode:
      local_cgp = _make_singlepsr_commongp(psr_gp_terms)
      if local_cgp is not None:
        commongp_terms.insert(0, local_cgp)

    if not global_gp_list:
      global_gp_arg = None
    elif len(global_gp_list) == 1:
      global_gp_arg = global_gp_list[0]
    else:
      global_gp_arg = global_gp_list

    if not commongp_terms:
      common_gp_arg = None
    elif len(commongp_terms) == 1:
      common_gp_arg = commongp_terms[0]
    else:
      common_gp_arg = commongp_terms

    if array_mode:
      pta = ds.ArrayLikelihood(psr_model_list, commongp=common_gp_arg, globalgp=global_gp_arg)
    else:
      pta = ds.ArrayLikelihood(psr_model_list)

    ptas[ii] = pta

  return ptas
