"""
An extension of enterprise_warp.py to discovery:
- Creates a discovery PTA object, with methods to compute likelihoods and priors.

Likelihood backends (select in the parameter file via ``discovery_likelihood``):

* ``array``  → ``discovery.ArrayLikelihood`` (default when common/global
  signals are present).  Needs a non-empty ``commongp`` whenever ``globalgp``
  is used (library limitation).
* ``global`` → ``discovery.GlobalLikelihood``.  Supports HD ``global_gp`` alone;
  per-pulsar GPs stay inside each ``PulsarLikelihood``.  ``common_gp`` is
  split into per-pulsar GP pieces and attached the same way.
* omit / legacy ``discovery_array_mode: 0`` with no common signals → bare
  ``ArrayLikelihood`` of independent pulsar terms.
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


def _make_singlepsr_commongp(psr_gps, enable_coeff_index=False):
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
  if enable_coeff_index:
    idx = {}
    for ii, (gp, width) in enumerate(zip(compounds, ns)):
      gp_index = dict(getattr(gp, "index", {}))
      if gp_index:
        first_key = next(iter(gp_index.keys()))
        psr_name = first_key.split("_", 1)[0]
      else:
        psr_name = f"psr{ii}"
      key = f"{psr_name}_local_gp_coefficients({nmax})"
      idx[key] = slice(ii * nmax, (ii + 1) * nmax)
    comm_gp.index = idx
  else:
    comm_gp.index = {}
  return comm_gp


def _split_commongp_to_psr_gps(commongp):
  """
  Turn ``makecommongp_fourier`` output (VariableGP over all pulsars) into a
  list of single-pulsar VariableGPs for ``GlobalLikelihood`` / PulsarLikelihood.
  """
  if dmatrix is None:
    raise RuntimeError("discovery.matrix is required to split common_gp")

  getN = commongp.Phi.getN
  Fmats = list(commongp.F)
  pieces = []
  for i, F in enumerate(Fmats):
    def priorfunc(params, _i=i):
      return getN(params)[_i]
    priorfunc.params = list(getN.params)
    priorfunc.type = getattr(getN, "type", None)
    pieces.append(dmatrix.VariableGP(dmatrix.NoiseMatrix1D_var(priorfunc), F))
  return pieces


def _resolve_likelihood_mode(params):
  """
  Return ``'array'``, ``'global'``, or ``'none'``.

  Precedence:
    1. ``discovery_likelihood`` ∈ {array, global}
    2. legacy ``discovery_array_mode`` (True → array, False → none)
  """
  like = getattr(params, "discovery_likelihood", None)
  if like is not None and str(like).strip() != "":
    like = str(like).strip().lower()
    if like in ("array", "global"):
      return like
    raise ValueError(
        "discovery_likelihood must be 'array' or 'global', got {!r}".format(like)
    )
  if bool(getattr(params, "discovery_array_mode", False)):
    return "array"
  return "none"


def init_pta_discovery(params_all):
  """
  Initiate discovery PTA object.
  """
  ptas = dict.fromkeys(params_all.models)

  for ii, params in params_all.models.items():

    allpsr_model = params_all.noise_model_obj(psr=params_all.psrs,
                                              params=params)

    like_mode = _resolve_likelihood_mode(params)
    use_four_coef = bool(getattr(params, "four_coef", False))
    array_mode = like_mode == "array"
    global_mode = like_mode == "global"

    if like_mode == "none" and params.common_signals:
      raise RuntimeError(
          "Discovery common/global signals require discovery_likelihood=array|global "
          "(or legacy discovery_array_mode=1 for ArrayLikelihood)."
      )

    if global_mode and use_four_coef:
      raise RuntimeError(
          "four_coef / clogL is only supported with discovery_likelihood=array "
          "(GlobalLikelihood has no clogL path)."
      )

    # Build common/global signal objects first (model methods may register priors).
    common_gp_list = []
    global_gp_list = []
    for psp, option in params.common_signals.items():
      if "common_gp" in psp:
        common_gp_list += [getattr(allpsr_model, psp)(option=option)]
      elif "global_gp" in psp:
        global_gp_list += [getattr(allpsr_model, psp)(option=option)]
      else:
        raise ValueError(
            "Only common_gp and global_gp are supported as common signals when "
            "using Discovery. Add deterministic common signals as single-pulsar "
            "signals in model files."
        )

    # Per common_gp: list of length n_psr of single-pulsar VariableGPs (global mode).
    common_gp_per_psr = []
    if global_mode and common_gp_list:
      for cgp in common_gp_list:
        common_gp_per_psr.append(_split_commongp_to_psr_gps(cgp))

    psr_model_list = []
    psr_gp_terms = [] if array_mode else None

    for pnum, psr in enumerate(params_all.psrs):
      psr_terms = [psr.residuals, ds.makegp_timing(psr, svd=True)]
      gp_terms = []

      singlepsr_model = params_all.noise_model_obj(psr=psr, params=params)
      if psr.name in params.noisemodel.keys():
        noise_model_dict_psr = params.noisemodel[psr.name]
      else:
        noise_model_dict_psr = params.to_remaining_psrs
      for psp, option in {**noise_model_dict_psr, **params.to_each_psr}.items():
        component = getattr(singlepsr_model, psp)(option=option)
        if array_mode and dmatrix is not None and isinstance(component, dmatrix.VariableGP):
          gp_terms.append(component)
        else:
          psr_terms.append(component)

      if global_mode and common_gp_per_psr:
        for pieces in common_gp_per_psr:
          psr_terms.append(pieces[pnum])

      psr_model_list += [ds.PulsarLikelihood(psr_terms)]
      if array_mode:
        psr_gp_terms.append(gp_terms)

    if not global_gp_list:
      global_gp_arg = None
    elif len(global_gp_list) == 1:
      global_gp_arg = global_gp_list[0]
    else:
      global_gp_arg = global_gp_list

    if global_mode:
      print("[discovery_warp] using GlobalLikelihood "
            "(global_gp alone is OK; common_gp attached per pulsar)")
      pta = ds.GlobalLikelihood(psr_model_list, globalgp=global_gp_arg)
    elif array_mode:
      commongp_terms = list(common_gp_list)
      local_cgp = _make_singlepsr_commongp(
          psr_gp_terms, enable_coeff_index=use_four_coef)
      if local_cgp is not None:
        commongp_terms.insert(0, local_cgp)

      if not commongp_terms:
        common_gp_arg = None
      elif len(commongp_terms) == 1:
        common_gp_arg = commongp_terms[0]
      else:
        common_gp_arg = commongp_terms

      if global_gp_arg is not None and common_gp_arg is None:
        raise RuntimeError(
            "ArrayLikelihood does not support global_gp without a commongp. "
            "Either add spin_noise/common_gp, or set discovery_likelihood: global."
        )

      print("[discovery_warp] using ArrayLikelihood")
      pta = ds.ArrayLikelihood(
          psr_model_list,
          commongp=common_gp_arg,
          globalgp=global_gp_arg,
          decenter=use_four_coef,
      )
    else:
      pta = ds.ArrayLikelihood(psr_model_list, decenter=use_four_coef)

    if use_four_coef:
      pta.logL = pta.clogL

    ptas[ii] = pta

  return ptas
