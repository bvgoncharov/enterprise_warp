"""
Gaussian-process coefficient and time-series reconstruction for discovery runs.

This module reconstructs the posterior mean coefficients of Fourier-basis GP
terms from an enterprise_warp parameter file and a completed output chain.  It
uses the same finite-GP basis and prior objects as discovery, while building the
white-noise covariance with the EFAC/EQUAD/ECORR helper already used by
hasasia_warp.
"""

import csv
import glob
import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import scipy.linalg as sl

from . import enterprise_warp
from .hasasia_warp import (RunLog, build_white_noise_model, posterior_mode,
                           _safe_name)
from .results import (DiscoveryWarpResult, EnterpriseWarpResult, ResultsParser,
                      normalize_result_args)


TIMING_PRIOR_VARIANCE = 1.0e40
DEFAULT_COMMON_GAMMA = 13.0 / 3.0
YR_SEC = 365.25 * 24.0 * 3600.0


class GPReconstructionParser(ResultsParser):
  """Standalone parser for GP reconstruction runs."""

  def __init__(self):
    super(GPReconstructionParser, self).__init__()
    self.parser.add_option("--num", default=0, type=int,
                           help="Realization/result-folder number.")
    self.parser.add_option("--gp_model", default=0, type=int,
                           help="Model block id.")
    self.parser.add_option("--gp_terms", default="all", type=str,
                           help="Comma-separated GP terms: all, red_noise, "
                                "crn, gw.")
    self.parser.add_option("--gp_psr", default="all", type=str,
                           help="Pulsar selector: all, integer index, exact "
                                "name, or unique substring.")
    self.parser.add_option("--gp_grid_points", default=600, type=int,
                           help="Dense time-grid points per pulsar.")
    self.parser.add_option("--gp_plot_toas", default=1, type=int,
                           help="Overplot timing residuals and TOA errors on "
                                "reconstruction plots (1/0).")
    self.parser.add_option("--gp_outdir", default=None, type=str,
                           help="Optional output directory. Defaults under the "
                                "loaded result directory.")

  def parse_args(self):
    opts = super(GPReconstructionParser, self).parse_args()
    opts.drop = 0
    opts.extra_model_terms = 'None'
    opts.realization = opts.num
    return opts


class ChainParameterProvider(object):
  """Posterior-mode values with noisefile and fixed-value fallbacks."""

  def __init__(self, pars=None, chain_burn=None, noisedict=None, defaults=None):
    self.pars = np.asarray([] if pars is None else pars, dtype=str)
    self.chain_burn = chain_burn
    self.noisedict = {} if noisedict is None else dict(noisedict)
    self.defaults = {} if defaults is None else dict(defaults)
    self._cache = {}

  def samples(self, par_name):
    if self.chain_burn is None:
      return None
    matches = np.where(self.pars == par_name)[0]
    if matches.size == 0:
      return None
    samples = np.asarray(self.chain_burn[:, matches[0]], dtype=float)
    samples = samples[np.isfinite(samples)]
    return samples if samples.size else None

  def mode(self, par_name):
    if par_name in self._cache:
      return self._cache[par_name]
    samples = self.samples(par_name)
    value = None if samples is None else posterior_mode(samples)
    if value is None and par_name in self.noisedict:
      value = self.noisedict[par_name]
    if value is None and par_name in self.defaults:
      value = self.defaults[par_name]
    self._cache[par_name] = value
    return value

  def vector_mode(self, base_name, length):
    values = []
    for ii in range(int(length)):
      value = None
      for par_name in ['{}[{}]'.format(base_name, ii),
                       '{}_{}'.format(base_name, ii)]:
        value = self.mode(par_name)
        if value is not None:
          break
      if value is None:
        return None
      values.append(value)
    return np.asarray(values, dtype=float)

  def parameter_dict(self, required_names):
    params = {}
    missing = []
    for name in required_names:
      parsed = _parse_vector_parameter_name(name)
      if parsed is None:
        value = self.mode(name)
        if value is None and name.endswith('_gamma') and (
            name.startswith('crn_') or name.startswith('gw_')):
          value = DEFAULT_COMMON_GAMMA
        if value is None:
          missing.append(name)
        else:
          params[name] = value
        continue

      base, length = parsed
      vector = self.vector_mode(base, length)
      if vector is None:
        missing.append(name)
      else:
        params[name] = vector
        params[base] = vector
    if missing:
      raise ValueError('Missing parameter values required for GP '
                       'reconstruction: {}'.format(', '.join(missing)))
    return params


class GPDescriptor(object):
  """Description of one coefficient block in the reconstruction system."""

  def __init__(self, term, psr_name, psr_index, basis, prior_group,
               prior_component, local_index=None, output=True, kind='gp',
               tspan=None):
    self.term = term
    self.psr_name = psr_name
    self.psr_index = int(psr_index)
    self.basis = np.asarray(basis, dtype=float)
    self.prior_group = prior_group
    self.prior_component = prior_component
    self.local_index = local_index
    self.output = bool(output)
    self.kind = kind
    self.tspan = tspan
    self.slice = None


def _parse_vector_parameter_name(name):
  match = re.match(r'^(.*)\((\d+)\)$', str(name))
  if not match:
    return None
  return match.group(1), int(match.group(2))


def _as_numpy(value):
  return np.asarray(value, dtype=float)


def _toas_for_output(toas):
  toas = np.asarray(toas, dtype=float)
  if np.nanmedian(np.abs(toas)) > 1.0e6:
    return toas / 86400.0
  return toas


def _weighted_rms(values, sigma):
  values = np.asarray(values, dtype=float)
  sigma = np.asarray(sigma, dtype=float)
  weights = np.where(sigma > 0.0, 1.0 / sigma**2, 0.0)
  if not np.any(weights > 0.0):
    return np.nan
  mean = np.sum(weights * values) / np.sum(weights)
  return float(np.sqrt(np.sum(weights * (values - mean)**2) / np.sum(weights)))


def _selection_from_option(value):
  if value is None:
    return []
  if str(value).lower() == 'all':
    return ['red_noise', 'crn', 'gw']
  selected = [item.strip() for item in str(value).split(',') if item.strip()]
  allowed = {'red_noise', 'crn', 'gw'}
  unknown = sorted(set(selected) - allowed)
  if unknown:
    raise ValueError('Unknown --gp_terms entries: {}'.format(
        ', '.join(unknown)))
  return selected


def detect_gp_terms(model_params, psrs):
  """Return model GP inventory for RN, CRN, and HD GWB terms."""
  inventory = []
  common = getattr(model_params, 'common_signals', {}) or {}
  for key, option in common.items():
    if 'common_gp' in key or key == 'crn':
      inventory.append({'term': 'crn', 'model_key': key,
                        'option': option, 'psr_name': 'all'})
    elif 'global_gp' in key or key == 'gw' or key == 'gwb':
      orf = option.get('orf') if isinstance(option, dict) else None
      inventory.append({'term': 'gw', 'model_key': key,
                        'option': option, 'psr_name': 'all',
                        'orf': orf})

  for psr in psrs:
    block = _single_pulsar_noise_block(model_params, psr)
    if 'spin_noise' in block:
      inventory.append({'term': 'red_noise', 'model_key': 'spin_noise',
                        'option': block['spin_noise'], 'psr_name': psr.name})
  return inventory


def _single_pulsar_noise_block(model_params, psr):
  noisemodel = getattr(model_params, 'noisemodel', {}) or {}
  block = noisemodel.get(psr.name,
                         getattr(model_params, 'to_remaining_psrs', {}) or {})
  merged = dict(block)
  merged.update(getattr(model_params, 'to_each_psr', {}) or {})
  return merged


def _option_nfreq(params_all, model_params, option, psr=None):
  if not isinstance(option, dict):
    match = re.search(r'(\d+)_nfreqs', str(option))
    return None if match is None else int(match.group(1))
  return params_all.noise_model_obj(psr=psr, params=model_params)\
                   .option_nfreqs(option, sel_func_name=None)


def _basis_from_tspan(toas, ncoeff, tspan):
  components = int(ncoeff) // 2
  freqs = np.arange(1, components + 1, dtype=float) / float(tspan)
  basis = np.zeros((len(toas), 2 * components), dtype=float)
  for ii, freq in enumerate(freqs):
    basis[:, 2 * ii] = np.sin(2.0 * np.pi * freq * toas)
    basis[:, 2 * ii + 1] = np.cos(2.0 * np.pi * freq * toas)
  return basis


def _precision_from_phi(phi, params):
  inv_func = phi.make_inv()
  precision, _ = inv_func(params)
  return _as_numpy(precision)


def _global_precision_from_component(component, params):
  inv_func = component.Phi_inv or component.Phi.make_inv()
  precision, _ = inv_func(params)
  return _as_numpy(precision)


def _block_diag_precision(precision):
  precision = _as_numpy(precision)
  if precision.ndim == 1:
    return np.diag(precision)
  if precision.ndim == 2:
    return precision
  if precision.ndim == 3:
    return sl.block_diag(*precision)
  raise ValueError('Unsupported precision shape {}'.format(precision.shape))


def solve_linear_gp_system(descriptors, residuals_by_psr, white_models,
                           prior_params):
  """Solve for coefficient posterior mean and covariance."""
  ncoef = 0
  for desc in descriptors:
    width = desc.basis.shape[1]
    desc.slice = slice(ncoef, ncoef + width)
    ncoef += width
  if ncoef == 0:
    raise ValueError('No GP/timing coefficient blocks were built.')

  tnt = np.zeros((ncoef, ncoef), dtype=float)
  tnr = np.zeros(ncoef, dtype=float)
  for psr_index, residuals in residuals_by_psr.items():
    psr_desc = [desc for desc in descriptors if desc.psr_index == psr_index]
    if not psr_desc:
      continue
    basis = np.hstack([desc.basis for desc in psr_desc])
    inv_basis = white_models[psr_index].apply_inverse(basis)
    inv_residuals = white_models[psr_index].apply_inverse(residuals)
    local_tnt = basis.T.dot(inv_basis)
    local_tnr = basis.T.dot(inv_residuals)
    cols = np.concatenate([np.arange(desc.slice.start, desc.slice.stop)
                           for desc in psr_desc])
    tnt[np.ix_(cols, cols)] += local_tnt
    tnr[cols] += local_tnr

  phiinv = np.zeros((ncoef, ncoef), dtype=float)
  for desc in descriptors:
    if desc.kind == 'timing':
      phiinv[desc.slice, desc.slice] = np.eye(desc.basis.shape[1]) / \
                                       TIMING_PRIOR_VARIANCE

  groups = {}
  for desc in descriptors:
    if desc.kind != 'gp':
      continue
    groups.setdefault(desc.prior_group, []).append(desc)

  for group_desc in groups.values():
    component = group_desc[0].prior_component
    if hasattr(component, 'Fs'):
      precision = _global_precision_from_component(component, prior_params)
      block = _block_diag_precision(precision)
      if block.shape[0] != sum(desc.basis.shape[1] for desc in group_desc):
        local_cols = np.concatenate([
            np.arange(desc.local_index.start, desc.local_index.stop)
            for desc in group_desc])
        block = block[np.ix_(local_cols, local_cols)]
    else:
      precision = _precision_from_phi(component.Phi, prior_params)
      is_vector_common = (
          precision.ndim == 2 and hasattr(component, 'F') and
          isinstance(component.F, (list, tuple)) and
          precision.shape[0] == len(component.F))
      if is_vector_common:
        rows = np.asarray([desc.psr_index for desc in group_desc], dtype=int)
        block = sl.block_diag(*[np.diag(row) for row in precision[rows]])
      else:
        block = _block_diag_precision(precision)

    cols = np.concatenate([np.arange(desc.slice.start, desc.slice.stop)
                           for desc in group_desc])
    if block.shape != (cols.size, cols.size):
      raise ValueError('Prior precision for {} has shape {}, expected {}.'
                       .format(group_desc[0].term, block.shape,
                               (cols.size, cols.size)))
    phiinv[np.ix_(cols, cols)] += block

  cf = sl.cho_factor(tnt + phiinv, lower=True, check_finite=False)
  covariance = sl.cho_solve(cf, np.eye(ncoef), check_finite=False)
  mean = covariance.dot(tnr)
  return mean, covariance


class GPReconstructionMixin(object):
  """Mixin that adds GP reconstruction to result loaders."""

  def main_pipeline(self):
    if not hasattr(self, 'params'):
      raise ValueError('gp_reconstruction requires --result to be a parameter '
                       'file.')
    if not getattr(self.opts, 'discovery', 0):
      raise NotImplementedError('gp_reconstruction currently supports '
                                'discovery runs only.')

    self._init_gp_pulsars()
    self.gp_result_dir = self._get_gp_result_dir()
    self.run_dir = self._new_run_dir()
    self.log = RunLog(os.path.join(self.run_dir, 'run.log'))
    self.log.write('GP reconstruction run')
    self.log.write('Result directory: {}'.format(self.gp_result_dir))
    self.log.write('Output directory: {}'.format(self.run_dir))

    self._load_result_chain()
    model_params, model_id = self._get_model_params()
    selected_terms = _selection_from_option(getattr(self.opts, 'gp_terms',
                                                    'all'))
    selected_psrs = self._select_psrs(getattr(self.opts, 'gp_psr', 'all'))
    selected_psr_names = [psr.name for psr in selected_psrs]
    self.log.write('Selected model id: {}'.format(model_id))
    self.log.write('Selected terms: {}'.format(', '.join(selected_terms)))
    self.log.write('Selected pulsars: {}'.format(', '.join(selected_psr_names)))

    inventory = detect_gp_terms(model_params, self.params.psrs)
    self.log.write('Detected GP inventory: {}'.format(json.dumps(inventory,
                                                                 sort_keys=True)))
    descriptors, residuals_by_psr, white_models, provider = \
        self._build_reconstruction_problem(model_params, selected_terms,
                                           selected_psr_names)
    prior_param_names = sorted(set(
        par for desc in descriptors if desc.kind == 'gp'
        for par in getattr(desc.prior_component.Phi, 'params', [])
    ) | set(
        par for desc in descriptors
        if desc.kind == 'gp' and hasattr(desc.prior_component, 'Phi_inv')
        and desc.prior_component.Phi_inv is not None
        for par in getattr(desc.prior_component.Phi_inv, 'params', [])
    ))
    prior_params = provider.parameter_dict(prior_param_names)
    mean, covariance = solve_linear_gp_system(
        descriptors, residuals_by_psr, white_models, prior_params)
    self._write_outputs(model_id, inventory, descriptors, residuals_by_psr,
                        mean, covariance, selected_terms, selected_psr_names)

  def _init_gp_pulsars(self):
    if self.params.opts is None:
      self.params.opts = self.opts
    for attr, default in [('drop', 0), ('extra_model_terms', 'None'),
                          ('num', getattr(self.opts, 'num', 0))]:
      if not hasattr(self.params.opts, attr):
        setattr(self.params.opts, attr, default)
    self.params.init_pulsars()
    self.params.clone_all_params_to_models()

  def _get_gp_result_dir(self):
    if self.params.array_analysis:
      return os.path.abspath(self.outdir_all)
    return os.path.abspath(getattr(self.params, 'output_dir', self.outdir_all))

  def _new_run_dir(self):
    base = getattr(self.opts, 'gp_outdir', None)
    if base is None:
      base = os.path.join(self.gp_result_dir, 'gp_reconstruction')
    run_dir = os.path.join(
        base, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

  def _load_result_chain(self):
    self.psr_dir = ''
    self.outdir = self.gp_result_dir
    self.get_pars()
    self.get_chain_file_name()
    if self.chain_file is None:
      self.log.write('No posterior chain found; using noisefile/fixed '
                     'fallbacks only.')
      self.chain = None
      self.chain_burn = None
      return False
    success = self.load_chains()
    if not success:
      self.chain = None
      self.chain_burn = None
    return success

  def _get_model_params(self):
    model_id = int(getattr(self.opts, 'gp_model', 0))
    if model_id not in self.params.models:
      raise ValueError('Model block {{{}}} not found.'.format(model_id))
    return self.params.models[model_id], model_id

  def _select_psrs(self, selector):
    selector = str(selector)
    if selector.lower() == 'all':
      return list(self.params.psrs)
    if selector.isdigit():
      idx = int(selector)
      if idx < 0 or idx >= len(self.params.psrs):
        raise IndexError('gp_psr index {} outside 0..{}'.format(
            idx, len(self.params.psrs) - 1))
      return [self.params.psrs[idx]]
    matches = [psr for psr in self.params.psrs if psr.name == selector]
    if not matches:
      matches = [psr for psr in self.params.psrs if selector in psr.name]
    if len(matches) != 1:
      raise ValueError('Pulsar selector {} matched {}'.format(
          selector, [psr.name for psr in matches]))
    return matches

  def _provider(self):
    defaults = {}
    for key in ['crn_gamma', 'gw_gamma']:
      defaults[key] = DEFAULT_COMMON_GAMMA
    return ChainParameterProvider(
        pars=getattr(self, 'pars', None),
        chain_burn=getattr(self, 'chain_burn', None),
        noisedict=getattr(self.params, 'noisedict', {}),
        defaults=defaults)

  def _build_reconstruction_problem(self, model_params, selected_terms,
                                    selected_psr_names):
    try:
      import discovery as ds
      from discovery import matrix as dmatrix
    except Exception as exc:
      raise RuntimeError('discovery is required for gp_reconstruction') from exc
    has_path = os.environ.get('HAS')
    if has_path is not None and has_path not in sys.path:
      sys.path.insert(0, has_path)
    try:
      import hasasia.sensitivity as hsen
    except Exception as exc:
      raise RuntimeError('hasasia is required to build ECORR white-noise '
                         'blocks for gp_reconstruction') from exc

    provider = self._provider()
    descriptors = []
    residuals_by_psr = {}
    white_models = {}

    allpsr_model = self.params.noise_model_obj(psr=self.params.psrs,
                                               params=model_params)
    common_components = {}
    global_components = {}
    for psp, option in (getattr(model_params, 'common_signals', {}) or {}).items():
      component = getattr(allpsr_model, psp)(option=option)
      if isinstance(component, dmatrix.VariableGP):
        common_components['crn'] = (component, option, psp)
      elif isinstance(component, dmatrix.GlobalVariableGP):
        global_components['gw'] = (component, option, psp)
      else:
        raise ValueError('Unsupported common signal {} of type {}'
                         .format(psp, type(component)))

    deterministic = {ii: [] for ii, _ in enumerate(self.params.psrs)}
    psr_red_components = {}
    for psr_index, psr in enumerate(self.params.psrs):
      singlepsr_model = self.params.noise_model_obj(psr=psr,
                                                    params=model_params)
      for psp, option in _single_pulsar_noise_block(model_params, psr).items():
        component = getattr(singlepsr_model, psp)(option=option)
        if isinstance(component, dmatrix.VariableGP):
          gpname = getattr(component, 'gpname', None)
          if gpname == 'red_noise':
            psr_red_components[psr.name] = (component, option, psp)
          elif gpname == 'ecorrGP' or psp == 'ecorr':
            continue
          else:
            raise ValueError('Unsupported GP term {} for {}. v1 supports '
                             'red_noise, crn, and gw only.'
                             .format(gpname or psp, psr.name))
        elif isinstance(component, dmatrix.Kernel):
          continue
        elif isinstance(component, dmatrix.ConstantGP) and psp == 'ecorr':
          continue
        elif callable(component):
          deterministic[psr_index].append(component)
        else:
          raise ValueError('Unsupported model component {} for {}: {}'
                           .format(psp, psr.name, type(component)))

    for psr_index, psr in enumerate(self.params.psrs):
      if psr.name not in selected_psr_names:
        continue
      noise = {key: val for key, val in getattr(self.params, 'noisedict',
                                                {}).items()
               if key.startswith(psr.name)}
      white_models[psr_index] = build_white_noise_model(psr, noise, hsen=hsen,
                                                        log=self.log)
      params_needed = sorted(set(par for delay in deterministic[psr_index]
                                 for par in getattr(delay, 'params', [])))
      delay_params = provider.parameter_dict(params_needed)
      residuals = np.asarray(psr.residuals, dtype=float).copy()
      for delay in deterministic[psr_index]:
        residuals -= np.asarray(delay(delay_params), dtype=float)
      residuals_by_psr[psr_index] = residuals

      designmatrix = getattr(psr, 'Mmat', getattr(psr, 'designmatrix', None))
      if designmatrix is None:
        raise ValueError('Pulsar {} has neither Mmat nor designmatrix.'
                         .format(psr.name))
      descriptors.append(GPDescriptor(
          term='timing', psr_name=psr.name, psr_index=psr_index,
          basis=np.asarray(designmatrix, dtype=float),
          prior_group='timing_{}'.format(psr.name), prior_component=None,
          output=False, kind='timing'))

      if psr.name in psr_red_components:
        component, option, _ = psr_red_components[psr.name]
        descriptors.append(GPDescriptor(
            term='red_noise', psr_name=psr.name, psr_index=psr_index,
            basis=np.asarray(component.F, dtype=float),
            prior_group='red_noise_{}'.format(psr.name),
            prior_component=component,
            output='red_noise' in selected_terms,
            tspan=float(np.max(psr.toas) - np.min(psr.toas))))

      if 'crn' in common_components:
        component, option, _ = common_components['crn']
        local = _component_slice(component, psr.name, 'crn')
        basis = np.asarray(component.F[psr_index], dtype=float)
        descriptors.append(GPDescriptor(
            term='crn', psr_name=psr.name, psr_index=psr_index,
            basis=basis, prior_group='crn', prior_component=component,
            local_index=local, output='crn' in selected_terms,
            tspan=float(self.params.Tspan)))

      if 'gw' in global_components:
        component, option, _ = global_components['gw']
        local = _component_slice(component, psr.name, 'gw')
        basis = np.asarray(component.Fs[psr_index], dtype=float)
        descriptors.append(GPDescriptor(
            term='gw', psr_name=psr.name, psr_index=psr_index,
            basis=basis, prior_group='gw', prior_component=component,
            local_index=local, output='gw' in selected_terms,
            tspan=float(self.params.Tspan)))

    if not any(desc.output for desc in descriptors):
      raise ValueError('No selected GP terms found for selected pulsars.')
    return descriptors, residuals_by_psr, white_models, provider

  def _write_outputs(self, model_id, inventory, descriptors, residuals_by_psr,
                     mean, covariance, selected_terms, selected_psr_names):
    try:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
    except Exception as exc:
      raise RuntimeError('matplotlib is required for gp_reconstruction plots') \
          from exc

    settings = {
        'result': str(self.opts.result),
        'result_dir': self.gp_result_dir,
        'selected_model': int(model_id),
        'selected_terms': selected_terms,
        'selected_pulsars': selected_psr_names,
        'grid_points': int(getattr(self.opts, 'gp_grid_points', 600)),
        'plot_toas': int(getattr(self.opts, 'gp_plot_toas', 1)),
        'inventory': inventory,
    }
    with open(os.path.join(self.run_dir, 'settings.json'), 'w') as fout:
      json.dump(settings, fout, indent=2, sort_keys=True)

    summary_rows = []
    by_psr = {}
    for desc in descriptors:
      by_psr.setdefault(desc.psr_name, []).append(desc)

    for psr_name, psr_desc in sorted(by_psr.items()):
      psr_index = psr_desc[0].psr_index
      psr = self.params.psrs[psr_index]
      residuals = residuals_by_psr.get(psr_index)
      if residuals is None:
        continue
      toaerrs = np.asarray(psr.toaerrs, dtype=float)
      toas = np.asarray(psr.toas, dtype=float)
      toas_out = _toas_for_output(toas)
      original_wrms = _weighted_rms(residuals, toaerrs)
      total_gp = np.zeros_like(residuals)

      for desc in psr_desc:
        if not desc.output:
          continue
        coeff_mean = mean[desc.slice]
        coeff_cov = covariance[desc.slice, desc.slice]
        coeff_sigma = np.sqrt(np.maximum(np.diag(coeff_cov), 0.0))
        gp_mean = desc.basis.dot(coeff_mean)
        gp_var = np.einsum('ij,jk,ik->i', desc.basis, coeff_cov, desc.basis)
        gp_sigma = np.sqrt(np.maximum(gp_var, 0.0))
        total_gp += gp_mean

        prefix = '{}_{}'.format(_safe_name(psr_name), _safe_name(desc.term))
        coeff_path = os.path.join(self.run_dir, prefix + '_coefficients.txt')
        coeff_table = np.column_stack([
            np.arange(coeff_mean.size, dtype=int), coeff_mean, coeff_sigma])
        np.savetxt(coeff_path, coeff_table,
                   header='index coefficient_mean coefficient_sigma')

        toa_path = os.path.join(self.run_dir, prefix + '_toas.txt')
        toa_table = np.column_stack([toas_out, residuals, toaerrs, gp_mean,
                                     gp_sigma, residuals - gp_mean])
        np.savetxt(toa_path, toa_table,
                   header='toa_mjd residual_s toaerr_s gp_mean_s gp_sigma_s '
                          'residual_minus_gp_s')

        dense_toas = np.linspace(float(toas.min()), float(toas.max()),
                                 int(getattr(self.opts, 'gp_grid_points', 600)))
        dense_basis = _basis_from_tspan(dense_toas, desc.basis.shape[1],
                                        desc.tspan)
        dense_mean = dense_basis.dot(coeff_mean)
        dense_var = np.einsum('ij,jk,ik->i', dense_basis, coeff_cov,
                              dense_basis)
        dense_sigma = np.sqrt(np.maximum(dense_var, 0.0))
        dense_out = _toas_for_output(dense_toas)
        dense_path = os.path.join(self.run_dir, prefix + '_dense.txt')
        dense_table = np.column_stack([
            dense_out, dense_mean, dense_sigma,
            dense_mean - dense_sigma, dense_mean + dense_sigma])
        np.savetxt(dense_path, dense_table,
                   header='toa_mjd gp_mean_s gp_sigma_s gp_lower68_s '
                          'gp_upper68_s')

        plot_path = os.path.join(self.run_dir, prefix + '.png')
        plt.figure(figsize=(9, 4.5))
        if int(getattr(self.opts, 'gp_plot_toas', 1)):
          plt.errorbar(toas_out, residuals * 1.0e6, yerr=toaerrs * 1.0e6,
                       fmt='.', color='0.65', ecolor='0.85', ms=3,
                       elinewidth=0.5, label='Residuals')
        plt.plot(dense_out, dense_mean * 1.0e6, color='C0',
                 lw=1.5, label=desc.term)
        plt.fill_between(dense_out,
                         (dense_mean - dense_sigma) * 1.0e6,
                         (dense_mean + dense_sigma) * 1.0e6,
                         color='C0', alpha=0.25, linewidth=0.0)
        plt.xlabel('TOA [MJD]')
        plt.ylabel('Residual [us]')
        plt.title('{} {}'.format(psr_name, desc.term))
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()

        sub_wrms = _weighted_rms(residuals - gp_mean, toaerrs)
        summary_rows.append({
            'pulsar': psr_name,
            'term': desc.term,
            'original_wrms_s': original_wrms,
            'subtracted_wrms_s': sub_wrms,
            'coefficients': coeff_mean.size,
        })

      if np.any(total_gp != 0.0):
        summary_rows.append({
            'pulsar': psr_name,
            'term': 'all_selected',
            'original_wrms_s': original_wrms,
            'subtracted_wrms_s': _weighted_rms(residuals - total_gp, toaerrs),
            'coefficients': '',
        })

    summary_path = os.path.join(self.run_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as fout:
      fieldnames = ['pulsar', 'term', 'original_wrms_s',
                    'subtracted_wrms_s', 'coefficients']
      writer = csv.DictWriter(fout, fieldnames=fieldnames)
      writer.writeheader()
      for row in summary_rows:
        writer.writerow(row)
    self.log.write('Wrote GP reconstruction outputs to {}'.format(
        self.run_dir))


def _component_slice(component, psr_name, term):
  key_part = '{}_{}_coefficients'.format(psr_name, term)
  for key, value in getattr(component, 'index', {}).items():
    if key.startswith(key_part):
      return value
  raise ValueError('Could not find coefficient slice for {} in {} index.'
                   .format(key_part, term))


class GPReconstructionDiscoveryWarp(GPReconstructionMixin,
                                    DiscoveryWarpResult):
  pass


class GPReconstructionEnterpriseWarp(GPReconstructionMixin,
                                     EnterpriseWarpResult):
  pass


def _load_custom_model(opts):
  if opts.custom_models is not None and opts.custom_models_py is not None:
    spec = importlib.util.spec_from_file_location("custom_models_obj",
                                                  opts.custom_models_py)
    cmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmod)
    return cmod.__dict__[opts.custom_models]
  if opts.custom_models is None and opts.custom_models_py is None:
    return None
  raise ValueError('Please set both --custom_models and --custom_models_py')


def main():
  opts = GPReconstructionParser().parse_args()
  result_args = normalize_result_args(opts.result)
  if len(result_args) != 1:
    raise ValueError('Please supply exactly one --result.')
  opts.result = result_args[0]
  cls = GPReconstructionDiscoveryWarp if opts.discovery \
      else GPReconstructionEnterpriseWarp
  cls(opts, custom_models_obj=_load_custom_model(opts)).main_pipeline()


if __name__ == '__main__':
  main()
