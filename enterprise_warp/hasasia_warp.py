"""
Single-pulsar hasasia sensitivity construction for enterprise_warp outputs.
"""

import glob
import json
import os
import pickle
import re
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from .results import (EnterpriseWarpResult, DiscoveryWarpResult,
                      ResultsParser, normalize_result_args)


YR_SEC = 365.25 * 24.0 * 3600.0


class HasasiaParser(ResultsParser):
  """Standalone parser for hasasia sensitivity runs."""
  def __init__(self):
    super(HasasiaParser, self).__init__()
    self.parser.add_option("--pta", default="psr", type=str,
                           help="Sensitivity mode: psr, gwb, cw, or directional.")
    self.parser.add_option("--snr", default=1.0, type=float,
                           help="Target SNR level for PTA sensitivity outputs.")
    self.parser.add_option("--hasasia", default=1, type=int,
                           help="Run hasasia sensitivity construction (1/0).")
    self.parser.add_option("--num", default=0, type=int,
                           help="Realization/result-folder number.")
    self.parser.add_option("--hasasia_psr", default="0", type=str,
                           help="Pulsar index or name.")
    self.parser.add_option("--hasasia_model", default=0, type=int,
                           help="Model block id.")
    self.parser.add_option("--hasasia_spectrum", default="spectrum", type=str,
                           help="spectrum, rrf, or rrf_projected.")
    self.parser.add_option("--hasasia_load_latest", default=0, type=int,
                           help="Load latest checkpoint (1/0).")
    self.parser.add_option("--hasasia_nf", default=600, type=int,
                           help="Sensitivity-curve frequency count.")
    self.parser.add_option("--hasasia_fmin", default=None, type=float,
                           help="Minimum sensitivity-curve frequency [Hz].")
    self.parser.add_option("--hasasia_fmax", default=2e-7, type=float,
                           help="Maximum sensitivity-curve frequency [Hz].")
    self.parser.add_option("--hasasia_average_toas", default=0, type=int,
                           help="Accepted for future use; TOA averaging is "
                                "not implemented in this step.")
    self.parser.add_option("--hasasia_skymap_nside", default=32, type=int,
                           help="HEALPix NSIDE for --pta directional.")
    self.parser.add_option("--hasasia_pulsar_term", action="store_true",
                           default=False,
                           help="Set pulsar_term=True for PTA CW/directional "
                                "hasasia sensitivity.")
    self.parser.add_option("--hasasia_directional_theta", default=0.0, type=float,
                           help="Sky colatitude theta [rad] for --pta directional "
                                "frequency curve.")
    self.parser.add_option("--hasasia_directional_phi", default=0.0, type=float,
                           help="Sky longitude phi [rad] for --pta directional "
                                "frequency curve.")
    self.parser.add_option("--hasasia_directional_freq", default=None, type=float,
                           help="Frequency [Hz] for --pta directional sky map. "
                                "Defaults to the most sensitive frequency.")

  def parse_args(self):
    opts = super(HasasiaParser, self).parse_args()
    opts.drop = 0
    opts.extra_model_terms = 'None'
    opts.realization = opts.num
    return opts


class RunLog(object):
  def __init__(self, path, append=False):
    self.path = path
    with open(self.path, 'a' if append else 'w') as fout:
      fout.write('hasasia sensitivity run\n')

  def write(self, message):
    print(message)
    with open(self.path, 'a') as fout:
      fout.write(str(message) + '\n')


def posterior_mode(samples, bins=50):
  samples = np.asarray(samples, dtype=float)
  samples = samples[np.isfinite(samples)]
  if samples.size == 0:
    return None
  hist, edges = np.histogram(samples, bins=bins)
  idx = int(np.argmax(hist))
  return 0.5 * (edges[idx] + edges[idx + 1])


def log10rho_psd(log10_rho, tspan):
  """Convert log10_rho delay amplitude to residual PSD in s^3."""
  return 10.0**(2.0 * np.asarray(log10_rho, dtype=float)) * float(tspan)


def _safe_name(name):
  return re.sub(r'[^A-Za-z0-9._+-]+', '_', str(name))


def _freq_match(left, right, rtol=1e-10, atol=0.0):
  left = np.asarray(left, dtype=float)
  right = np.asarray(right, dtype=float)
  return left.shape == right.shape and np.allclose(left, right, rtol=rtol,
                                                   atol=atol)


def _pta_snr_scaled_curves(pta_mode, sensitivity, snr):
  """Return SNR-scaled effective noise and characteristic strain curves."""
  snr = float(snr)
  if snr <= 0.0:
    raise ValueError('--snr must be positive.')

  if pta_mode == 'gwb':
    scaled_seff = snr * np.asarray(sensitivity.S_eff, dtype=float)
    scaled_hc = np.sqrt(snr) * np.asarray(sensitivity.h_c, dtype=float)
  elif pta_mode in ['cw', 'directional']:
    scaled_seff = snr**2 * np.asarray(sensitivity.S_eff, dtype=float)
    scaled_hc = snr * np.asarray(sensitivity.h_c, dtype=float)
  else:
    raise ValueError('Unknown --pta {}'.format(pta_mode))
  return scaled_seff, scaled_hc


def _noise_lookup(noise, psr_name, backend, suffixes, default=None):
  if isinstance(suffixes, str):
    suffixes = [suffixes]
  for suffix in suffixes:
    key = '{}_{}_{}'.format(psr_name, backend, suffix)
    if key in noise:
      return noise[key], key
  return default, None


def build_white_noise_covariance(psr, noise, hsen=None, log=None):
  """Build EFAC/EQUAD/ECORR covariance following hasasia's tutorial."""
  toaerrs = np.asarray(psr.toaerrs, dtype=float)
  corr = np.zeros((toaerrs.size, toaerrs.size), dtype=float)
  flags = getattr(psr, 'flags', {})
  backend_flags = np.asarray(flags['f']).astype(str) if 'f' in flags \
                  else np.repeat('all', toaerrs.size)
  if 'f' not in flags and log is not None:
    log.write('No psr.flags["f"] found; using a single white-noise backend.')

  sigma_sqr = np.zeros(toaerrs.size, dtype=float)
  for backend in np.unique(backend_flags):
    mask = backend_flags == backend
    efac, efac_key = _noise_lookup(noise, psr.name, backend, 'efac', 1.0)
    equad_log10, equad_key = _noise_lookup(
        noise, psr.name, backend,
        ['log10_t2equad', 'log10_equad', 'log10_tnequad'])
    equad = 0.0 if equad_log10 is None else 10.0**float(equad_log10)
    sigma_sqr[mask] = float(efac)**2 * toaerrs[mask]**2 + equad**2
    if log is not None:
      log.write('white {}: efac={} ({}) equad={} ({})'.format(
          backend, efac, efac_key, equad, equad_key))
  np.fill_diagonal(corr, sigma_sqr)

  if hsen is None:
    return corr

  _, _, _, _, buckets = hsen.quantize_fast(np.asarray(psr.toas, dtype=float),
                                           toaerrs, flags=backend_flags, dt=1)
  ecorr_counts = {}
  for bucket in buckets:
    bucket = np.asarray(bucket, dtype=int)
    for backend in np.unique(backend_flags[bucket]):
      local = bucket[backend_flags[bucket] == backend]
      ecorr_log10, ecorr_key = _noise_lookup(noise, psr.name, backend,
                                             'log10_ecorr')
      if ecorr_log10 is None:
        continue
      corr[np.ix_(local, local)] += 10.0**(2.0 * float(ecorr_log10))
      ecorr_counts.setdefault(backend, {'key': ecorr_key,
                                        'value': 10.0**float(ecorr_log10),
                                        'epochs': 0, 'toas': 0})
      ecorr_counts[backend]['epochs'] += 1
      ecorr_counts[backend]['toas'] += local.size

  if log is not None:
    for backend, summary in sorted(ecorr_counts.items()):
      log.write('ecorr {}: {} ({}) across {} epochs and {} TOAs'.format(
          backend, summary['value'], summary['key'],
          summary['epochs'], summary['toas']))
  return corr


class WhiteNoiseModel(object):
  """Structured EFAC/EQUAD/ECORR covariance with cheap inverse application."""

  def __init__(self, diag_var, ecorr_blocks):
    self.diag_var = np.asarray(diag_var, dtype=float)
    self.inv_diag = 1.0 / self.diag_var
    self.ecorr_blocks = [(np.asarray(idx, dtype=int), float(var))
                         for idx, var in ecorr_blocks if float(var) > 0.0]

  def apply_inverse(self, values):
    values = np.asarray(values)
    was_1d = values.ndim == 1
    if was_1d:
      values = values[:, None]
    result = self.inv_diag[:, None] * values
    for idx, ecorr_var in self.ecorr_blocks:
      block_inv_diag = self.inv_diag[idx]
      denom = 1.0 / ecorr_var + np.sum(block_inv_diag)
      weighted_sum = np.sum(result[idx, :], axis=0)
      result[idx, :] -= (block_inv_diag[:, None] * weighted_sum[None, :] /
                         denom)
    return result[:, 0] if was_1d else result


def build_white_noise_model(psr, noise, hsen=None, log=None):
  """Build a structured white-noise model matching build_white_noise_covariance."""
  toaerrs = np.asarray(psr.toaerrs, dtype=float)
  flags = getattr(psr, 'flags', {})
  backend_flags = np.asarray(flags['f']).astype(str) if 'f' in flags \
                  else np.repeat('all', toaerrs.size)
  if 'f' not in flags and log is not None:
    log.write('No psr.flags["f"] found; using a single white-noise backend.')

  sigma_sqr = np.zeros(toaerrs.size, dtype=float)
  for backend in np.unique(backend_flags):
    mask = backend_flags == backend
    efac, efac_key = _noise_lookup(noise, psr.name, backend, 'efac', 1.0)
    equad_log10, equad_key = _noise_lookup(
        noise, psr.name, backend,
        ['log10_t2equad', 'log10_equad', 'log10_tnequad'])
    equad = 0.0 if equad_log10 is None else 10.0**float(equad_log10)
    sigma_sqr[mask] = float(efac)**2 * toaerrs[mask]**2 + equad**2
    if log is not None:
      log.write('white {}: efac={} ({}) equad={} ({})'.format(
          backend, efac, efac_key, equad, equad_key))

  if hsen is None:
    return WhiteNoiseModel(sigma_sqr, [])

  ecorr_blocks = []
  _, _, _, _, buckets = hsen.quantize_fast(np.asarray(psr.toas, dtype=float),
                                           toaerrs, flags=backend_flags, dt=1)
  ecorr_counts = {}
  for bucket in buckets:
    bucket = np.asarray(bucket, dtype=int)
    for backend in np.unique(backend_flags[bucket]):
      local = bucket[backend_flags[bucket] == backend]
      ecorr_log10, ecorr_key = _noise_lookup(noise, psr.name, backend,
                                             'log10_ecorr')
      if ecorr_log10 is None:
        continue
      ecorr_var = 10.0**(2.0 * float(ecorr_log10))
      ecorr_blocks.append((local, ecorr_var))
      ecorr_counts.setdefault(backend, {'key': ecorr_key,
                                        'value': 10.0**float(ecorr_log10),
                                        'epochs': 0, 'toas': 0})
      ecorr_counts[backend]['epochs'] += 1
      ecorr_counts[backend]['toas'] += local.size

  if log is not None:
    for backend, summary in sorted(ecorr_counts.items()):
      log.write('ecorr {}: {} ({}) across {} epochs and {} TOAs'.format(
          backend, summary['value'], summary['key'],
          summary['epochs'], summary['toas']))
  return WhiteNoiseModel(sigma_sqr, ecorr_blocks)


class TimingMarginalizedWhiteOperator(object):
  """Apply timing-marginalized inverse white covariance in bilinear form."""

  def __init__(self, white_model, designmatrix):
    self.white_model = white_model
    designmatrix = np.asarray(designmatrix, dtype=float)
    nrows, ncols = designmatrix.shape
    if ncols >= nrows:
      raise ValueError('Projected RRF requires fewer timing columns than TOAs.')
    self.timing_basis = np.linalg.svd(
        designmatrix, full_matrices=False)[0][:, :ncols]
    self.cinv_timing_basis = self.white_model.apply_inverse(self.timing_basis)
    gram = np.matmul(self.timing_basis.T, self.cinv_timing_basis)
    self.gram_inv = np.linalg.pinv(gram, hermitian=True)

  def inner(self, left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    cinv_right = self.white_model.apply_inverse(right)
    first = np.matmul(np.conjugate(left).T, cinv_right)
    left_cinv_basis = np.matmul(np.conjugate(left).T, self.cinv_timing_basis)
    basis_cinv_right = np.matmul(self.timing_basis.T, cinv_right)
    return first - np.matmul(left_cinv_basis,
                             np.matmul(self.gram_inv, basis_cinv_right))


class ProjectedRRFSpectrum(object):
  """hasasia-like spectrum using projected Woodbury algebra without dense G."""

  def __init__(self, name, toas, toaerrs, freqs, ncalinv, hsen,
               phi=None, theta=None, pdist=None):
    self.name = name
    self.toas = np.asarray(toas, dtype=float)
    self.toaerrs = np.asarray(toaerrs, dtype=float)
    self.freqs = np.asarray(freqs, dtype=float)
    self.phi = None if phi is None else float(phi)
    self.theta = None if theta is None else float(theta)
    self.pdist = pdist
    self.NcalInv = np.asarray(ncalinv, dtype=float)
    self.S_R = 1.0 / self.NcalInv
    self.S_I = 1.0 / hsen.resid_response(self.freqs) / self.NcalInv
    self.h_c = np.sqrt(self.freqs * self.S_I)


def _fourier_design(toas, nfreq, tspan):
  freqs = np.arange(1, int(nfreq) + 1, dtype=float) / float(tspan)
  design = np.empty((toas.size, 2 * int(nfreq)), dtype=float)
  phase = 2.0 * np.pi * toas[:, None] * freqs[None, :]
  design[:, ::2] = np.sin(phase)
  design[:, 1::2] = np.cos(phase)
  return design


def _duplicated_power(power, nfreq):
  vals = np.zeros(2 * int(nfreq), dtype=float)
  vals[::2] = power
  vals[1::2] = power
  return vals


def _rrf_component_nfreq(tspan, curve_freqs, minimum_nfreq):
  """Return a Fourier-component count that reaches the plotted frequency range."""
  minimum_nfreq = int(minimum_nfreq or 0)
  if curve_freqs is None or len(curve_freqs) == 0:
    return minimum_nfreq
  required_nfreq = int(np.ceil(float(np.max(curve_freqs)) * float(tspan)))
  return max(minimum_nfreq, required_nfreq)


def projected_rrf_ncalinv(toas, designmatrix, white_model, curve_freqs,
                          tspan_common, common_nfreq, amp_gw, gamma_gw,
                          red_nfreq, amp_irn, gamma_irn, hsen):
  """Compute Spectrum_RRF NcalInv without explicit G or dense TOA covariance."""
  toas = np.asarray(toas, dtype=float)
  common_nfreq = int(common_nfreq)
  red_nfreq = int(red_nfreq or 0)
  nfreq_irn = max(red_nfreq, common_nfreq)
  if nfreq_irn <= 0:
    raise ValueError('Projected RRF requires at least one Fourier component.')

  operator = TimingMarginalizedWhiteOperator(white_model, designmatrix)
  fourier = _fourier_design(toas, nfreq_irn, tspan_common)
  phi_diag = np.zeros(2 * nfreq_irn, dtype=float)

  freqs_irn = np.arange(1, nfreq_irn + 1, dtype=float) / float(tspan_common)
  if amp_irn is None or gamma_irn is None:
    irn_power = hsen.red_noise_powerlaw(A=1e-40, gamma=0.0, freqs=freqs_irn)
  else:
    irn_power = hsen.red_noise_powerlaw(
        A=float(amp_irn), gamma=float(gamma_irn), freqs=freqs_irn)
  phi_diag += _duplicated_power(irn_power, nfreq_irn) / float(tspan_common)

  freqs_gw = freqs_irn[:common_nfreq]
  gw_power = hsen.red_noise_powerlaw(
      A=float(amp_gw), gamma=float(gamma_gw), freqs=freqs_gw)
  phi_diag[:2 * common_nfreq] += (
      _duplicated_power(gw_power, common_nfreq) / float(tspan_common))

  ff = np.asarray(curve_freqs, dtype=float)
  exp_design = np.exp(1j * 2.0 * np.pi * toas[:, None] * ff[None, :])
  s_ff = operator.inner(fourier, fourier)
  s_fe = operator.inner(fourier, exp_design)
  s_ee = operator.inner(exp_design, exp_design)
  sigma = s_ff + np.diag(1.0 / phi_diag)
  sigma_inv_s_fe = np.linalg.solve(sigma, s_fe)
  correction = np.sum(np.conjugate(s_fe) * sigma_inv_s_fe, axis=0)
  ncalinv = np.real(np.diag(s_ee) - correction) / (2.0 * (toas.max() - toas.min()))
  return ncalinv


class HasasiaWarpMixin(object):
  """Mixin that adds hasasia sensitivity generation to result loaders."""

  def main_pipeline(self):
    if not hasattr(self, 'params'):
      raise ValueError('hasasia_warp requires --result to be a parameter file.')

    self._init_hasasia_pulsars()
    self.hasasia_result_dir = self._get_hasasia_result_dir()
    pta_mode = self._pta_mode()

    if pta_mode == 'psr':
      self._run_single_pulsar_pipeline()
      return
    self._run_full_pta_pipeline(pta_mode)

  def _init_hasasia_pulsars(self):
    if self.params.opts is None:
      self.params.opts = self.opts
    if not hasattr(self.params.opts, 'drop'):
      self.params.opts.drop = 0
    if not hasattr(self.params.opts, 'extra_model_terms'):
      self.params.opts.extra_model_terms = 'None'
    if not hasattr(self.params.opts, 'num'):
      self.params.opts.num = getattr(self.opts, 'num', 0)
    self.params.init_pulsars()

  def _select_psr(self, selector):
    selector = str(selector)
    if selector.isdigit():
      idx = int(selector)
      if idx < 0 or idx >= len(self.params.psrs):
        raise IndexError('hasasia_psr index {} outside 0..{}'.format(
            idx, len(self.params.psrs) - 1))
      return self.params.psrs[idx], idx
    matches = [ii for ii, psr in enumerate(self.params.psrs)
               if psr.name == selector]
    if not matches:
      matches = [ii for ii, psr in enumerate(self.params.psrs)
                 if selector in psr.name]
    if len(matches) != 1:
      raise ValueError('Pulsar selector {} matched {}'.format(
          selector, [self.params.psrs[ii].name for ii in matches]))
    return self.params.psrs[matches[0]], matches[0]

  def _pta_mode(self):
    pta_mode = str(getattr(self.opts, 'pta', 'psr')).lower()
    allowed = ['psr', 'gwb', 'cw', 'directional']
    if pta_mode not in allowed:
      raise ValueError('Unknown --pta {} (allowed: {})'.format(
          pta_mode, ', '.join(allowed)))
    return pta_mode

  def _run_single_pulsar_pipeline(self):
    psr, psr_index = self._select_psr(getattr(self.opts, 'hasasia_psr', '0'))
    self.hasasia_psr = psr
    self.hasasia_psr_index = psr_index

    if getattr(self.opts, 'hasasia_load_latest', 0):
      self._load_latest_checkpoint(psr)
      return

    self.run_dir = self._new_single_pulsar_run_dir(psr.name)
    self.log = RunLog(os.path.join(self.run_dir, 'run.log'))
    self.log.write('Result directory: {}'.format(self.hasasia_result_dir))
    self.log.write('Selected pulsar {} at loaded index {}'.format(psr.name, psr_index))

    self._load_result_chain()
    hpsr, spectrum = self._build_hasasia_objects(psr)
    self._write_outputs(hpsr, spectrum)

  def _run_full_pta_pipeline(self, pta_mode):
    self.run_dir = os.path.join(
        self.hasasia_result_dir, 'hasasia',
        '{}_pta_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'),
                           _safe_name(pta_mode)))
    os.makedirs(self.run_dir, exist_ok=True)
    self.log = RunLog(os.path.join(self.run_dir, 'run.log'))
    self.log.write('Result directory: {}'.format(self.hasasia_result_dir))
    self.log.write('PTA sensitivity mode: {}'.format(pta_mode))
    if getattr(self.opts, 'hasasia_load_latest', 0):
      self.log.write('--hasasia_load_latest is only used for --pta psr; '
                     'full PTA modes always collect latest compatible '
                     'single-pulsar checkpoints.')

    self._load_result_chain()
    spectra, checkpoint_paths = self._load_or_build_pta_spectra()
    sensitivity, extra_settings = self._build_full_pta_object(pta_mode, spectra)
    self._write_full_pta_outputs(pta_mode, sensitivity, checkpoint_paths,
                                 extra_settings)

  def _get_hasasia_result_dir(self, psr=None):
    if self.params.array_analysis:
      return os.path.abspath(self.outdir_all)
    return os.path.abspath(getattr(self.params, 'output_dir', self.outdir_all))

  def _new_single_pulsar_run_dir(self, psr_name):
    run_dir = os.path.join(
        self.hasasia_result_dir, 'hasasia',
        '{}_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'),
                       _safe_name(psr_name)))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

  def _load_result_chain(self):
    self.psr_dir = ''
    self.outdir = self.hasasia_result_dir
    self.get_pars()
    self.get_chain_file_name()
    if self.chain_file is None:
      self.log.write('No posterior chain found; using noisefile fallbacks only.')
      self.chain = None
      self.chain_burn = None
      return False
    success = self.load_chains()
    if not success:
      self.chain = None
      self.chain_burn = None
    return success

  def _get_model_params(self):
    model_id = int(getattr(self.opts, 'hasasia_model', 0))
    if model_id not in self.params.models:
      raise ValueError('Model block {{{}}} not found.'.format(model_id))
    return self.params.models[model_id], model_id

  def _legacy_nfreq(self, value):
    if isinstance(value, str):
      match = re.search(r'(\d+)_nfreqs', value)
      if match:
        return int(match.group(1))
    return None

  def _option_nfreq(self, option, model_params, psr=None):
    if isinstance(option, dict):
      return self.params.noise_model_obj(psr=psr, params=model_params)\
                        .option_nfreqs(option, sel_func_name=None)
    return self._legacy_nfreq(option)

  def _common_nfreq(self, model_params):
    common = getattr(model_params, 'common_signals', {})
    for key in ['common_gp', 'global_gp', 'gwb', 'crn']:
      if key in common:
        return self._option_nfreq(common[key], model_params)
    for option in common.values():
      nfreq = self._option_nfreq(option, model_params)
      if nfreq is not None:
        return nfreq
    return None

  def _red_nfreq(self, model_params, psr):
    blocks = []
    noisemodel = getattr(model_params, 'noisemodel', {})
    if psr.name in noisemodel:
      blocks.append(noisemodel[psr.name])
    blocks.extend([getattr(model_params, 'to_remaining_psrs', {}),
                   getattr(model_params, 'to_each_psr', {})])
    for block in blocks:
      if isinstance(block, dict) and 'spin_noise' in block:
        return self._option_nfreq(block['spin_noise'], model_params, psr=psr)
    return None

  def _mode(self, par_name):
    if getattr(self, 'chain_burn', None) is None:
      return None
    pars = np.asarray(self.pars, dtype=str)
    matches = np.where(pars == par_name)[0]
    if matches.size == 0:
      return None
    return posterior_mode(self.chain_burn[:, matches[0]])

  def _samples(self, par_name):
    if getattr(self, 'chain_burn', None) is None:
      return None
    pars = np.asarray(self.pars, dtype=str)
    matches = np.where(pars == par_name)[0]
    if matches.size == 0:
      return None
    samples = np.asarray(self.chain_burn[:, matches[0]], dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
      return None
    return samples

  def _rho_modes(self, nfreq):
    if nfreq is None:
      return None, None
    for prefix in ['gw', 'crn']:
      vals = []
      for ii in range(int(nfreq)):
        val = None
        for name in ['{}_log10_rho[{}]'.format(prefix, ii),
                     '{}_log10_rho_{}'.format(prefix, ii)]:
          val = self._mode(name)
          if val is not None:
            break
        if val is None:
          vals = []
          break
        vals.append(val)
      if vals:
        return prefix, np.asarray(vals)
    return None, None

  def _common_powerlaw_modes(self):
    for prefix in ['gw', 'crn']:
      amp = self._mode('{}_log10_A'.format(prefix))
      gamma = self._mode('{}_gamma'.format(prefix))
      if amp is not None and gamma is not None:
        return prefix, amp, gamma
    return None

  def _astro_common_powerlaw_modes(self):
    """Approximate astro common-background hyperparameters as a power law.

    The Sato-Polito/Zaldarriaga discrete-background model stores either
    log10_h2c directly or the pair log10_Nc/log10_h2peak.  For the current
    hasasia RRF path we need a power-law strain amplitude A.  These runs use
    the standard f_ref = 1/yr convention, so log10_A = 0.5 * log10_h2c.
    """
    gamma = 13.0 / 3.0
    for prefix in ['gw', 'crn']:
      h2c_samples = self._samples('{}_log10_h2c'.format(prefix))
      if h2c_samples is not None:
        amp = posterior_mode(0.5 * h2c_samples)
        if amp is not None:
          return prefix, amp, gamma, 'derived from {}_log10_h2c'.format(prefix)

      log10_nc = self._samples('{}_log10_Nc'.format(prefix))
      if log10_nc is None:
        nc = self._samples('{}_Nc'.format(prefix))
        if nc is not None and np.all(nc > 0.0):
          log10_nc = np.log10(nc)
      log10_h2peak = self._samples('{}_log10_h2peak'.format(prefix))
      if log10_nc is None or log10_h2peak is None:
        continue
      size = min(log10_nc.size, log10_h2peak.size)
      log10_h2c = log10_nc[:size] + log10_h2peak[:size]
      amp = posterior_mode(0.5 * log10_h2c)
      if amp is not None:
        return prefix, amp, gamma, (
            'derived from {}_log10_Nc + {}_log10_h2peak'.format(prefix, prefix))
    return None

  def _toas_seconds(self, psr):
    toas = np.asarray(psr.toas, dtype=float)
    if np.nanmedian(np.abs(toas)) < 1.0e6:
      return toas * 86400.0, 'MJD converted to seconds'
    return toas, 'seconds'

  def _curve_freqs(self, tspan):
    fmin = getattr(self.opts, 'hasasia_fmin', None)
    fmin = 1.0 / (5.0 * tspan) if fmin is None else float(fmin)
    fmax = float(getattr(self.opts, 'hasasia_fmax', 2e-7))
    return np.logspace(np.log10(fmin), np.log10(fmax),
                       int(getattr(self.opts, 'hasasia_nf', 600)))

  def _ensure_hasasia_path(self):
    has_path = os.environ.get('HAS')
    if has_path is not None and has_path not in sys.path:
      sys.path.insert(0, has_path)

  def _full_pta_tspan(self):
    all_toas = [self._toas_seconds(pp)[0] for pp in self.params.psrs]
    return float(max(tt.max() for tt in all_toas) -
                 min(tt.min() for tt in all_toas))

  def _compatible_spectrum_settings(self, settings, psr_name, curve_freqs):
    if settings.get('selected_pulsar') != psr_name:
      return False
    if settings.get('spectrum') != str(
        getattr(self.opts, 'hasasia_spectrum', 'spectrum')).lower():
      return False
    if int(settings.get('curve_nf', -1)) != int(len(curve_freqs)):
      return False
    if not np.isclose(float(settings.get('curve_fmin', np.nan)),
                      float(curve_freqs[0])):
      return False
    if not np.isclose(float(settings.get('curve_fmax', np.nan)),
                      float(curve_freqs[-1])):
      return False
    if int(settings.get('average_toas', -1)) != int(
        getattr(self.opts, 'hasasia_average_toas', 0)):
      return False
    return True

  def _load_checkpoint_settings(self, path):
    settings_path = os.path.join(path, 'settings.json')
    if not os.path.isfile(settings_path):
      return None
    with open(settings_path, 'r') as fin:
      return json.load(fin)

  def _find_latest_compatible_checkpoint(self, psr_name, curve_freqs):
    pattern = os.path.join(self.hasasia_result_dir, 'hasasia',
                           '*_{}'.format(_safe_name(psr_name)))
    candidates = sorted(glob.glob(pattern), reverse=True)
    for path in candidates:
      if not os.path.isfile(os.path.join(path, 'spectrum.pkl')) or \
         not os.path.isfile(os.path.join(path, 'pulsar.pkl')):
        continue
      settings = self._load_checkpoint_settings(path)
      if settings is not None and self._compatible_spectrum_settings(
          settings, psr_name, curve_freqs):
        return path

    self._ensure_hasasia_path()
    for path in candidates:
      spectrum_path = os.path.join(path, 'spectrum.pkl')
      if not os.path.isfile(spectrum_path):
        continue
      with open(spectrum_path, 'rb') as fin:
        spectrum = pickle.load(fin)
      if _freq_match(getattr(spectrum, 'freqs', None), curve_freqs):
        return path
    return None

  def _load_spectrum_checkpoint(self, path):
    self._ensure_hasasia_path()
    with open(os.path.join(path, 'pulsar.pkl'), 'rb') as fin:
      hpsr = pickle.load(fin)
    with open(os.path.join(path, 'spectrum.pkl'), 'rb') as fin:
      spectrum = pickle.load(fin)
    _ = spectrum.NcalInv
    return hpsr, spectrum

  def _build_single_pulsar_checkpoint(self, psr, psr_index):
    previous_log = getattr(self, 'log', None)
    previous_run_dir = getattr(self, 'run_dir', None)
    previous_psr = getattr(self, 'hasasia_psr', None)
    previous_psr_index = getattr(self, 'hasasia_psr_index', None)
    run_dir = self._new_single_pulsar_run_dir(psr.name)
    self.run_dir = run_dir
    self.log = RunLog(os.path.join(run_dir, 'run.log'))
    self.log.write('Result directory: {}'.format(self.hasasia_result_dir))
    self.log.write('Selected pulsar {} at loaded index {}'.format(
        psr.name, psr_index))
    self.hasasia_psr = psr
    self.hasasia_psr_index = psr_index
    try:
      hpsr, spectrum = self._build_hasasia_objects(psr)
      self._write_outputs(hpsr, spectrum, run_dir=run_dir)
    finally:
      self.log = previous_log
      self.run_dir = previous_run_dir
      self.hasasia_psr = previous_psr
      self.hasasia_psr_index = previous_psr_index
    return hpsr, spectrum, run_dir

  def _load_or_build_pta_spectra(self):
    curve_freqs = self._curve_freqs(self._full_pta_tspan())
    spectra = {}
    checkpoint_paths = {}
    missing = []
    for psr_index, psr in enumerate(self.params.psrs):
      checkpoint = self._find_latest_compatible_checkpoint(psr.name, curve_freqs)
      if checkpoint is None:
        missing.append((psr_index, psr))
        continue
      _, spectrum = self._load_spectrum_checkpoint(checkpoint)
      spectra[psr.name] = spectrum
      checkpoint_paths[psr.name] = checkpoint
      self.log.write('Loaded spectrum checkpoint for {}: {}'.format(
          psr.name, checkpoint))

    if missing:
      from tqdm import tqdm
      self.log.write('Building {} missing single-pulsar spectra.'.format(
          len(missing)))
      for psr_index, psr in tqdm(missing, desc='hasasia spectra', unit='psr'):
        _, spectrum, checkpoint = self._build_single_pulsar_checkpoint(
            psr, psr_index)
        spectra[psr.name] = spectrum
        checkpoint_paths[psr.name] = checkpoint
        self.log.write('Built spectrum checkpoint for {}: {}'.format(
            psr.name, checkpoint))
    return [spectra[psr.name] for psr in self.params.psrs], checkpoint_paths

  def _build_full_pta_object(self, pta_mode, spectra):
    self._ensure_hasasia_path()
    import hasasia.sensitivity as hsen

    extra_settings = {}
    if pta_mode == 'gwb':
      sensitivity = hsen.GWBSensitivityCurve(spectra)
      extra_settings['pta_class'] = 'hasasia.sensitivity.GWBSensitivityCurve'
    elif pta_mode == 'cw':
      kwargs = {}
      if getattr(self.opts, 'hasasia_pulsar_term', False):
        kwargs['pulsar_term'] = True
      sensitivity = hsen.DeterSensitivityCurve(spectra, **kwargs)
      extra_settings['pta_class'] = 'hasasia.sensitivity.DeterSensitivityCurve'
      extra_settings['pulsar_term'] = sensitivity.pulsar_term
    elif pta_mode == 'directional':
      import healpy as hp
      import hasasia.skymap as hsky

      nside = int(getattr(self.opts, 'hasasia_skymap_nside', 32))
      npix = hp.nside2npix(nside)
      ipix = np.arange(npix)
      theta_gw, phi_gw = hp.pix2ang(nside=nside, ipix=ipix)
      kwargs = {}
      if getattr(self.opts, 'hasasia_pulsar_term', False):
        kwargs['pulsar_term'] = True
      sensitivity = hsky.SkySensitivity(spectra, theta_gw, phi_gw, **kwargs)
      extra_settings.update({
          'pta_class': 'hasasia.skymap.SkySensitivity',
          'pulsar_term': sensitivity.pulsar_term,
          'skymap_nside': nside,
          'skymap_npix': int(npix),
      })
    else:
      raise ValueError('Unknown --pta {}'.format(pta_mode))
    return sensitivity, extra_settings

  def _directional_selection(self, sensitivity):
    theta = float(getattr(self.opts, 'hasasia_directional_theta', 0.0))
    phi = float(getattr(self.opts, 'hasasia_directional_phi', 0.0))
    dphi = np.asarray(sensitivity.phi_gw, dtype=float) - phi
    cos_sep = (np.cos(np.asarray(sensitivity.theta_gw, dtype=float)) * np.cos(theta)
               + np.sin(np.asarray(sensitivity.theta_gw, dtype=float)) * np.sin(theta)
               * np.cos(dphi))
    sky_idx = int(np.argmax(cos_sep))
    sky_freq = getattr(self.opts, 'hasasia_directional_freq', None)
    if sky_freq is None:
      freq_idx = int(np.argmin(sensitivity.S_eff_mean))
    else:
      freq_idx = int(sensitivity.fidx(float(sky_freq))[0])
    return sky_idx, freq_idx

  def _build_hasasia_objects(self, psr):
    self._ensure_hasasia_path()
    import hasasia.sensitivity as hsen

    model_params, model_id = self._get_model_params()
    common_nfreq = self._common_nfreq(model_params)
    red_nfreq = self._red_nfreq(model_params, psr)
    self.log.write('Selected model id: {}'.format(model_id))
    self.log.write('Selected model file: {}'.format(
        getattr(model_params, 'model_file', None)))
    self.log.write('Model common n_freqs: {}'.format(common_nfreq))
    self.log.write('Model red-noise n_freqs for {}: {}'.format(psr.name, red_nfreq))

    toas, unit_note = self._toas_seconds(psr)
    all_toas = [self._toas_seconds(pp)[0] for pp in self.params.psrs]
    common_tspan = float(max(tt.max() for tt in all_toas) -
                         min(tt.min() for tt in all_toas))
    self.log.write('TOA unit handling: {}'.format(unit_note))

    noise = {key: val for key, val in getattr(self.params, 'noisedict', {}).items()
             if key.startswith(psr.name)}
    self.log.write('Loaded {} noise parameters for {}'.format(len(noise), psr.name))

    spectrum_kind = str(getattr(self.opts, 'hasasia_spectrum', 'spectrum')).lower()
    designmatrix = getattr(psr, 'Mmat', getattr(psr, 'designmatrix', None))
    if designmatrix is None:
      raise ValueError('Selected pulsar has neither Mmat nor designmatrix.')
    designmatrix = np.asarray(designmatrix, dtype=float)
    work_psr = SimpleNamespace(name=psr.name, toas=toas,
                               toaerrs=np.asarray(psr.toaerrs, dtype=float),
                               flags=getattr(psr, 'flags', {}),
                               phi=psr.phi, theta=psr.theta)
    if getattr(self.opts, 'hasasia_average_toas', 0):
      raise NotImplementedError('--hasasia_average_toas is accepted but not '
                                'implemented in this step.')
    toaerrs = np.asarray(psr.toaerrs, dtype=float)
    projected_rrf = spectrum_kind in ['rrf_projected', 'rrf_nodense',
                                      'rrf_no_dense']
    if projected_rrf:
      white_model = build_white_noise_model(work_psr, noise, hsen=hsen,
                                            log=self.log)
      total_n = None
      self.log.write('Using structured white-noise model for projected RRF.')
    else:
      white_model = None
      total_n = build_white_noise_covariance(work_psr, noise, hsen=hsen,
                                             log=self.log)
    self.log.write('TOA averaging disabled; passing original TOAs to hasasia.')
    red_tspan = float(toas.max() - toas.min())
    self.log.write('TOA rows passed to hasasia: {}'.format(toas.size))
    self.log.write('TOA span for selected pulsar [yr]: {}'.format(red_tspan / YR_SEC))

    red_amp_log10 = self._mode('{}_red_noise_log10_A'.format(psr.name))
    red_gamma = self._mode('{}_red_noise_gamma'.format(psr.name))
    red_source = 'posterior'
    if red_amp_log10 is None or red_gamma is None:
      red_amp_log10 = noise.get('{}_red_noise_log10_A'.format(psr.name))
      red_gamma = noise.get('{}_red_noise_gamma'.format(psr.name))
      red_source = 'noisefile'
    if red_amp_log10 is not None and red_gamma is not None and red_nfreq is not None:
      red_freqs = np.arange(1, int(red_nfreq) + 1, dtype=float) / red_tspan
      red_psd = hsen.red_noise_powerlaw(
          A=10.0**float(red_amp_log10), gamma=float(red_gamma), freqs=red_freqs)
      if spectrum_kind == 'rrf' or projected_rrf:
        self.log.write('Using red noise in Spectrum_RRF: source={} log10_A={} '
                       'gamma={} nfreq={}'.format(
                           red_source, red_amp_log10, red_gamma, red_nfreq))
      else:
        total_n += np.asarray(hsen.corr_from_psd(red_freqs, red_psd, toas))
        self.log.write('Added red noise from {}: log10_A={} gamma={} nfreq={}'
                       .format(red_source, red_amp_log10, red_gamma, red_nfreq))
    else:
      self.log.write('No pulsar red-noise covariance added.')

    common_powerlaw = None
    common_powerlaw_source = None
    if spectrum_kind == 'rrf' or projected_rrf:
      common_powerlaw = self._common_powerlaw_modes()
      if common_powerlaw is not None:
        common_powerlaw_source = 'explicit log10_A/gamma'
      if common_powerlaw is None:
        astro_powerlaw = self._astro_common_powerlaw_modes()
        if astro_powerlaw is not None:
          prefix, amp_log10, gamma, source = astro_powerlaw
          common_powerlaw = (prefix, amp_log10, gamma)
          common_powerlaw_source = source
      if common_powerlaw is None:
        rho_prefix, _ = self._rho_modes(common_nfreq)
        if rho_prefix is not None:
          self.log.write('Found {} free-spectrum common process, but '
                         'Spectrum_RRF requires log10_A/gamma or astro '
                         'log10_Nc/log10_h2peak.'.format(rho_prefix))
        else:
          self.log.write('No common GWB/CRN covariance added.')
      else:
        prefix, amp_log10, gamma = common_powerlaw
        self.log.write('Using {} power-law common process in Spectrum_RRF: '
                       'log10_A={} gamma={} nfreq={} ({})'.format(
                           prefix, amp_log10, gamma, common_nfreq,
                           common_powerlaw_source))
    else:
      rho_prefix, log10_rho = self._rho_modes(common_nfreq)
      if rho_prefix is not None:
        common_freqs = np.arange(1, int(common_nfreq) + 1, dtype=float) / common_tspan
        total_n += np.asarray(hsen.corr_from_psd(
            common_freqs, log10rho_psd(log10_rho, common_tspan), toas))
        self.log.write('Added {} free-spectrum common process with nfreq={}'
                       .format(rho_prefix, common_nfreq))
      else:
        common_powerlaw = self._common_powerlaw_modes()
        if common_powerlaw is not None and common_nfreq is not None:
          prefix, amp_log10, gamma = common_powerlaw
          common_freqs = np.arange(1, int(common_nfreq) + 1, dtype=float) / common_tspan
          common_psd = hsen.red_noise_powerlaw(
              A=10.0**float(amp_log10), gamma=float(gamma), freqs=common_freqs)
          total_n += np.asarray(hsen.corr_from_psd(common_freqs, common_psd, toas))
          self.log.write('Added {} power-law common process: log10_A={} gamma={} '
                         'nfreq={}'.format(prefix, amp_log10, gamma, common_nfreq))
        else:
          self.log.write('No common GWB/CRN covariance added.')

    curve_freqs = self._curve_freqs(common_tspan)
    rrf_common_nfreq = common_nfreq
    rrf_red_nfreq = red_nfreq
    if spectrum_kind == 'rrf' or projected_rrf:
      rrf_common_nfreq = _rrf_component_nfreq(
          common_tspan, curve_freqs, common_nfreq)
      rrf_red_nfreq = _rrf_component_nfreq(
          common_tspan, curve_freqs, max(int(red_nfreq or 0), int(common_nfreq or 0)))
      self.log.write('Expanded RRF Fourier components to reach plotted fmax: '
                     'common {} -> {}, red {} -> {}'.format(
                         common_nfreq, rrf_common_nfreq,
                         red_nfreq, rrf_red_nfreq))

    if spectrum_kind == 'spectrum':
      hpsr = hsen.Pulsar(toas=toas, toaerrs=toaerrs,
                         phi=psr.phi, theta=psr.theta, name=psr.name,
                         N=total_n, designmatrix=designmatrix)
      spectrum = hsen.Spectrum(hpsr, freqs=curve_freqs)
    elif spectrum_kind == 'rrf':
      if common_powerlaw is None:
        raise ValueError('Spectrum_RRF requires common-process log10_A/gamma.')
      _, amp_log10, gamma = common_powerlaw
      hpsr = hsen.Pulsar(toas=toas, toaerrs=toaerrs,
                         phi=psr.phi, theta=psr.theta, name=psr.name,
                         N=total_n, designmatrix=designmatrix)
      spectrum = hsen.Spectrum_RRF(
          hpsr, Tspan=common_tspan, freqs_gw_comp=int(rrf_common_nfreq),
          amp_gw=10.0**float(amp_log10), gamma_gw=float(gamma),
          freqs_irn_comp=int(rrf_red_nfreq),
          amp_irn=None if red_amp_log10 is None else 10.0**float(red_amp_log10),
          gamma_irn=None if red_gamma is None else float(red_gamma),
          freqs=curve_freqs)
    elif projected_rrf:
      if common_powerlaw is None:
        raise ValueError('Projected RRF requires common-process log10_A/gamma.')
      _, amp_log10, gamma = common_powerlaw
      ncalinv = projected_rrf_ncalinv(
          toas=toas, designmatrix=designmatrix, white_model=white_model,
          curve_freqs=curve_freqs, tspan_common=common_tspan,
          common_nfreq=int(rrf_common_nfreq), amp_gw=10.0**float(amp_log10),
          gamma_gw=float(gamma), red_nfreq=int(rrf_red_nfreq),
          amp_irn=None if red_amp_log10 is None else 10.0**float(red_amp_log10),
          gamma_irn=None if red_gamma is None else float(red_gamma),
          hsen=hsen)
      hpsr = SimpleNamespace(toas=toas, toaerrs=toaerrs, phi=psr.phi,
                             theta=psr.theta, name=psr.name,
                             designmatrix=designmatrix)
      spectrum = ProjectedRRFSpectrum(psr.name, toas, toaerrs, curve_freqs,
                                      ncalinv, hsen, phi=psr.phi,
                                      theta=psr.theta,
                                      pdist=getattr(psr, 'pdist', None))
    else:
      raise ValueError('Unknown --hasasia_spectrum {}'.format(spectrum_kind))
    _ = spectrum.NcalInv

    self.hasasia_settings = {
        'result': str(self.opts.result),
        'result_dir': self.hasasia_result_dir,
        'selected_model': model_id,
        'selected_model_file': getattr(model_params, 'model_file', None),
        'selected_pulsar': psr.name,
        'selected_pulsar_index': self.hasasia_psr_index,
        'realization_num': int(getattr(self.opts, 'num', 0)),
        'common_nfreq': common_nfreq,
        'common_powerlaw_prefix': None if common_powerlaw is None
                                  else common_powerlaw[0],
        'common_powerlaw_log10_A': None if common_powerlaw is None
                                  else float(common_powerlaw[1]),
        'common_powerlaw_gamma': None if common_powerlaw is None
                                 else float(common_powerlaw[2]),
        'common_powerlaw_source': common_powerlaw_source,
        'red_noise_nfreq': red_nfreq,
        'rrf_common_nfreq': None if spectrum_kind == 'spectrum'
                            else int(rrf_common_nfreq),
        'rrf_red_noise_nfreq': None if spectrum_kind == 'spectrum'
                               else int(rrf_red_nfreq),
        'curve_nf': int(getattr(self.opts, 'hasasia_nf', 600)),
        'curve_fmin': float(curve_freqs[0]),
        'curve_fmax': float(curve_freqs[-1]),
        'spectrum': spectrum_kind,
        'average_toas': int(getattr(self.opts, 'hasasia_average_toas', 0)),
    }
    return hpsr, spectrum

  def _write_outputs(self, psr, spectrum, run_dir=None, settings=None):
    run_dir = self.run_dir if run_dir is None else run_dir
    settings = self.hasasia_settings if settings is None else settings
    with open(os.path.join(run_dir, 'settings.json'), 'w') as fout:
      json.dump(settings, fout, indent=2, sort_keys=True)
      fout.write('\n')
    with open(os.path.join(run_dir, 'pulsar.pkl'), 'wb') as fout:
      pickle.dump(psr, fout, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(run_dir, 'spectrum.pkl'), 'wb') as fout:
      pickle.dump(spectrum, fout, protocol=pickle.HIGHEST_PROTOCOL)

    table = np.column_stack([spectrum.freqs, spectrum.h_c,
                             spectrum.S_I, spectrum.S_R, spectrum.NcalInv])
    txt_path = os.path.join(run_dir,
                            'sensitivity_{}.txt'.format(_safe_name(psr.name)))
    np.savetxt(txt_path, table, header='freq_Hz h_c S_I S_R NcalInv')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4, 4.8))
    plt.loglog(spectrum.freqs, spectrum.h_c, color='C0')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Characteristic Strain, h_c')
    plt.title(psr.name)
    plt.grid(which='both', alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(run_dir,
                             'sensitivity_{}.png'.format(_safe_name(psr.name)))
    plt.savefig(plot_path, dpi=150)
    plt.close()
    self.log.write('Saved sensitivity table {}'.format(txt_path))
    self.log.write('Saved sensitivity plot {}'.format(plot_path))

  def _write_full_pta_outputs(self, pta_mode, sensitivity, checkpoint_paths,
                              extra_settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    snr = float(getattr(self.opts, 'snr', 1.0))
    scaled_seff, scaled_hc = _pta_snr_scaled_curves(
        pta_mode, sensitivity, snr)
    self.hasasia_settings = {
        'result': str(self.opts.result),
        'result_dir': self.hasasia_result_dir,
        'pta': pta_mode,
        'snr': snr,
        'npsr': len(self.params.psrs),
        'curve_nf': int(len(sensitivity.freqs)),
        'curve_fmin': float(sensitivity.freqs[0]),
        'curve_fmax': float(sensitivity.freqs[-1]),
        'spectrum': str(getattr(self.opts, 'hasasia_spectrum', 'spectrum')).lower(),
        'average_toas': int(getattr(self.opts, 'hasasia_average_toas', 0)),
        'single_pulsar_checkpoints': checkpoint_paths,
    }
    self.hasasia_settings.update(extra_settings)
    with open(os.path.join(self.run_dir, 'settings.json'), 'w') as fout:
      json.dump(self.hasasia_settings, fout, indent=2, sort_keys=True)
      fout.write('\n')

    if pta_mode == 'gwb':
      table = np.column_stack([sensitivity.freqs, scaled_hc,
                               scaled_seff])
      txt_path = os.path.join(self.run_dir, 'pta_gwb_sensitivity.txt')
      np.savetxt(txt_path, table, header='freq_Hz h_c_snr S_eff_snr')
      plt.figure(figsize=(6.4, 4.8))
      plt.loglog(sensitivity.freqs, scaled_hc, color='C0')
      plt.xlabel('Frequency [Hz]')
      plt.ylabel('Characteristic Strain, h_c')
      plt.title('PTA GWB Sensitivity (SNR={})'.format(snr))
      plt.grid(which='both', alpha=0.3)
      plt.tight_layout()
      plot_path = os.path.join(self.run_dir, 'pta_gwb_sensitivity.png')
      plt.savefig(plot_path, dpi=150)
      plt.close()
      self.log.write('Saved PTA sensitivity table {}'.format(txt_path))
      self.log.write('Saved PTA sensitivity plot {}'.format(plot_path))
      return

    if pta_mode == 'cw':
      h0_snr = snr * np.sqrt(np.asarray(sensitivity.S_eff, dtype=float) /
                             float(sensitivity.Tspan))
      table = np.column_stack([sensitivity.freqs, scaled_hc,
                               scaled_seff, h0_snr])
      txt_path = os.path.join(self.run_dir, 'pta_cw_sensitivity.txt')
      np.savetxt(txt_path, table, header='freq_Hz h_c_snr S_eff_snr h0_snr')

      plt.figure(figsize=(6.4, 4.8))
      plt.loglog(sensitivity.freqs, scaled_hc, color='C0')
      plt.xlabel('Frequency [Hz]')
      plt.ylabel('Characteristic Strain, h_c')
      plt.title('PTA CW Sensitivity (SNR={})'.format(snr))
      plt.grid(which='both', alpha=0.3)
      plt.tight_layout()
      hc_plot_path = os.path.join(self.run_dir, 'pta_cw_hc.png')
      plt.savefig(hc_plot_path, dpi=150)
      plt.close()

      plt.figure(figsize=(6.4, 4.8))
      plt.loglog(sensitivity.freqs, h0_snr, color='C1')
      plt.xlabel('Frequency [Hz]')
      plt.ylabel(r'$h_0$ for target SNR')
      plt.title('PTA CW Sensitivity (SNR={})'.format(snr))
      plt.grid(which='both', alpha=0.3)
      plt.tight_layout()
      h0_plot_path = os.path.join(self.run_dir, 'pta_cw_h0.png')
      plt.savefig(h0_plot_path, dpi=150)
      plt.close()
      self.log.write('Saved PTA sensitivity table {}'.format(txt_path))
      self.log.write('Saved PTA sensitivity plots {}, {}'.format(
          hc_plot_path, h0_plot_path))
      return

    if pta_mode == 'directional':
      import healpy as hp

      sky_idx, freq_idx = self._directional_selection(sensitivity)
      h0_snr = snr * np.sqrt(np.asarray(sensitivity.S_eff, dtype=float) /
                             float(sensitivity.Tspan))
      curve_table = np.column_stack([
          sensitivity.freqs,
          scaled_hc[:, sky_idx],
          scaled_seff[:, sky_idx],
          h0_snr[:, sky_idx],
      ])
      curve_txt_path = os.path.join(
          self.run_dir, 'pta_directional_curve.txt')
      np.savetxt(curve_txt_path, curve_table,
                 header='freq_Hz h_c_snr S_eff_snr h0_snr')

      skymap_table = np.column_stack([
          np.arange(len(sensitivity.theta_gw), dtype=int),
          np.asarray(sensitivity.theta_gw, dtype=float),
          np.asarray(sensitivity.phi_gw, dtype=float),
          scaled_hc[freq_idx, :],
          scaled_seff[freq_idx, :],
          h0_snr[freq_idx, :],
      ])
      skymap_txt_path = os.path.join(
          self.run_dir, 'pta_directional_skymap.txt')
      np.savetxt(skymap_txt_path, skymap_table,
                 header='ipix theta_gw_rad phi_gw_rad h_c_snr S_eff_snr h0_snr')

      self.hasasia_settings.update({
          'directional_curve_theta_rad': float(getattr(
              self.opts, 'hasasia_directional_theta', 0.0)),
          'directional_curve_phi_rad': float(getattr(
              self.opts, 'hasasia_directional_phi', 0.0)),
          'directional_curve_selected_pixel': sky_idx,
          'directional_curve_selected_theta_rad': float(
              np.asarray(sensitivity.theta_gw, dtype=float)[sky_idx]),
          'directional_curve_selected_phi_rad': float(
              np.asarray(sensitivity.phi_gw, dtype=float)[sky_idx]),
          'directional_skymap_selected_freq_hz': float(
              sensitivity.freqs[freq_idx]),
          'directional_skymap_selected_freq_index': freq_idx,
      })
      with open(os.path.join(self.run_dir, 'settings.json'), 'w') as fout:
        json.dump(self.hasasia_settings, fout, indent=2, sort_keys=True)
        fout.write('\n')

      plt.figure(figsize=(6.4, 4.8))
      plt.loglog(sensitivity.freqs, scaled_hc[:, sky_idx], color='C0')
      plt.xlabel('Frequency [Hz]')
      plt.ylabel('Characteristic Strain, h_c')
      plt.title('Directional PTA Sensitivity (SNR={})'.format(snr))
      plt.grid(which='both', alpha=0.3)
      plt.tight_layout()
      hc_plot_path = os.path.join(self.run_dir, 'pta_directional_curve.png')
      plt.savefig(hc_plot_path, dpi=150)
      plt.close()

      plt.figure(figsize=(6.4, 4.8))
      plt.loglog(sensitivity.freqs, h0_snr[:, sky_idx], color='C1')
      plt.xlabel('Frequency [Hz]')
      plt.ylabel(r'$h_0$ for target SNR')
      plt.title('Directional PTA Sensitivity (SNR={})'.format(snr))
      plt.grid(which='both', alpha=0.3)
      plt.tight_layout()
      h0_plot_path = os.path.join(self.run_dir, 'pta_directional_h0_snr.png')
      plt.savefig(h0_plot_path, dpi=150)
      plt.close()

      hp.mollview(h0_snr[freq_idx, :], rot=(180, 0, 0),
                  title='Directional PTA h0 at {:.3e} Hz'.format(
                      sensitivity.freqs[freq_idx]),
                  cmap='viridis_r')
      hp.visufunc.projscatter(sensitivity.thetas, sensitivity.phis, marker='*',
                              color='white', edgecolors='k', s=60)
      hp.graticule()
      plt.tight_layout()
      sky_plot_path = os.path.join(self.run_dir, 'pta_directional_skymap.png')
      plt.savefig(sky_plot_path, dpi=150)
      plt.close()
      self.log.write('Saved PTA directional tables {}, {}'.format(
          curve_txt_path, skymap_txt_path))
      self.log.write('Saved PTA directional plots {}, {}, {}'.format(
          hc_plot_path, h0_plot_path, sky_plot_path))
      return

    raise ValueError('Unknown --pta {}'.format(pta_mode))

  def _load_latest_checkpoint(self, psr):
    self._ensure_hasasia_path()
    pattern = os.path.join(self.hasasia_result_dir, 'hasasia',
                           '*_{}'.format(_safe_name(psr.name)))
    candidates = [path for path in glob.glob(pattern)
                  if os.path.isfile(os.path.join(path, 'pulsar.pkl'))
                  and os.path.isfile(os.path.join(path, 'spectrum.pkl'))]
    if not candidates:
      raise ValueError('No hasasia checkpoint found under {}'.format(pattern))
    latest = sorted(candidates)[-1]
    log = RunLog(os.path.join(latest, 'run.log'), append=True)
    log.write('Loading latest checkpoint {}'.format(latest))
    with open(os.path.join(latest, 'pulsar.pkl'), 'rb') as fin:
      hpsr = pickle.load(fin)
    with open(os.path.join(latest, 'spectrum.pkl'), 'rb') as fin:
      spectrum = pickle.load(fin)
    _ = spectrum.NcalInv
    log.write('Loaded checkpoint for {}'.format(hpsr.name))


class HasasiaEnterpriseWarp(HasasiaWarpMixin, EnterpriseWarpResult):
  pass


class HasasiaDiscoveryWarp(HasasiaWarpMixin, DiscoveryWarpResult):
  pass


def _load_custom_model(opts):
  if opts.custom_models is not None and opts.custom_models_py is not None:
    import importlib
    spec = importlib.util.spec_from_file_location("custom_models_obj",
                                                  opts.custom_models_py)
    cmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmod)
    return cmod.__dict__[opts.custom_models]
  if opts.custom_models is None and opts.custom_models_py is None:
    return None
  raise ValueError('Please set both --custom_models and --custom_models_obj')


def main():
  opts = HasasiaParser().parse_args()
  result_args = normalize_result_args(opts.result)
  if len(result_args) != 1:
    raise ValueError('Please supply exactly one --result.')
  opts.result = result_args[0]
  cls = HasasiaDiscoveryWarp if opts.discovery else HasasiaEnterpriseWarp
  cls(opts, custom_models_obj=_load_custom_model(opts)).main_pipeline()


if __name__ == '__main__':
  main()
