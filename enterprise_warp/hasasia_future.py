"""Future TOA projection helpers for hasasia sensitivity forecasts."""

import hashlib
from types import SimpleNamespace

import numpy as np


YR_SEC = 365.25 * 24.0 * 3600.0
FUTURE_CADENCE_WINDOW_YEARS = 1.0
FUTURE_RESIDUAL_MODEL = 'correlated_gaussian_ecorr'
FUTURE_DESIGNMATRIX_POLICY = 'zero_row_pad'
FUTURE_EPOCH_SECONDS = 24.0 * 3600.0
FUTURE_DESIGNMATRIX_NOTE = (
    'Synthetic future TOAs use zero-padded timing-model rows, so the '
    'historical timing-fit projection is preserved on existing TOAs while '
    'future rows are not additionally timing-fit coupled. Low-frequency '
    'future sensitivity may therefore be optimistic.')


def backend_flags(psr, size=None):
  """Return per-TOA backend labels for enterprise or discovery pulsars."""
  backend_vals = getattr(psr, 'backend_flags', None)
  if backend_vals is None:
    flags = getattr(psr, 'flags', None)
    if isinstance(flags, dict):
      for key in ['f', 'backend', 'be']:
        if key in flags:
          backend_vals = flags[key]
          break
  if backend_vals is None:
    if size is None:
      size = len(np.asarray(getattr(psr, 'toas', []), dtype=float))
    return np.repeat('all', int(size)).astype(str)
  backend_vals = np.asarray(backend_vals).astype(str)
  if size is not None and backend_vals.size != int(size):
    raise ValueError('Backend flag shape {} does not match size {}.'.format(
        backend_vals.size, size))
  return backend_vals


def _extract_auxiliary_array(psr, names, size):
  for name in names:
    values = getattr(psr, name, None)
    if values is None:
      continue
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == int(size):
      return arr
  return np.full(int(size), np.nan, dtype=float)


def _quantize_toas(toas, dt=FUTURE_EPOCH_SECONDS):
  toas = np.asarray(toas, dtype=float).reshape(-1)
  if toas.size == 0:
    return np.zeros(0, dtype=int)
  order = np.argsort(toas)
  bins = np.zeros(toas.size, dtype=int)
  current_bin = 0
  current_toa = float(toas[order[0]])
  for idx in order:
    if float(toas[idx]) - current_toa > float(dt):
      current_toa = float(toas[idx])
      current_bin += 1
    bins[idx] = current_bin
  return bins


def group_toa_epoch_indices(toas, epoch_seconds=FUTURE_EPOCH_SECONDS):
  toas = np.asarray(toas, dtype=float).reshape(-1)
  if toas.size == 0:
    return []
  bins = _quantize_toas(toas, dt=epoch_seconds)
  return [np.where(bins == bin_id)[0]
          for bin_id in range(int(np.max(bins)) + 1)]


def _epoch_centers(toas, epoch_indices):
  toas = np.asarray(toas, dtype=float).reshape(-1)
  if not epoch_indices:
    return np.zeros(0, dtype=float)
  return np.asarray([float(np.median(toas[idx])) for idx in epoch_indices],
                    dtype=float)


def _mean_epoch_spacing(epoch_centers):
  epoch_centers = np.asarray(epoch_centers, dtype=float).reshape(-1)
  if epoch_centers.size < 2:
    return None
  diffs = np.diff(np.sort(epoch_centers))
  diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
  if diffs.size == 0:
    return None
  return float(np.mean(diffs))


def _recent_positions(epoch_centers, current_end, window_seconds):
  epoch_centers = np.asarray(epoch_centers, dtype=float).reshape(-1)
  if epoch_centers.size == 0:
    return np.zeros(0, dtype=int)
  return np.where(epoch_centers >= float(current_end) - float(window_seconds))[0]


def _stable_seed(*tokens):
  payload = '|'.join(str(token) for token in tokens)
  digest = hashlib.sha256(payload.encode('utf-8')).digest()
  return int.from_bytes(digest[:4], byteorder='big', signed=False)


def extend_psr_for_future(psr, designmatrix, future_years, white_noise_resolver,
                          run_signature, cadence_window_years=
                          FUTURE_CADENCE_WINDOW_YEARS,
                          historical_whitened_residuals=None):
  """Return a pulsar-like object extended with synthetic future TOAs."""
  future_years = float(future_years)
  if future_years <= 0.0:
    raise ValueError('Future extension requires future_years > 0, got {}.'
                     .format(future_years))

  toas = np.asarray(psr.toas, dtype=float).reshape(-1)
  toaerrs = np.asarray(psr.toaerrs, dtype=float).reshape(-1)
  if toas.size != toaerrs.size:
    raise ValueError('TOAs and TOA errors must match, got {} and {}.'.format(
        toas.size, toaerrs.size))
  if toas.size == 0:
    raise ValueError('Cannot extend a pulsar with no TOAs.')

  backend_vals = backend_flags(psr, size=toas.size)
  raw_residuals = _extract_auxiliary_array(psr, ['residuals', 'resids'],
                                           toas.size)
  freqs = _extract_auxiliary_array(psr, ['freqs', 'freq'], toas.size)
  if historical_whitened_residuals is None:
    whitened_residuals = None
  else:
    whitened_residuals = np.asarray(historical_whitened_residuals,
                                    dtype=float).reshape(-1)
    if whitened_residuals.size != toas.size:
      raise ValueError('Historical whitened residual count {} does not match '
                       'TOA count {}.'.format(whitened_residuals.size,
                                              toas.size))

  current_end = float(np.max(toas))
  window_seconds = float(cadence_window_years) * YR_SEC
  global_epoch_indices = group_toa_epoch_indices(toas)
  global_epoch_centers = _epoch_centers(toas, global_epoch_indices)
  recent_global = _recent_positions(global_epoch_centers, current_end,
                                    window_seconds)
  recent_global_spacing = _mean_epoch_spacing(
      global_epoch_centers[recent_global])
  if recent_global_spacing is None:
    recent_global_spacing = _mean_epoch_spacing(global_epoch_centers)

  future_toas = []
  future_toaerrs = []
  future_backends = []
  future_residuals = []
  future_whitened = []
  future_freqs = []
  records = []

  for backend in np.unique(backend_vals):
    backend_mask = backend_vals == backend
    backend_toas = toas[backend_mask]
    backend_toaerrs = toaerrs[backend_mask]
    backend_freqs = freqs[backend_mask]
    backend_epochs = group_toa_epoch_indices(backend_toas)
    if not backend_epochs:
      continue
    backend_centers = _epoch_centers(backend_toas, backend_epochs)
    recent_backend = _recent_positions(backend_centers, current_end,
                                       window_seconds)
    if recent_backend.size > 0:
      template_epoch = int(recent_backend[-1])
      template_source = 'recent_year'
    else:
      template_epoch = len(backend_epochs) - 1
      template_source = 'latest_overall'
    template_idx = np.asarray(backend_epochs[template_epoch], dtype=int)
    template_toas = np.asarray(backend_toas[template_idx], dtype=float)
    template_toaerrs = np.asarray(backend_toaerrs[template_idx], dtype=float)
    template_freqs = np.asarray(backend_freqs[template_idx], dtype=float)
    template_offsets = template_toas - float(np.median(template_toas))

    spacing = None
    cadence_source = None
    if recent_backend.size >= 2:
      spacing = _mean_epoch_spacing(backend_centers[recent_backend])
      cadence_source = 'backend_recent_mean'
    if spacing is None:
      spacing = _mean_epoch_spacing(backend_centers)
      cadence_source = 'backend_full_span_mean'
    if spacing is None:
      spacing = recent_global_spacing
      cadence_source = 'pta_recent_mean'
    if spacing is None or not np.isfinite(spacing) or spacing <= 0.0:
      raise ValueError('Cannot infer a positive future cadence for {} backend '
                       '{}.'.format(psr.name, backend))

    min_offset = float(np.min(template_offsets)) if template_offsets.size else 0.0
    first_center = max(float(backend_centers[-1]) + spacing,
                       current_end - min_offset + 1.0)
    target_end = current_end + future_years * YR_SEC
    centers = np.arange(first_center, target_end + 0.5 * spacing, spacing,
                        dtype=float)
    if centers.size == 0 and target_end > current_end:
      centers = np.asarray([max(target_end, current_end - min_offset + 1.0)],
                           dtype=float)
    if centers.size == 0:
      continue

    efac, equad, ecorr = white_noise_resolver(backend)
    sigma_white = np.sqrt(float(efac)**2 * template_toaerrs**2 +
                          float(equad)**2)
    backend_future_toas = []
    backend_future_residuals = []
    for epoch_idx, center in enumerate(centers):
      epoch_toas = center + template_offsets
      if np.any(epoch_toas <= current_end):
        shift = current_end - float(np.min(epoch_toas)) + 1.0
        epoch_toas = epoch_toas + shift
      seed = _stable_seed(run_signature, psr.name, backend, epoch_idx,
                          '{:.6f}'.format(center))
      rng = np.random.RandomState(seed)
      shared_jitter = float(ecorr) * rng.normal() if float(ecorr) > 0.0 else 0.0
      epoch_residuals = sigma_white * rng.normal(size=sigma_white.size)
      epoch_residuals = epoch_residuals + shared_jitter
      backend_future_toas.append(epoch_toas)
      backend_future_residuals.append(epoch_residuals)

    backend_future_toas = np.concatenate(backend_future_toas)
    backend_future_residuals = np.concatenate(backend_future_residuals)
    future_toas.append(backend_future_toas)
    future_toaerrs.append(np.tile(template_toaerrs, centers.size))
    future_backends.append(np.repeat(str(backend), backend_future_toas.size))
    future_residuals.append(backend_future_residuals)
    future_freqs.append(np.tile(template_freqs, centers.size))
    if whitened_residuals is not None:
      future_whitened.append(np.asarray(backend_future_residuals, dtype=float))
    records.append({
        'backend': str(backend),
        'template_source': template_source,
        'cadence_source': cadence_source,
        'template_size': int(template_toas.size),
        'future_epochs': int(centers.size),
        'cadence_days': float(spacing / 86400.0),
    })

  if not future_toas:
    raise ValueError('Future extension produced no synthetic TOAs for {}.'
                     .format(psr.name))

  future_toas = np.concatenate(future_toas)
  future_toaerrs = np.concatenate(future_toaerrs)
  future_backends = np.concatenate(future_backends).astype(str)
  future_residuals = np.concatenate(future_residuals)
  future_freqs = np.concatenate(future_freqs)
  combined_toas = np.concatenate([toas, future_toas])
  combined_toaerrs = np.concatenate([toaerrs, future_toaerrs])
  combined_backends = np.concatenate([backend_vals, future_backends]).astype(str)
  combined_raw_residuals = np.concatenate([raw_residuals, future_residuals])
  combined_freqs = np.concatenate([freqs, future_freqs])
  future_mask = np.concatenate([
      np.zeros(toas.size, dtype=bool),
      np.ones(future_toas.size, dtype=bool),
  ])
  if whitened_residuals is None:
    combined_whitened = None
  else:
    combined_whitened = np.concatenate([whitened_residuals, future_residuals])

  if designmatrix is not None:
    designmatrix = np.asarray(designmatrix, dtype=float)
    if designmatrix.shape[0] != toas.size:
      raise ValueError('Design matrix row count {} does not match TOA count {}.'
                       .format(designmatrix.shape[0], toas.size))
    future_rows = np.zeros((future_toas.size, designmatrix.shape[1]),
                           dtype=float)
    combined_designmatrix = np.vstack([designmatrix, future_rows])
  else:
    combined_designmatrix = None

  flags = {'f': combined_backends}
  extended = SimpleNamespace(
      name=psr.name,
      toas=combined_toas,
      toaerrs=combined_toaerrs,
      backend_flags=combined_backends,
      flags=flags,
      residuals=combined_raw_residuals,
      freqs=combined_freqs,
      phi=getattr(psr, 'phi'),
      theta=getattr(psr, 'theta'),
      pdist=getattr(psr, 'pdist', None),
      designmatrix=combined_designmatrix,
      future_mask=future_mask,
      future_records=records,
  )
  if combined_whitened is not None:
    extended.whitened_residuals = combined_whitened

  metadata = {
      'future_years': future_years,
      'future_mask': future_mask,
      'future_records': records,
      'future_cadence_window_years': float(cadence_window_years),
      'future_residual_model': FUTURE_RESIDUAL_MODEL,
      'future_designmatrix_policy': FUTURE_DESIGNMATRIX_POLICY,
      'future_designmatrix_note': FUTURE_DESIGNMATRIX_NOTE,
  }
  return extended, metadata
