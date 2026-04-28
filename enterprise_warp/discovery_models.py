"""
Documentation for discovery_warp.models.
"""
import warnings
import numpy as np

try:
  import discovery as ds
  from .discovery_warp import init_pta_discovery
except Exception as ex:
  print(ex)
  warnings.warn("discovery is not available")

from .utils import get_noise_dict
from .enterprise_models import EnterpriseModels

try:
  from mpi4py import MPI
  process_rank = MPI.COMM_WORLD.Get_rank()
except:
  process_rank = 0

from packaging.version import Version

import inspect
import types
import sys

class DiscoveryModels(EnterpriseModels):
  """
  EnterpriseModels repurposed to return Discovery signals 
  instead of Enterprise signals.
  """
  def __init__(self,psr=None,params=None):
    super(DiscoveryModels, self).__init__(psr=psr,params=params)
  
  def _white_noise_constant(self, parname, prior):
    """
    Return a constant value for a white-noise parameter if it is fixed.

    Mirrors interpret_white_noise_prior logic: scalar -> constant; iterable -> Uniform.
    """
    if parname in getattr(self.params, "noisedict", {}):
      return self.params.noisedict[parname]
    if prior is not None and np.isscalar(prior):
      return float(prior)
    return None

  def _efac_no_selection(self):
    """
    EFAC-only white noise when no backend selection is used.
    """
    efac = f'{self.psr.name}_efac'
    efac_val = self._white_noise_constant(efac, getattr(self.params, "efac", None))

    if efac_val is not None:
      noise = efac_val**2 * (self.psr.toaerrs**2)
      return ds.NoiseMatrix1D_novar(noise)
    else:
      toaerrs = ds.jnparray(self.psr.toaerrs)
      def getnoise(params):
        return params[efac]**2 * (toaerrs**2)
      getnoise.params = [efac]
      return ds.NoiseMatrix1D_var(getnoise)

  def _equad_no_selection(self):
    """
    EQUAD-only white noise when no backend selection is used.
    """
    log10_t2equad = f'{self.psr.name}_log10_t2equad'
    equad_val = self._white_noise_constant(log10_t2equad, getattr(self.params, "equad", None))

    if equad_val is not None:
      equad2 = 10.0**(2.0 * equad_val)
      noise = self.psr.toaerrs**2 + equad2
      return ds.NoiseMatrix1D_novar(noise)
    else:
      toaerrs = ds.jnparray(self.psr.toaerrs)
      def getnoise(params):
        return toaerrs**2 + 10.0**(2.0 * params[log10_t2equad])
      getnoise.params = [log10_t2equad]
      return ds.NoiseMatrix1D_var(getnoise)

  # Signle pulsar noise models

  def _measurement_noise_no_selection(self):
    efac = f'{self.psr.name}_efac'
    log10_t2equad = f'{self.psr.name}_log10_t2equad'

    efac_val = self._white_noise_constant(efac, getattr(self.params, "efac", None))
    equad_val = self._white_noise_constant(log10_t2equad, getattr(self.params, "equad", None))

    variable_params = []
    if efac_val is None:
      variable_params.append(efac)
    if equad_val is None:
      variable_params.append(log10_t2equad)

    toaerrs = ds.jnparray(self.psr.toaerrs)

    if not variable_params:
      noise = efac_val**2 * (toaerrs**2 + 10.0**(2.0 * equad_val))
      return ds.NoiseMatrix1D_novar(noise)

    def getnoise(params):
      ef = efac_val if efac_val is not None else params[efac]
      eq = equad_val if equad_val is not None else params[log10_t2equad]
      return ef**2 * (toaerrs**2 + 10.0**(2.0 * eq))
    getnoise.params = variable_params

    return ds.NoiseMatrix1D_var(getnoise)

  def measurement_noise(self, option={}):
    selection = option.get("selection", "by_backend")
    if selection == "no_selection":
      return self._measurement_noise_no_selection()

    if selection != "by_backend": # not in selections.__dict__.keys():
      raise ValueError('Only selection "no_selection" and "by_backend" are supported for Discovery, for now')
    else:
      se = ds.signals.selection_backend_flags
    measurement_noise_values = ds.makenoise_measurement(self.psr, noisedict=self.params.noisedict, selection=se)
    return measurement_noise_values

  def efac(self,option={}):
    """
    EFAC signal:  multiplies ToA variance by EFAC**2, where ToA variance
    are diagonal components of the Likelihood covariance matrix.
    """
    selection = option.get("selection", "by_backend")
    if selection == "no_selection":
      return self._efac_no_selection()

    if selection != "by_backend": # not in selections.__dict__.keys():
      raise ValueError('Only selection "no_selection" and "by_backend" are supported for Discovery, for now')
    else:
      se = ds.signals.selection_backend_flags

    efs = ds.makenoise_measurement(self.psr, noisedict=self.params.noisedict, selection=se)
    return efs


  def equad(self,option={}):
    """
    EQUAD signal: adds EQUAD**2 to the ToA variance, where ToA variance
    are diagonal components of the Likelihood covariance matrix.
    TempoNest format: sigma**2 = EFAC**2 * toaerr**2 + EQUAD**2
    """
    selection = option.get("selection", "by_backend")
    if selection == "no_selection":
      return self._equad_no_selection()

    if selection != "by_backend": # not in selections.__dict__.keys():
      raise ValueError('Only selection "no_selection" and "by_backend" are supported for Discovery, for now')
    else:
      se = ds.signals.selection_backend_flags

    eqs = ds.makenoise_measurement(self.psr, noisedict=self.params.noisedict, selection=se)
    return eqs

  def ecorr(self,option={}):
    """
    Similar to EFAC and EQUAD, ECORR is a white noise parameter that
    describes a correlation between ToAs in a single epoch (observation).

    Arzoumanian, Zaven, et al. The Astrophysical Journal 859.1 (2018): 47.
    """

    if option.get("selection", "by_backend") != "by_backend": # not in selections.__dict__.keys():
      raise ValueError('Only selection by_backend is supported for Discovery, for now')
    else:
      se = ds.signals.selection_backend_flags

    ecs = ds.makegp_ecorr(self.psr, noisedict=self.params.noisedict, enterprise=False, scale=1.0, selection=se, name='ecorrGP')
    return ecs

  def spin_noise(self, option={}):
    """
    Achromatic red noise process is called spin noise, although generally
    this model is used to model any unknown red noise. If this model is
    preferred over chromatic models then the observed noise is really spin
    noise, associated with pulsar rotational irregularities.
    """
    nfreqs = self.option_nfreqs(option, sel_func_name=None)
    pl = ds.__dict__[option["psd"]]

    sn = ds.makegp_fourier(self.psr, pl, components=nfreqs, name='red_noise')
    return sn
  
  def dm_noise(self,option="powerlaw"):
    """
    A term to account for stochastic variations in DM. It is based on spin
    noise model, with Fourier amplitudes depending on radio frequency nu
    as ~ 1/nu^2.
    """
    nfreqs = self.option_nfreqs(option, sel_func_name=None)
    pl = ds.__dict__[option["psd"]]

    dmn = ds.makegp_fourier(self.psr, pl, components=nfreqs, fourierbasis=ds.dmfourierbasis, name='dm_gp')
    return dmn
  
  def common_gp(self, option={}):
    """
    Common-spectrum red process (common red noise)
    More information: https://doi.org/10.3847/2041-8213/ac17f4
    """
    nfreqs = self.option_nfreqs(option, sel_func_name=None)
    pl = ds.__dict__[option["psd"]]
    return ds.makecommongp_fourier(self.params.psrs, pl, nfreqs, self.params.Tspan, name='crn', common=['crn_log10_A', 'crn_gamma','crn_log10_rho'])

  def global_gp(self, option={}):
    """
    Gaussian process with inter-pulsar correlations (e.g., Hellings-Downs)
    """
    name = option["orf"]
    nfreqs = self.option_nfreqs(option, sel_func_name=None)
    pl = ds.__dict__[option["psd"]]
    orf = ds.__dict__[option["orf"]]
    return ds.makeglobalgp_fourier(self.params.psrs, pl, orf, nfreqs, self.params.Tspan, name='gw')
