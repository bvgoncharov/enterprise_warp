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

  # Signle pulsar noise models

  def measurement_noise(self, option={}):
    if option["selection"] != "by_backend": # not in selections.__dict__.keys():
      raise ValueError('Only selection by_backend is supported for Discovery, for now')
    else:
      se = ds.signals.selection_backend_flags
    measurement_noise_values = ds.makenoise_measurement(self.psr, noisedict=self.params.noisedict, selection=se)
    return measurement_noise_values

  def efac(self,option={}):
    """
    EFAC signal:  multiplies ToA variance by EFAC**2, where ToA variance
    are diagonal components of the Likelihood covariance matrix.
    """
    if option["selection"] != "by_backend": # not in selections.__dict__.keys():
      raise ValueError('Only selection by_backend is supported for Discovery, for now')
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
    if option["selection"] != "by_backend": # not in selections.__dict__.keys():
      raise ValueError('Only selection by_backend is supported for Discovery, for now')
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
    if option["selection"] != "by_backend": # not in selections.__dict__.keys():
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
    pl = ds.powerlaw

    sn = ds.makegp_fourier(self.psr, pl, components=nfreqs, name='red_noise')
    return sn
  
  def dm_noise(self,option="powerlaw"):
    """
    A term to account for stochastic variations in DM. It is based on spin
    noise model, with Fourier amplitudes depending on radio frequency nu
    as ~ 1/nu^2.
    """
    nfreqs = self.option_nfreqs(option, sel_func_name=None)
    pl = ds.powerlaw

    dmn = ds.makegp_fourier(self.psr, pl, components=nfreqs, fourierbasis=ds.dmfourierbasis, name='dm_gp')
    return dmn
  
  def common_gp(self, option={}):
    """
    Common-spectrum red process (common red noise)
    More information: https://doi.org/10.3847/2041-8213/ac17f4
    """
    nfreqs = self.option_nfreqs(option, sel_func_name=None, common_signal=True)
    pl = ds.__dict__[option["psd"]]
    return ds.makecommongp_fourier(self.params.psrs, pl, nfreqs, self.params.Tspan, name='gw', common=['crn_log10_A', 'crn_gamma'])

  def global_gp(self, option={}):
    """
    Gaussian process with inter-pulsar correlations (e.g., Hellings-Downs)
    """
    name = option["orf"]
    nfreqs = self.option_nfreqs(option, sel_func_name=None, common_signal=True)
    pl = ds.__dict__[option["psd"]]
    orf = ds.__dict__[option["orf"]]
    return ds.makeglobalgp_fourier(self.params.psrs, pl, orf, nfreqs, self.params.Tspan, name='gw')
  
  def option_nfreqs(self, option, sel_func_name=None, selection_flag=None, selection_flagval=None, common_signal=False):
    """
    Selecting and removing nfreqs from option, otherwise from 1/Tobs to 1/60days
    """

    # For determining T_span
    if selection_flag is not None:
        self.psr.sys_flags.append(selection_flag)
        self.psr.sys_flagvals.append(selection_flagval)
    
    if "n_freqs" in option.keys():
        nfreqs = option["n_freqs"]
    elif "n_days" in option.keys():
        nfreqs = self.determine_nfreqs(sel_func_name=sel_func_name, cadence=option["ndays"], common_signal=common_signal)
    else:
        nfreqs = self.determine_nfreqs(sel_func_name=sel_func_name, common_signal=common_signal)
      
    return nfreqs
  