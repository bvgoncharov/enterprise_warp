"""
To-dos:
 - We need to set defaults for label_attr_map, in case it is not supplied in the parameter file
"""

import numpy as np
import enterprise.constants as const
from enterprise.signals import signal_base
from enterprise.signals import utils
#import model_constants as mc
from enterprise_extensions import models
import enterprise.signals.parameter as parameter
import enterprise.signals.gp_signals as gp_signals
import enterprise.signals.white_signals as white_signals
import enterprise.signals.selections as selections
import my_selections

class StandardModels(object):
  def __init__(self,psrname=None,params=None):
    self.psrname = psrname
    self.params = params
    # Make sure that default value types are correct (add ".0" for float)
    # Dict values are default prior boundaries, if not set in param file
    self.priors = {
      "efac": [0., 10.],
      "equad": [-10., -5.],
      "ecorr": [-10., -5.],
      "sn_lgA": [-20., -6.],
      "sn_gamma": [0., 10.],
      "sn_fc": [-10., -6.],
      "dmn_lgA": [-20., -6.],
      "dmn_gamma": [0., 10.],
      "gwb_lgA": [-20., -6.],
      "syn_lgA": [-20., -6.],
      "syn_gamma": [0., 10.],
      "gwb_gamma": [0., 10.],
      "red_general_nfreqs": 30,
      "red_general_nfouriercomp": 2
    }

  def get_label_attr_map(self):
    """
    Convert self.priors dict to enterprise_warp.Params.label_attr_map dict
    """
    label_attr_map = dict()
    for key, val in self.priors.items():
      if hasattr(val,'__iter__'):
        lam_types = [type(val[0]) for ii in range(len(val))]
      else:
        lam_types = [type(val)]
      label_attr_map[key+':'] = [key] + lam_types
    return label_attr_map

  def get_default_prior(self, key):
    return self.priors[key]

  # Signle pulsar noise models

  def efac(self,option="by_backend"):
    if option not in selections.__dict__.keys():
      raise ValueError('EFAC option must be Enterprise selection function name')
    se=selections.Selection(selections.__dict__[option])
    efacpr = interpret_white_noise_prior(self.params.efac)
    efs = white_signals.MeasurementNoise(efac=efacpr,selection=se)
    return efs

  def equad(self,option="by_backend"):
    if option not in selections.__dict__.keys():
      raise ValueError('EQUAD option must be Enterprise selection function \
                        name')
    se=selections.Selection(selections.__dict__[option])
    equadpr = interpret_white_noise_prior(self.params.equad)
    eqs = white_signals.EquadNoise(log10_equad=equadpr,selection=se)
    return eqs

  def ecorr(self,option="by_backend"):
    if option not in selections.__dict__.keys():
      raise ValueError('ECORR option must be Enterprise selection function \
                        name')
    se=selections.Selection(selections.__dict__[option])
    ecorrpr = interpret_white_noise_prior(self.params.ecorr)
    ecs = white_signals.EcorrKernelNoise(efac=efacpr,selection=se)
    return efs

  def spin_noise(self,option="powerlaw"):
    log10_A = parameter.Uniform(self.params.sn_lgA[0],self.params.sn_lgA[1])
    gamma = parameter.Uniform(self.params.sn_gamma[0],self.params.sn_gamma[1])
    if option=="powerlaw":
      pl = utils.powerlaw(log10_A=log10_A, gamma=gamma, \
                          components=self.params.red_general_nfouriercomp)
    elif option=="turnover":
      fc = parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1])
      pl = powerlaw_bpl(log10_A=log10_A, gamma=gamma, fc=fc,
                        components=self.params.red_general_nfouriercomp)
    sn = gp_signals.FourierBasisGP(spectrum=pl, Tspan=self.params.Tspan,
                                   name='red_noise',
                                   components=self.params.red_general_nfreqs)
    return sn

  def dm_noise(self,option="powerlaw"):
    log10_A = parameter.Uniform(self.params.dmn_lgA[0],self.params.dmn_lgA[1])
    gamma = parameter.Uniform(self.params.dmn_gamma[0],self.params.dmn_gamma[1])
    if option=="powerlaw":
      pl = utils.powerlaw(log10_A=log10_A, gamma=gamma, \
                          components=self.params.red_general_nfouriercomp)
    elif option=="turnover":
      fc = parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1])
      pl = powerlaw_bpl(log10_A=log10_A, gamma=gamma, fc=fc,
                        components=self.params.red_general_nfouriercomp)
    dm_basis = utils.createfourierdesignmatrix_dm(\
               nmodes = self.params.red_general_nfreqs,\
               Tspan=self.params.Tspan, fref=self.params.fref)
    dmn = gp_signals.BasisGP(pl, dm_basis, name='dm_gp')
    return dmn

  #def system_noise(self,option=[]):
  #  log10_A = parameter.Uniform(self.params.syn_lgA[0],self.params.syn_lgA[1])
  #  gamma = parameter.Uniform(self.params.syn_gamma[0],self.params.syn_gamma[1])
  #  pl = utils.powerlaw(log10_A=log10_A, gamma=gamma, \
  #                      components=self.params.red_general_nfouriercomp)
  #  syn = gp_signals.FourierBasisGP(spectrum=pl, Tspan=self.params.Tspan,
  #                                  name='system_noise',
  #                                  selection=selections.Selection(selections.__dict__[system_name])
  #                                  components=self.params.red_general_nfreqs)
  #  return syn

  # Common noise for multiple pulsars

  def gwb(self,option="common_pl"):
    gwb_log10_A = parameter.Uniform(params.gwb_lgA[0],params.gwb_lgA[1])
    if option=="common_pl":
      gwb_gamma = parameter.Uniform(params.gwb_gamma[0],params.gwb_gamma[1])
    elif option=="fixed_gamma":
      gwb_gamma = parameter.Constant(4.33)
    gwb_pl = utils.powerlaw(log10_A=gwb_log10_A, gamma=gwb_gamma)
    orf = utils.hd_orf()
    gwb = gp_signals.FourierBasisCommonGP(gwb_pl, orf, \
                                 components=self.params.red_general_nfreqs, \
                                 name='gwb', Tspan=self.params.Tspan)
    return gwb

  def bayes_ephem(self,option="default"):
    eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)
    return eph


def interpret_white_noise_prior(prior):
  if not np.isscalar(prior):
    return parameter.Uniform(prior[0],prior[1])
  else:
    return parameter.Constant()

@signal_base.function
def powerlaw_bpl(f, log10_A=-16, gamma=5, fc=-9, components=2):
    """
    Broken power law red noise from Goncharov, Zhu, Thrane (2019).
    If fc<0 we assume we have lg(fc).
    """
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    if fc < 0 : fc = 10**fc
    return ((10**log10_A)**2 / 12.0 / np.pi**2 *
            const.fyr**(-3) * ((f+fc)/const.fyr)**(-gamma) * np.repeat(df, components))
