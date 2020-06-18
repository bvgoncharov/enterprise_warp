import numpy as np
import enterprise.constants as const
from enterprise.signals import signal_base
from enterprise_extensions import models
import enterprise.signals.parameter as parameter
import enterprise.signals.gp_signals as gp_signals
import enterprise.signals.selections as selections

from enterprise_warp.enterprise_models import StandardModels

class CustomModels(StandardModels):
    """
    Please follow this example to add your own models for enterprise_warp.
    """
    def __init__(self,psr=None,params=None):
      super(CustomModels, self).__init__(psr=psr,params=params)
      self.priors.update({
        "my_amp": [1e2, 1e4],
        "my_cc": [15.0, 18.0],
        "event_j1713_t0": [54500., 54900.]
      })

    def my_powerlaw(self,option="default"):
      """
      Example for custom power-law red noise with parameters amp and cc
      """
      amp = parameter.Uniform(self.params.my_amp[0],self.params.my_amp[1])
      cc = parameter.Uniform(self.params.my_cc[0],self.params.my_cc[1])
      pl = powerlaw_my(amp=amp, cc=cc, \
                       components=self.params.red_general_nfouriercomp)
      nfreqs = self.determine_nfreqs(sel_func_name=None)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=nfreqs, \
                                     Tspan=self.params.Tspan,name='my_powerlaw')
      return rn

    def event_j1713(self,option="default"):
      """
      Example for borrowing a signal from enterprise_extensions
      """
      if self.psr.name=='J1713+0747':
        j1713_dmexp=models.dm_exponential_dip(\
                                    tmin=self.params.event_j1713_t0[0],\
                                    tmax=self.params.event_j1713_t0[1])
        return j1713_dmexp

    def add_system_noise_selection_markers(self,markers):
      self.psrs[0].__dict__['custom_flag_bor'] = 'group'
      self.psrs[0].__dict__['flagval_bor'] = 'PDFB_10CM'

@signal_base.function
def powerlaw_my(f, amp=1e2, cc=15, components=2):
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    return amp * ((f+cc)/const.fyr)**(-2) * np.repeat(df, components)
