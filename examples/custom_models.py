from enterprise.signals import signal_base
import model_constants as mc
from enterprise_extensions import models

class CustomModels(object):
    def __init__(self,psrname=None,params=None):
      self.psrname = psrname
      self.params = params
      self.label_attr_map = dict()
      self.label_attr_map = {
        "my_amp:": ["my_amp", float, float],
        "my_cc:": ["my_cc", float, float],
        "event_j1713:" ["event_j1713", float, float]
      }
    def my_powerlaw(self):
      amp = parameter.Uniform(self.params.my_amp[0],self.params.my_amp[1])
      cc = parameter.Uniform(self.params.my_cc[0],self.params.my_cc[1])
      pl = powerlaw_my(amp=amp, cc=cc)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=params.components, Tspan=params.Tspan,name='my_powerlaw')
      return rn

    def event_j1713(self):
      if self.psrname=='J1713+0747':
        j1713_dmexp = models.dm_exponential_dip(tmin=self.params.event_j1713[0]\
          tmax=self.params.event_j1713[1])
        return j1713_dmexp

@signal_base.function
def powerlaw_my(f, amp=1e2, cc=15):
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    return amp * ((f+cc)/const.fyr)**(-2) * np.repeat(df, components))
