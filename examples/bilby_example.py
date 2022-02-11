# Example for accesing Bilby with Enterprise
# Adopted from https://enterprise.readthedocs.io/en/latest/mdc.html

from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
import enterprise.signals.white_signals as white_signals
import enterprise.signals.gp_signals as gp_signals
from enterprise.signals import signal_base
import glob

import bilby
from enterprise_warp import bilby_warp

parfiles = sorted(glob.glob('data/*.par'))
timfiles = sorted(glob.glob('data/*.tim'))

psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t)
    psrs.append(psr)

##### parameters and priors #####

# Uniform prior on EFAC
efac = parameter.Uniform(0.1, 5.0)

# white noise
ef = white_signals.MeasurementNoise(efac=efac)

# timing model
tm = gp_signals.TimingModel()

# full model is sum of components
model = ef + tm

# initialize PTA
pta = signal_base.PTA([model(psrs[0])])

priors = bilby_warp.get_bilby_prior_dict(pta)

parameters = dict.fromkeys(priors.keys())
likelihood = bilby_warp.PTABilbyLikelihood(pta,parameters)
label = 'test_bilby'
bilby.run_sampler(likelihood=likelihood, priors=priors, outdir='out/', label=label, sampler='dynesty', resume=True, nwalkers=500)
