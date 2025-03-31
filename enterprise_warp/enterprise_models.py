"""
Documentation for enterprise_warp.models. Here, :class:`EnterpriseModels` has a set of methods which return enterprise signal objects. Names of the class methods are keys in .json noise model files, and method kwargs "option" must be a dict with values from .json noise model files. The class is used under the hood, unless the user choses to use their custom models. For custom models, it is recommended to create a child class of :class:`CustomModels`, with additional methods, and then pass it to :class:`Params`:

>>> custom = my_models.My_CustomModels
>>> params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=custom)
>>> pta = enterprise_warp.init_pta(params)

Examples of how to add signal and noise term to .json noise model files:

"measurement_noise": {
  "selection": "no_selection" 
}

Here, value "no_selection" is the name of the function in enterprise.selection or this Python file (globals()).

"spin_noise": {
    "psd": "powerlaw", # enterprise.gp_priors or 
    "n_freqs": 30 # n_days: 240
}

Here, "powerlaw" is function name from enterprise.gp_priors or this Python file (globals()). E.g., gp_priors has functions powerlaw, free_spectrum, whereas this file has a function powerlaw_bpl (see below).

"system_noise": {
    "psd": "powerlaw", # enterprise.gp_priors or 
    "n_freqs": 30 # n_days: 240
    "flags_flagvals": {
    	"group": []
    }
}

"global_gp": {
    "psd": "powerlaw",
    "n_freqs": 30, # n_days: 240
    "orf": "hd_orf",
    "parameters": {
      "log10_A": -15.0,
      "gamma": 4.33
    }
}

Here, "orf" value is a function name from enterprise.utils (hd_orf, monopole_orf, dipole_orf) or this Python file (globals()).
"""

import numpy as np

try:
  import enterprise.constants as const
  from enterprise.signals import signal_base
  from enterprise.signals import utils
  from enterprise.signals import gp_bases
  from enterprise.signals import gp_priors
  import enterprise.signals.parameter as parameter
  import enterprise.signals.gp_signals as gp_signals
  import enterprise.signals.deterministic_signals as deterministic_signals
  import enterprise.signals.white_signals as white_signals
  import enterprise.signals.selections as selections
  from enterprise.signals.parameter import function as parameter_function
except Exception as ex:
  print(ex)
  warnings.warn('enterprise is not available')

try:
  from mpi4py import MPI
  process_rank = MPI.COMM_WORLD.Get_rank()
except:
  process_rank = 0

from packaging.version import Version

import inspect
import types
import sys

class EnterpriseModels(object):
  """
  Standard models for pulsar timing analyses.

  Single-pulsar signals include white noise (efac, equad, ecorr), spin noise,
  DM noise, band noise, system noise, chromatic noise.
  Common signals include errors in Solar System ephemerides and
  gravitational-wave background with Hellings-Downs spatial correlations.

  Custom models should be derived from this class. See /examples.

  Parameters
  ----------
  psr: enterprise.pulsar.Pulsar
    Enterprise Pulsar object, where custom atrributes are written for
    band noise and system noise because enterprise.selections function
    have access to Pulsar object attributes to select certain parts of data.

    For common signals between multiple pulsars this parameter is not needed
    and can remain as "None".
  params: enterprise_warp.Params
    Parameter object, where default prior distribution parameters are added
    from self.priors if not specified in a parameter file (--prfile ...)

  Attributes
  ----------
  psr: enterprise.pulsar.Pulsar
    Enterprise Pulsar object
  params: enterprise_warp.Params
    Parameter object
  priors: dict
    Dictionary with keys being prior probability parameters for models
    or some model-specific settings, described in the current model object
    (if not hard-coded). Dictionary values serve as default parameters and as
    parameter format for input from parameter files.
  sys_noise_count: int
    Internal variable that counts how many times we created a new selection
    function for band noise, system noise, or any other noise with
    multiple terms for different segments of the data.
  """
  def __init__(self,psr=None,params=None):
    self.psr = psr
    self.params = params
    self.sys_noise_count = 0
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
      "dmn_fc": [-10., -6.],
      "chrom_idx": [0., 6.],
      "syn_lgA": [-20., -6.],
      "syn_gamma": [0., 10.],
      "gwb_lgA": [-20., -6.],
      "gwb_lgA_prior": "Uniform",
      "gwb_lgrho": [-10., -4.],
      "gwb_gamma": [0., 10.],
      "gwb_fc": [-10., -6.],
      "red_general_freqs": "tobs_60days",
      "red_general_nfouriercomp": 2
    }
    # For system_noise
    if self.psr is not None and type(self.psr) is not list:
      if not hasattr(self.psr,'sys_flags'):
        setattr(self.psr,'sys_flags',[])
        setattr(self.psr,'sys_flagvals',[])

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

  def measurement_noise(self, option={}):
    """
    EFAC + EQUAD in tempo2 format:
    sigma**2 = EFAC**2 * (toaerr**2 + EQUAD**2)

    option format: {
        "selection": "by_backend"
    }
    """
    if option["selection"] not in selections.__dict__.keys():
      raise ValueError('EFAC option must be Enterprise selection function name')
    se=selections.Selection(selections.__dict__[option["selection"]])
    efacpr = interpret_white_noise_prior(self.params.efac)
    equadpr = interpret_white_noise_prior(self.params.equad)
    efeq = white_signals.MeasurementNoise(efac=efacpr, log10_t2equad=equadpr, \
                                          selection=se)
    return efeq

  def efac(self, option={}):
    """
    EFAC signal:  multiplies ToA variance by EFAC**2, where ToA variance
    are diagonal components of the Likelihood covariance matrix.
    """
    if option["selection"] not in selections.__dict__.keys():
      raise ValueError('EFAC option must be Enterprise selection function name')
    se=selections.Selection(selections.__dict__[option["selection"]])
    efacpr = interpret_white_noise_prior(self.params.efac)
    efs = white_signals.MeasurementNoise(efac=efacpr,selection=se)
    return efs

  def equad(self, option={}):
    """
    EQUAD signal: adds EQUAD**2 to the ToA variance, where ToA variance
    are diagonal components of the Likelihood covariance matrix.
    TempoNest format: sigma**2 = EFAC**2 * toaerr**2 + EQUAD**2
    """
    if option["selection"] not in selections.__dict__.keys():
      raise ValueError('EQUAD option must be Enterprise selection function \
                        name')
    se=selections.Selection(selections.__dict__[option["selection"]])
    equadpr = interpret_white_noise_prior(self.params.equad)
    eqs = white_signals.TNEquadNoise(log10_tnequad=equadpr,selection=se)
    return eqs

  def ecorr(self, option={}):
    """
    Similar to EFAC and EQUAD, ECORR is a white noise parameter that
    describes a correlation between ToAs in a single epoch (observation).

    Arzoumanian, Zaven, et al. The Astrophysical Journal 859.1 (2018): 47.
    """
    if option["selection"] not in selections.__dict__.keys():
      raise ValueError('ECORR option must be Enterprise selection function \
                        name')
    se=selections.Selection(selections.__dict__[option["selection"]])
    ecorrpr = interpret_white_noise_prior(self.params.ecorr)
    ecs = white_signals.EcorrKernelNoise(log10_ecorr=ecorrpr,selection=se)
    return ecs

  def option_nfreqs(self, option, sel_func_name=None, selection_flag=None, selection_flagval=None):
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
      nfreqs = self.determine_nfreqs(sel_func_name=sel_func_name, cadence=option["ndays"])
    else:
      nfreqs = self.determine_nfreqs(sel_func_name=sel_func_name)
      
    return nfreqs

  def spin_noise(self,option={}):
    """
    Achromatic red noise process is called spin noise, although generally
    this model is used to model any unknown red noise. If this model is
    preferred over chromatic models then the observed noise is really spin
    noise, associated with pulsar rotational irregularities.
    """
    kwargs = {
        "log10_A": parameter.Uniform(self.params.sn_lgA[0],self.params.sn_lgA[1]),
        "gamma": parameter.Uniform(self.params.sn_gamma[0],self.params.sn_gamma[1]),
        "fc": parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1]),
    }
    nfreqs = self.option_nfreqs(option, sel_func_name=None)

    pl = get_enterprise_function(option, "psd", kwargs, gp_priors)
    
    sn = gp_signals.FourierBasisGP(spectrum=pl, Tspan=self.params.Tspan,
                                   name='red_noise', components=nfreqs)
    return sn

  def dm_noise(self,option={}):
    """
    A term to account for stochastic variations in DM. It is based on spin
    noise model, with Fourier amplitudes depending on radio frequency nu
    as ~ 1/nu^2.
    """
    kwargs = {
        "log10_A": parameter.Uniform(self.params.dmn_lgA[0],self.params.dmn_lgA[1]),
        "gamma": parameter.Uniform(self.params.dmn_gamma[0],self.params.dmn_gamma[1]),
        "fc": parameter.Uniform(self.params.dmn_fc[0],self.params.dmn_fc[1]),
    }
    nfreqs = self.option_nfreqs(option, sel_func_name=None)

    pl = get_enterprise_function(option, "psd", kwargs, gp_priors)

    dm_basis = utils.createfourierdesignmatrix_dm(nmodes = nfreqs,
                                                  Tspan=self.params.Tspan,
                                                  fref=self.params.fref)
    dmn = gp_signals.BasisGP(pl, dm_basis, name='dm_gp')

    return dmn

  def chromred(self,option={}):
    """
    This is an generalization of DM noise, with the dependence of Fourier
    amplitudes on radio frequency nu as ~ 1/nu^chi, where chi is a free
    parameter.

    Examples of chi:

    - Pulse scattering in the ISM: chi = 4 (Lyne A., Graham-Smith F., 2012,
      Pulsar astronomy)
    - Refractive propagation: chi = 6.4 (Shannon, R. M., and J. M. Cordes.
      MNRAS, 464.2 (2017): 2075-2089).
    """
    kwargs = {
        "log10_A": parameter.Uniform(self.params.sn_lgA[0],self.params.sn_lgA[1]),
        "gamma": parameter.Uniform(self.params.sn_gamma[0],self.params.sn_gamma[1]),
        "fc": parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1]),
    }
    nfreqs = self.option_nfreqs(option, sel_func_name=None)

    pl = get_enterprise_function(option, "psd", kwargs, gp_priors)

    if option["idx"]=="vary":
      idx = parameter.Uniform(self.params.chrom_idx[0], \
                              self.params.chrom_idx[1])
    else:
      idx = option

    chr_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=nfreqs,
                                                   Tspan=self.params.Tspan,
                                                   idx=idx)

    chrn = gp_signals.BasisGP(pl, chr_basis, name='chrom_gp')

    return chrn

  def system_noise(self,option={}):
    """
    Including red noise terms by "-group" flag, only with flagvals in noise
    model file.

    See Lentati, Lindley, et al. MNRAS 458.2 (2016): 2161-2187.
    """

    kwargs = {
        "log10_A": parameter.Uniform(self.params.sn_lgA[0],self.params.sn_lgA[1]),
        "gamma": parameter.Uniform(self.params.sn_gamma[0],self.params.sn_gamma[1]),
        "fc": parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1]),
    }

    pl = get_enterprise_function(option, "psd", kwargs, gp_priors)

    print("============")
    print("Adding system noise (count, flag, flagval, nfreqs, tspan [yr]): ")
    for flag, flagval_list in option["flags_flagvals"].items():
      for flagval in flagval_list:
        selection_function_name = 'sys_noise_selection_'+str(self.sys_noise_count)
        setattr(self, selection_function_name, 
                selection_factory(selection_function_name))
        nfreqs = self.option_nfreqs(option, selection_flag=flag, selection_flagval=flagval, sel_func_name=selection_function_name)
        tspan = self.determine_tspan(sel_func_name=selection_function_name)
        
        print(self.sys_noise_count, flag, flagval, nfreqs, tspan/const.yr)
        
        syn_term = gp_signals.FourierBasisGP(spectrum=pl, Tspan=tspan,
                                        name='system_noise_' + \
                                        str(self.sys_noise_count),
                                        selection=selections.Selection( \
                                        self.__dict__[selection_function_name] ),
                                        components=nfreqs)
        if self.sys_noise_count == 0:
          syn = syn_term
        else:
          syn += syn_term
  
        self.sys_noise_count += 1
        
    print("============")

    return syn

  # PTA-wide signals and noise
  
  def common_gp(self, option={}):
    """
    Common-spectrum red process (common red noise)
    More information: https://doi.org/10.3847/2041-8213/ac17f4
    """
    name = 'crn'
    nfreqs = self.option_nfreqs(option, sel_func_name=None)
    kwargs = {
        "log10_A": parameter.__dict__[self.params.gwb_lgA_prior]\
        	(self.params.sn_lgA[0],self.params.sn_lgA[1]),
        "gamma": parameter.Uniform(self.params.sn_gamma[0],self.params.sn_gamma[1]),
        "fc": parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1]),
        "log10_rho": parameter.Uniform(self.params.gwb_lgrho[0],
                                       self.params.gwb_lgrho[1],
                                       size=nfreqs)
    }

    pl = get_enterprise_function(option, "psd", kwargs, gp_priors)
    
    crn = gp_signals.FourierBasisGP(pl, components=nfreqs,
                            name='gw', Tspan=self.params.Tspan)
                                            
    return crn

  def global_gp(self, option={}):
    """
    Gaussian process with inter-pulsar correlations (e.g., Hellings-Downs)
    """
    name = option["orf"]
    nfreqs = self.option_nfreqs(option, sel_func_name=None)
    kwargs = {
        "log10_A": parameter.__dict__[self.params.gwb_lgA_prior]\
        	(self.params.sn_lgA[0],self.params.sn_lgA[1]),
        "gamma": parameter.Uniform(self.params.sn_gamma[0],self.params.sn_gamma[1]),
        "fc": parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1]),
        "log10_rho": parameter.Uniform(self.params.gwb_lgrho[0],
                                       self.params.gwb_lgrho[1],
                                       size=nfreqs)
    }

    pl = get_enterprise_function(option, "psd", kwargs, gp_priors)
    
    orf = get_enterprise_function(option, "orf", kwargs, utils)
    
    gwb = gp_signals.FourierBasisCommonGP(pl, orf, components=nfreqs,
                                            name='gw',
                                            Tspan=self.params.Tspan)
                                            
    return gwb
    
  def global_gp_2(self, option={}):
    """
    A second global_gp, in case two need to be added.
    """
    return self.global_gp(option=option)

  def global_gp_3(self, option={}):
    """
    A third global_gp, in case three need to be added.
    """
    return self.global_gp(option=option)

  def bayes_ephem(self,option={}):
    """
    Deterministic signal from errors in Solar System ephemerides.
    """
    eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)
    return eph

  # Utility functions for noise model object

  def determine_nfreqs(self, sel_func_name=None, cadence=60,
                       common_signal=False):
    """
    Determine whether to model red noise process with a fixed number of
    Fourier frequencies or whether to choose a number frequencies
    between the inverse of observation time and cadence (default 60) days.

    Parameters
    ----------
    sel_func_name: str
      Name of the selection function, stored in the current noise model
      object. It is needed to determine the observation span for a selected
      data (which is equal or smaller than the total observation span).
      If None, then enterprise.signals.selections.no_selection is assumed.
    cadence: float
      Period of highest-frequency component modelled (days)
    common_signal: bool
      True if determining a baseline observation span for a whole pulsar
      timing array with several pulsars.
    """

    if self.params.red_general_freqs.isdigit():
      n_freqs = int(self.params.red_general_freqs)
    elif self.params.red_general_freqs == "tobs_60days":
      tobs = self.determine_tspan(sel_func_name=sel_func_name,
                                  common_signal=common_signal)
      n_freqs = int(np.round((1./cadence/const.day - 1/tobs)/(1/tobs)))

    if self.params.opts is not None:
      if process_rank == 0:
        self.save_nfreqs_information(sel_func_name, n_freqs)

    return n_freqs

  def determine_tspan(self, sel_func_name=None, common_signal=False):
    """
    Determine the time span of TOAs under a given selection

    Parameters
    ----------
    sel_func_name: str
      Name of the selection function, stored in the current noise model
      object. It is needed to determine the observation span for a selected
      data (which is equal or smaller than the total observation span).
      If None, then enterprise.signals.selections.no_selection is assumed.
    common_signal: bool
      True if determining a baseline observation span for a whole pulsar
      timing array with several pulsars.
    """
    if common_signal:
      if not type(self.psr) is list:
        raise ValueError('Expecting a list of enterprise.pulsar.Pulsar objects \
                          in self.psr for a common signal')
      tmin_global = np.min([np.min(pp.toas) for pp in self.psr])
      tmax_global = np.max([np.max(pp.toas) for pp in self.psr])
      tspan = tmax_global - tmin_global
    else:
      if sel_func_name is None:
        selfunc = selections.no_selection
      else:
        selfunc = self.__dict__[sel_func_name]
      selection_mask = toa_mask_from_selection_function(self.psr, selfunc)
      toas = self.psr.toas[selection_mask]
      tspan = np.max(toas) - np.min(toas)

    return tspan

  def save_nfreqs_information(self, sel_func_name, n_freqs):
    """
    Enterprise does not store a number of Fourier frequencies for red noise
    processes. This function stores this information in the output folder.

    Parameters
    ----------
    sel_func_name: str
      Selection function name (None for no selection)
    n_freqs: int
      Number of Fourier frequencies of a red noise process
    """
    if sel_func_name is None:
      filename = 'no_selection'
      sel_func_name = 'none'
    else:
      filename = sel_func_name
    line = ''
    # System and band noise convention:
    if sel_func_name.split('_')[-1].isdigit():
      idx = int(sel_func_name.split('_')[-1])
      line += self.psr.sys_flags[idx]
      line += ';'
      line += self.psr.sys_flagvals[idx]
      line += ';'
    # Spin and DM noise convention: no number in selection func name
    else:
      line += 'no selection;-;'
    line += str(n_freqs)
    line += '\n'

    with open(self.params.output_dir + filename + '_nfreqs.txt', 'w') \
                                                           as comp_file:
      comp_file.write(line)

# General utility functions

def interpret_white_noise_prior(prior):
  """
  Interpret prior distribution parameters, passed from parameter file.
  Adding only one numbers sets prior to be a constant, while two numbers
  are interpreted as Uniform prior bounds.
  """
  if not np.isscalar(prior):
    return parameter.Uniform(prior[0],prior[1])
  elif np.isscalar(prior):
    return parameter.Constant(prior)
  else:
    raise ValueError('Unknown prior ', prior)

def get_enterprise_function(option, key, kwargs, module, custom_models={}):
    """
    Returns enterprise @signal_base function with set up priors, passing only those priors from kwargs which are relevant for this function. The function is collected from:
    1. This module (enterprise_warp/enteprirse_models.py)
    2. A relevant imported enterprise module (arg module)
    3. Custom models globals(), if applicable (kwarg custom_models)

    Arguments:
    option: dict, the standard kwarg of methods of EnterpriseModels.
    module: enterprise module name (e.g., gp_priors).
    key: str, a key in option (dict), the value of which corresponds a function in module. Example keys: "psd", "orf".
    kwargs: dict, all possible parameters for the function corresponding to option[key].

    custom_models: globals() if used in a child class of EnterpriseModels, to pass all local applicable functions. Otherwise, {}.

    Equivalent to returning `pl` here: 
    >>> log10_A = parameter.Uniform(self.params.sn_lgA[0],self.params.sn_lgA[1])
    >>> gamma = parameter.Uniform(self.params.sn_gamma[0],self.params.sn_gamma[1])
    >>> pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    """
    all_models = {**globals(), **module.__dict__, **custom_models}
    selected_func = all_models[option[key]]
    signature = inspect.signature(selected_func)
    kwargs = {kk: vv for kk, vv in kwargs.items() if kk in signature.parameters}
    return selected_func(**kwargs)

# Signal models

@signal_base.function
def regularized_powerlaw(f, log10_A=-16, gamma=5, components=2, min_psd_df=-20.):
    """
    We avoid numerical issues in the likelihood by introducing a minimum possible PSD value.
    Note, this value should be much below the minimum measurable value, such that the result
    obtained with this function is the same as with the standard power law model.

    min_psd_df=1e-20 corresponds to psd=1e-21 times df=1e-9
    For the reference on PSDs and sensitivity, see Goncharov, Thrane, Shannon (2022).
    """
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    psd_df =  (
        (10**log10_A) ** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma - 3) * f ** (-gamma) * np.repeat(df, components)
    )
    psd_df[psd_df<=10**(min_psd_df)] = 10**(min_psd_df)
    return psd_df

@signal_base.function
def powerlaw_bpl(f, log10_A=-16, gamma=5, fc=-9, components=2):
    """
    Broken power law red noise from Goncharov, Zhu, Thrane (2019):
    `arXiv:1910.05961 <https://arxiv.org/abs/1910.05961>`__
    If fc < 0, lg(fc) is assumed instead of fc.
    """
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    if fc < 0 : fc = 10**fc
    return ((10**log10_A)**2 / 12.0 / np.pi**2 *
            const.fyr**(-3) * ((f+fc)/const.fyr)**(-gamma) * np.repeat(df, components))

@parameter_function
def hd_orf_noauto(pos1, pos2):
    """Hellings & Downs spatial correlation function."""
    if np.all(pos1 == pos2):
        return 0
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5

# Selection functions

def selection_factory(new_selection_name):
  """
  This function constructs new selection functions for band and system noise,
  so that the specific band/system selection with flag and flag values (i.e.,
  "group" and "CPSR2_50CM") are passed as arrays "sys_flags" and "sys_flagvals"
  in enterprise.pulsar.Pulsar object. The selection function name contains an
  index for these arrays, which tell the function which flag and value to use.

  This method is not ideal, but it allows to create the right number of
  selection functions for any given number of band/system noise terms,
  without the need to pre-define them or modify the Enterprise code.

  Parameters
  ----------
  new_selection_name: str
    Python string with selection function name that we need to create.
    Selection function name format is "name_N", where N is an integer,
    an array index for "sys_flags" and "sys_flagvals".
  """

  def template_sel(flags,sys_flags,sys_flagvals):
    """
    Arguments "sys_flags" and "sys_flagvals" are variables
    inside Enterprise Pulsar object - they can be added there manually.
    They contain a list of flags and flagvals for system noise.
    """
    frame = inspect.currentframe()
    # Extracting array index from function name
    idx = int(inspect.getframeinfo(frame).function.split('_')[-1])
    if sys_flags==None or sys_flagvals==None:
        print('Kwargs sys_flags and sys_flagvals must be specified!')
        raise ValueError
    seldict = dict()

    seldict[sys_flagvals[idx]] = flags[sys_flags[idx]]==sys_flagvals[idx]
    return seldict

  list_codetype_args = [template_sel.__code__.co_argcount,
                        template_sel.__code__.co_nlocals,
                        template_sel.__code__.co_stacksize,
                        template_sel.__code__.co_flags,
                        template_sel.__code__.co_code,
                        template_sel.__code__.co_consts,
                        template_sel.__code__.co_names,
                        template_sel.__code__.co_varnames,
                        template_sel.__code__.co_filename,
                        new_selection_name,
                        template_sel.__code__.co_firstlineno,
                        template_sel.__code__.co_lnotab]

  #dealing with backwards-compatibility for types.CodeType
  sys_pyversion = Version(sys.version.split()[0])

  if Version("3.0") < sys_pyversion < Version("3.8"):
    list_codetype_args_ext = [template_sel.__code__.co_kwonlyargcount]
  elif sys_pyversion >= Version("3.8"):
    list_codetype_args_ext = [template_sel.__code__.co_posonlyargcount,\
                              template_sel.__code__.co_kwonlyargcount]
  else:
    raise ValueError("Python versions < 3.0 are not supported")

  list_codetype_args = list_codetype_args[:1] + \
                       list_codetype_args_ext + \
                       list_codetype_args[1:]

  template_selection_code = types.CodeType(*list_codetype_args)

  return types.FunctionType(template_selection_code, template_sel.__globals__,
                            new_selection_name)

def toa_mask_from_selection_function(psr,selfunc):
  """
  Create numpy array mask for ToA array, by applying selection function
  to enterprise.pulsar.Pulsar object.

  Parameters
  ----------
  psr: enterprise.pulsar.Pulsar
    Pulsar object in Enterprise format
  selfunc: function
    Selection function. Examples are in enterprise.signals.selections
  """
  args_selfunc = inspect.getargspec(selfunc).args
  argdict = {attr: getattr(psr,attr) for attr in dir(psr) \
                                              if attr in args_selfunc}
  selection_mask_dict = selfunc(**argdict)
  if len(selection_mask_dict.keys())==1:
    return [val for val in selection_mask_dict.values()][0]
  else:
    raise NotImplementedError
