"""
The main module for running enterprise_warp:
- Reads command line arguments, parameter files (.dat). 
- Based on the input, creates a enterprise.PTA object, with methods to compute likelihoods and priors.
"""

import numpy as np
import json
import glob
import os
import optparse
import warnings
import hashlib
import pickle

from .utils import get_noise_dict, get_noise_dict_psr

try:
  import discovery as ds
  import discovery.const as const
  from .discovery_warp import init_pta_discovery
except Exception as ex:
  print(ex)
  warnings.warn("discovery is not available")

try:
  import enterprise.signals.parameter as parameter
  from enterprise.signals import signal_base
  import enterprise.signals.gp_signals as gp_signals
  from enterprise.pulsar import Pulsar
  import enterprise.constants as const
  from .enterprise_models import EnterpriseModels
  from .discovery_models import DiscoveryModels
except Exception as ex:
  print(ex)
  warnings.warn("enterprise is not available")

try:
  import pandas as pd
except Exception as ex:
  print(ex)
  warnings.warn("pandas is not available, required for mcmc_covm_csv parameter")

try:
  from mpi4py import MPI
  process_rank = MPI.COMM_WORLD.Get_rank()
except:
  process_rank = 0

try:
  from bilby import sampler as bimpler
except:
  warnings.warn("Warning: failed to import bilby.sampler")

class EWParser(object):
  def __init__(self):
    self.parser = optparse.OptionParser()

    self.parser.add_option("-n", "--num", \
      help="Pulsar number",  default=0, type=int)
    self.parser.add_option("-p", "--prfile", \
      help="Parameter file", type=str)
    self.parser.add_option("-d", "--drop", \
      help="Drop pulsar with index --num in a full-PTA run \
      (0 - No / 1 - Yes)", default=0, type=int)
    self.parser.add_option("-w", "--wipe_old_output", \
      help="Wipe contents of the output directory. Otherwise, \
      the code will attempt to resume the previous run. \
      Be careful: all subdirectories are removed too!", \
      default=0, type=int)
    self.parser.add_option("-x", "--extra_model_terms", \
      help="Extra noise terms to add to the .json noise model \
      file, a string that will be converted to dict. \
      E.g. {'J0437-4715': {'system_noise': \
      'CPSR2_20CM'}}. Extra terms are applied either on \
      the only model, or the second model.", \
      default='None', type=str)

def parse_commandline():
  """
  Parse the command-line options.

  Most important options:

  - ``--prfile``: parameter file, the only option that must be set.
  - ``--num``: index of a pulsar in a data directory (default: 0).

  See other options and their description in the code.

  Returns
  -------
  opts: optparse.OptionParser
    Command line arguments to be used later in the code.
  """

  ewp = EWParser()

  opts, args = ewp.parser.parse_args()

  return opts

class ModelParams(object):
  """
  A simple template for a new class. Used as a nested enterprise_warp.Params
  class for multiple model parameters within ont enterprise_warp.Params class.
  It is used with product-space sampling method and ptmcmcsampler, which
  evaluate posterior odds ratios (Bayes factors) for a given number of
  compared models.

  Parameters
  ----------
  model_id: int
    Index of a model
  """
  def __init__(self,model_id):
    self.model_id = model_id
    self.model_name = 'Untitled'

class Params(object):
  """
  Load parameters with instructions for how to run Enterprise.

  Parameters
  ----------
  input_file_name: str
    Path to enterprise_warp parameter file.
  opts: optparse.OptionParser
    The output of enterprise_warp.parse_commandline()
  custom_models_obj: enterprise_models.EnterpriseModels or a child class
    A class of enterprise_warp models to use later on (not initialized)
  init_pulsars: bool
    Initiate enterprise pulsars (use True by default)
  """
  def __init__(self, input_file_name, opts=None, custom_models_obj=None,
               init_pulsars=True):
    self.input_file_name = input_file_name
    self.opts = opts
    self.psrs = list()
    self.Tspan = None # over all pulsar ToAs
    self.sampler_kwargs = {}
    self.noisedict = {}
    self.label_attr_map = {
      "paramfile_label:": ["paramfile_label", str], 
      "pta_package:": ["pta_package", str], # enterprise/discovery
      "datadir:": ["datadir", str],
      "out:": ["out", str],
      "overwrite:": ["overwrite", str],
      "array_analysis:": ["array_analysis", int], # 0 / 1
      "timing_package:": ["timing_package", str], # tempo2 / pint
      "noisefiles:": ["noisefiles", str],
      "model_file:": ["model_file", str],
      "job_config_xlsx:": ["job_config_xlsx", str],
      "load_toa_filenames:": ["load_toa_filenames", str],
      "sampler:": ["sampler", str],
      "nsamp:": ["nsamp", int],
      "num_samples:": ["num_samples", int],
      "num_warmup:": ["num_warmup", int],
      "num_chains:": ["num_chains", int],
      "nuts_target_accept_prob:": ["nuts_target_accept_prob", float],
      "nuts_max_tree_depth:": ["nuts_max_tree_depth", int],
      "nuts_dense_mass:": ["nuts_dense_mass", int],
      "nuts_forward_mode_differentiation:": ["nuts_forward_mode_differentiation", int],
      "nuts_chain_method:": ["nuts_chain_method", str],
      "mcmc_covm_csv:": ["mcmc_covm_csv", str],
      "psrlist:": ["psrlist", str],
      "psrdistfile:": ["psrdistfile", str],
      "ssephem:": ["ssephem", str],
      "clock:": ["clock", str],
      "AMweight:": ["AMweight", int],
      "DMweight:": ["DMweight", int],
      "SCAMweight:": ["SCAMweight", int],
      "tm:": ["tm", str], # default / fast / ridge_regression
      "tm_svd:": ["tm_svd", int],
      "fref:": ["fref", str]
    }
    self.pta_package = get_pta_package(input_file_name)
    if self.pta_package=="enterprise":
      self.noise_model_obj = EnterpriseModels
    elif self.pta_package=="discovery":
      self.noise_model_obj = DiscoveryModels
    if custom_models_obj is not None:
      if self.pta_package=='discovery' and not custom_models_obj.__bases__[0] == DiscoveryModels:
        warnings.warn('Parameter pta_package is \'discovery\', but the custom model object is not based on discovery_models.DiscoveryModels. Using discovery_models.DiscoveryModels instead.')
      elif self.pta_package=='enterprise' and not custom_models_obj.__bases__[0] == EnterpriseModels:
        warnings.warn('Parameter pta_package is \'enterprise\', but the custom model object is not based on enterprise_models.EnterpriseModels. Using enterprise_models.EnterpriseModels instead.')
      else:
        self.noise_model_obj = custom_models_obj
        print('Using a custom noise model object:', custom_models_obj)
    self.label_attr_map.update( self.noise_model_obj().get_label_attr_map() )
    model_id = None
    self.model_ids = list()
    self.__dict__['models'] = dict()
    if self.opts is not None and self.opts.extra_model_terms != 'None':
      pn = str(list(eval(self.opts.extra_model_terms).values())[0])
      self.extra_term_label = pn.replace(\
                              '\'','').replace('[','').replace(']',\
                              '').replace(':','').replace('{',\
                              '').replace('}','').replace(' ','_')
    else:
      self.extra_term_label = ''

    with open(input_file_name, 'r') as input_file:
      for line in input_file:
        # Determining start of the new model, where line looks like this: {N}
        between_curly_brackets = line[line.find('{')+1 : line.find('}')]
        if between_curly_brackets.isdigit():
          model_id = int(between_curly_brackets)
          self.create_model(model_id)
          continue

        # Skipping comment lines:
        if line[0] == '#':
          continue

        row = line.split()
        label = row[0]
        data = row[1:]  # rest of row is data list

        if not label in self.label_attr_map.keys():
          if init_pulsars:
            raise ValueError('Uknown parameter: '+label)
          else:
            # For enterprise_warp.results, no pulsar initialization
            continue

        attr = self.label_attr_map[label][0]
        datatypes = self.label_attr_map[label][1:]
        if len(datatypes)==1 and len(data)>1:
          datatypes = [datatypes[0] for dd in data]

        values = [(datatypes[i](data[i])) if not datatypes[i] is type(None) \
                  else int(data[i]) for i in range(len(data))]

        # Adding sampler kwargs to self.label_attr_map
        if attr == 'sampler' and 'bimpler' in globals():
          if data[0] in bimpler.IMPLEMENTED_SAMPLERS.keys():
            self.sampler_kwargs = bimpler.IMPLEMENTED_SAMPLERS[data[0]].default_kwargs
            if type(self.sampler_kwargs) is dict:
              self.label_attr_map.update( dict_to_label_attr_map(\
                                          self.sampler_kwargs) )
            else:
              warnings.warn('sampler kwargs type:'+str(type(self.sampler_kwargs))+', expected dict')
              self.sampler_kwargs = {}
              warnings.warn('Reading sampler kwargs from enterprise_warp parameter files is not supported for the selected sampler.')
          else:
            error_message = 'Unknown sampler: ' + data[0] + '\n' + \
                            'Known samplers: ' + \
                            ', '.join(bimpler.IMPLEMENTED_SAMPLERS.keys())
            raise ValueError(error_message)

        if model_id == None:
          self.__dict__[attr] = values if len(values) > 1 else values[0]
        else:
          self.models[model_id].__dict__[attr] = \
                                values if len(values) > 1 else values[0]

    if not self.models:
      model_id = 0
      self.create_model(model_id)
    self.label = os.path.basename(os.path.normpath(self.out))
    self.override_params_using_opts()
    self.set_default_params()
    self.read_modeldicts()
    self.update_sampler_kwargs()
    if init_pulsars:
      self.init_pulsars()
      self.clone_all_params_to_models()

  def override_params_using_opts(self):
    """
    If opts from command line parser has a non-None parameter argument,
    override this parameter for all models.
    """
    if self.opts is not None:
      for key, val in self.models.items():
        for opt in self.opts.__dict__:
          if opt in self.models[key].__dict__ \
                  and self.opts.__dict__[opt] is not None:
            self.models[key].__dict__[opt] = self.opts.__dict__[opt]
            self.label+='_'+opt+'_'+str(self.opts.__dict__[opt])
            print('Model: ',key,'. Overriding parameter ',opt,' to ',\
              self.opts.__dict__[opt])
            print('Setting label to ',self.label)

  def clone_all_params_to_models(self):
    for key, val in self.__dict__.items():
      for mm in self.models:
        if key not in self.models[mm].__dict__.keys(): # This activates custom parameters/priors for every model, otherwise they are assumed the same for all models
          self.models[mm].__dict__[key] = val

  def create_model(self, model_id):
    self.model_ids.append(model_id)
    self.models[model_id] = ModelParams(model_id)

  def update_sampler_kwargs(self):
    print('Setting sampler kwargs from the parameter file:')
    for samkw in self.sampler_kwargs.keys():
      if samkw in self.__dict__.keys():
        self.sampler_kwargs[samkw] = self.__dict__[samkw]
        upd_message = 'Setting ' + samkw + ' to ' + str(self.__dict__[samkw])
        print(upd_message)
    print('------------------')

  def set_default_params(self):
    """
    Setting some default parameters here
    """
    print('------------------')
    print('Setting default parameters with file ', self.input_file_name)
    if self.pta_package == 'enterprise':
      globals()['init_pta'] = init_pta_enterprise
      print('PTA package: Enterprise (default).')
    elif self.pta_package == 'discovery':
      globals()['init_pta'] = init_pta_discovery
      print('PTA package: Discovery.')
    if 'timing_package' not in self.__dict__:
      # A keyword argument of enterprise.pulsar.Pulsar()
      self.__dict__['timing_package'] = 'tempo2'
    if 'ssephem' not in self.__dict__:
      self.__dict__['ssephem'] = 'DE438'
      print('Setting default Solar System Ephemeris: DE438')
    if 'clock' not in self.__dict__:
      self.__dict__['clock'] = None
      print('Setting a default Enterprise clock convention (check the code)')
    if 'psrlist' in self.__dict__:
      self.psrlist = np.loadtxt(self.psrlist, dtype=np.unicode_)
      print('Only using pulsars from psrlist:', self.psrlist)
    else:
      self.__dict__['psrlist'] = np.array([])
      print('Using all available pulsars from .par/.tim directory')
    if 'psrdistfile' not in self.__dict__:
        self.__dict__['psrdistfile'] = None
    if 'tm' not in self.__dict__:
      self.tm = 'default'
      print('Setting a default linear timing model')
    if 'tm_svd' not in self.__dict__:
      self.tm_svd = 0
      print('Setting timing model SVD to 0 (False)')
    if 'inc_events' not in self.__dict__:
      self.inc_events = True
      print('Including transient events to specific pulsar models')
    if 'fref' not in self.__dict__:
      self.fref = 1400 # MHz
      print('Setting reference radio frequency to 1400 MHz')
    if 'mcmc_covm_csv' in self.__dict__ and os.path.isfile(self.mcmc_covm_csv):
      print('MCMC jump covariance matrix is available')
      self.__dict__['mcmc_covm'] = pd.read_csv(self.mcmc_covm_csv, index_col=0)
    else:
      self.__dict__['mcmc_covm'] = None

    # Copying default priors from EnterpriseModels/CustomModels object
    # Priors are chosen not to be model-specific because HyperModel
    # (which is the only reason to have multiple models) does not support
    # different priors for different models
    # (This is actually enabled in clone_all_params_to_models)
    for prior_key, prior_default in self.noise_model_obj().priors.items():
      if prior_key not in self.__dict__.keys():
        self.__dict__[prior_key] = prior_default

    # Model-dependent parameters
    for mkey in self.models:

      self.models[mkey].modeldict = dict()

    print('------------------')

  def read_modeldicts(self):
    """
    Reading general noise model (which will overwrite model-specific ones,
    if they exists).
    """
    if 'model_file' in self.__dict__.keys():
      self.__dict__['noisemodel'] = read_json_dict(self.model_file)
      self.__dict__['common_signals'] = self.noisemodel['common_signals']
      self.__dict__['model_name'] = self.noisemodel['model_name']
      self.__dict__['to_remaining_psrs'] = self.noisemodel['to_remaining_psrs']
      self.__dict__['to_each_psr'] = self.noisemodel['to_each_psr']
      if self.opts.extra_model_terms != 'None':
        self.__dict__['noisemodel'] = merge_two_noise_model_dicts(\
                     self.__dict__['noisemodel'],\
                     eval(self.opts.extra_model_terms))
      del self.noisemodel['common_signals']
      del self.noisemodel['to_remaining_psrs']
      del self.noisemodel['to_each_psr']
      del self.noisemodel['model_name']
    # Reading model-specific noise model
    for mkey in self.models:
      if 'model_file' in self.models[mkey].__dict__.keys():
        self.models[mkey].__dict__['noisemodel'] = read_json_dict(\
                                  self.models[mkey].model_file)
        self.models[mkey].__dict__['common_signals'] = \
                                  self.models[mkey].noisemodel['common_signals']
        self.models[mkey].__dict__['model_name'] = \
                                  self.models[mkey].noisemodel['model_name']
        self.models[mkey].__dict__['to_remaining_psrs'] = \
                                  self.models[mkey].noisemodel['to_remaining_psrs']
        self.models[mkey].__dict__['to_each_psr'] = \
                                  self.models[mkey].noisemodel['to_each_psr']
        # Including an extra term only when there is only one model,
        # or only to the second model if there are two models compared.
        if self.opts is not None and self.opts.extra_model_terms != 'None' and \
           (len(self.models)==1 or (len(self.models)==2 and mkey==1)):
          self.models[mkey].__dict__['noisemodel'] = merge_two_noise_model_dicts(\
                                    self.models[mkey].__dict__['noisemodel'],\
                                    eval(self.opts.extra_model_terms))
        del self.models[mkey].noisemodel['common_signals']
        del self.models[mkey].noisemodel['model_name']
        del self.models[mkey].noisemodel['to_remaining_psrs']
        del self.models[mkey].noisemodel['to_each_psr']
    self.label_models = '_'.join([self.models[mkey].model_name \
                                                    for mkey in self.models])

  def selection_pulsars(self, psr_strings):
      """
      psr_strings: list of pulsar names or paths with pulsar names
      """
      psr_strings = sorted(psr_strings)
      if not self.array_analysis and len(psr_strings)>0:
        psr_strings = [psr_strings[self.opts.num]]
      else:
        # Skip pulsars with index self.opts.num (--num)
        if self.opts.drop:
          psr_strings = [pstr for ii, pstr in enumerate(psr_strings) if ii!=self.opts.num]
        # Skip pulsars which are not in the pulsar list
        if len(self.psrlist) > 0:
          psr_strings = [pstr for pstr in psr_strings if psrname_from_filename(pstr) in self.psrlist]
        
      return sorted(psr_strings)

  def init_pulsars(self):
      """
      Initiate Enterprise or Discovery pulsar objects.
      """
      # Pickled enterprise pulsars, simulation realizations
      if '.pkl' in self.datadir:
        # determine if datadir points to simulation realizations
        if '{:.0f}' in self.datadir:
          self.datadir = self.datadir.format(self.opts.num)
        with open(self.datadir, 'rb') as pif:
          self.psrs = pickle.load(pif)
        sel_p = self.selection_pulsars([psr.name for psr in self.psrs])
        self.psrs = [psr for psr in self.psrs if psr.name in sel_p]
        print('Loaded pulsars', [psr.name for psr in self.psrs])
        print('From', self.datadir)
        print('------------------')
      else:
        feathers = self.selection_pulsars(glob.glob(self.datadir + '/*.feather'))
        parfiles = self.selection_pulsars(glob.glob(self.datadir + '/*.par'))
        timfiles = self.selection_pulsars(glob.glob(self.datadir + '/*.tim'))
        if len(feathers)>0 and self.pta_package=='discovery':
          self.psrs = [ds.Pulsar.read_feather(ff) for ff in feathers]
        else:
          self.psrs = []
          print('Loading .par and .tim files from', self.datadir)
          for pp, tt in zip(parfiles, timfiles):
            print(pp.split('/')[-1],tt.split('/')[-1])
            psr = Pulsar(pp, tt, ephem=self.ssephem, 
                  clk=self.clock, 
                  drop_t2pulsar=False, 
                  timing_package=self.timing_package, 
                  distance_file=self.psrdistfile)
            psr.__dict__['parfile_name'] = pp
            psr.__dict__['timfile_name'] = tt
            if 'load_toa_filenames' in self.__dict__.keys() and \
                  self.load_toa_filenames=='True':
                  psr.__dict__['filenames'] = read_tim(t, column=1)
            self.psrs.append(psr)

            if self.pta_package=='discovery':
              if process_rank == 0:
                # Saving feather file for future use
                feather = pp.replace('par','feather')
                if 'noisefiles' in self.__dict__.keys():
                  noise_dict_psr = get_noise_dict_psr(psr.name, \
                        self.noisefiles)
                  self.validate_noisedict(noise_dict_psr)
                else:
                  noise_dict_psr = {}
                psr.to_feather(feather, noisedict=noise_dict_psr)
                print('Saved:',feather)
              feathers = self.selection_pulsars(glob.glob(self.datadir + '/*.feather'))
              self.psrs = [ds.Pulsar.read_feather(ff) for ff in feathers]
              print('------------------')

      # Determining Tspan
      tmin = [p.toas.min() for p in self.psrs]
      tmax = [p.toas.max() for p in self.psrs]
      self.Tspan = np.max(tmax) - np.min(tmin)
      print('N_psrs: ', len(self.psrs))
      print('Tspan [yr]:', self.Tspan/const.yr)
      print('------------------')

      # Loading noisefiles (if set)
      if 'noisefiles' in self.__dict__.keys():
        self.noisedict = get_noise_dict(psrlist=[p.name for p in self.psrs], noisefiles=self.noisefiles)
        self.validate_noisedict(self.noisedict)
      else:
        self.noisedict = {}

      # Setting pulsar noise parameters
      if self.pta_package=='discovery':
        for psr in self.psrs:
            noisedict = {par: val for par, val in self.noisedict.items() if par.startswith(psr.name) and par.endswith(('efac', 'equad', 'ecorr'))}
            psr.noisedict = noisedict

      # Creating an output directory
      self.output_dir = self.out + self.label_models + '_' + \
                        self.paramfile_label + '/' + \
                        self.extra_term_label + '/' + \
                        str(self.opts.num)
      if self.array_analysis:
        self.output_dir += '/'
      else:
        self.output_dir += '_' + self.psrs[0].name + '/'
      if self.opts is not None:
        if process_rank == 0:
          if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

  def validate_noisedict(self, noisedict):
    if 'noisefiles' in self.__dict__.keys():
      if not noisedict:
        raise ValueError("Noise dictionary is empty, check if noisefiles directory exists (noisefiles).")

def init_pta_enterprise(params_all):
  """
  Initiate enterprise signal models and enterprise.signals.signal_base.PTA.
  """

  ptas = dict.fromkeys(params_all.models)
  for ii, params in params_all.models.items():

    allpsr_model = params_all.noise_model_obj(psr=params_all.psrs,
                                              params=params)

    models = []

    # Including parameters common for all pulsars
    if params.tm=='default':
      tm = gp_signals.TimingModel(use_svd=bool(params.tm_svd))
    elif params.tm=='fast':
      tm = gp_signals.MarginalizingTimingModel(use_svd=bool(params.tm_svd))
    elif params.tm=='ridge_regression':
      log10_variance = parameter.Uniform(-20, -10)
      basis = scaled_tm_basis()
      prior = ridge_prior(log10_variance=log10_variance)
      tm = gp_signals.BasisGP(prior, basis, name='ridge')

    # Adding common signal/noise terms for all pulsars
    # Only those common signals are added that are listed in the noise model
    # file, getting Enterprise models from the noise model object.
    if 'pta_model' in locals():
      del pta_model
    for psp, option in params.common_signals.items():
      if 'pta_model' in locals():
        pta_model += getattr(allpsr_model, psp)(option=option)
      else:
        pta_model = tm + getattr(allpsr_model, psp)(option=option)

    # Including single pulsar noise models
    for pnum, psr in enumerate(params_all.psrs):

      singlepsr_model = params_all.noise_model_obj(psr=psr, params=params)

      if psr.name in params.noisemodel.keys():
        noise_model_dict_psr = params.noisemodel[psr.name]
      else:
        noise_model_dict_psr = params.to_remaining_psrs
      for psp, option in {**noise_model_dict_psr, **params.to_each_psr}.items():
        if 'psr_model' in locals():
          psr_model += getattr(singlepsr_model, psp)(option=option)
        elif 'pta_model' in locals():
          psr_model = pta_model + getattr(singlepsr_model, psp)(option=option)
        else:
          psr_model = tm + getattr(singlepsr_model, psp)(option=option)

      models.append(psr_model(psr))
      del psr_model

    pta = signal_base.PTA(models)

    if 'noisefiles' in params.__dict__.keys():
      print('Setting default PTA parameters based on noisefiles:',params_all.noisedict)
      pta.set_default_params(params_all.noisedict)

    print('Model',ii,'params (',len(pta.param_names),') in order: ', pta.param_names)

    if params.opts is not None:
      if process_rank == 0:
        np.savetxt(params.output_dir + '/pars.txt', pta.param_names, fmt='%s')
        
    ptas[ii]=pta

  return ptas

def psrname_from_filename(filename):
    """
    This function splits a strong based on symbols:
    ".", "/", "_". Thus, it converts any of the strings below:
    1. /path/0_J0437-4715.par
    2. /path/J0437-4715.tim
    3. J0437-4715
    Into pulsar name: J0437-4715
    """
    return filename.split('/')[-1].split('_')[0].split('.')[0]

def checkifconstpar(params):
    [doconstpar_ef, doconstpar_eq, doconstpar_ec] = [False, False, False]
    if np.isscalar(params.efacpr) and params.efacpr<0:
      doconstpar_ef = True
    if np.isscalar(params.equadpr) and params.equadpr<0:
      doconstpar_eq = True
    if np.isscalar(params.ecorrpr) and params.ecorrpr<0:
      doconstpar_ec = True
    return (doconstpar_ef or doconstpar_eq or doconstpar_ec)

def readconstpar(prior,noisemodel,mark,psrname,constpar):
    islg=''
    if not mark=='efac': islg='log10_'
    if np.isscalar(prior) and prior<0:
        for key,val in noisemodel.items():
            if key.startswith(mark):
              constpar_dictkey = psrname+'_'+val.flagval+'_'+islg+mark
              if mark=='efac': constpar[constpar_dictkey] = val.val
              else: constpar[constpar_dictkey] = np.log(val.val)
              #constpar[constpar_dictkey] = val.val
    return constpar

def read_json_dict(json_file):
    out_dict = dict()
    with open(json_file, 'r') as fin:
        out_dict.update(json.load(fin))
    return out_dict

def load_to_dict(filename):
    """
    Load file to Python dictionary
    """
    dictionary = dict()
    with open(filename) as ff:
        for line in ff:
            (key, val) = line.split()
            dictionary[key] = val
    return dictionary

def dict_to_label_attr_map(input_dict):
    """
    Converting python dict with one value into Params.label_attr_map format
    """
    return {key+':': [key, type(val)] for key, val in input_dict.items()}

def merge_two_noise_model_dicts(dict1, dict2):
    """
    Merging dict2 into dict1 given the noise model dict format:
    {'PSR_name': {'noise_term': parameter}}, parameter is either
    a string or a list of strings.
    """
    for psr in dict2.keys():
      if psr not in dict1.keys():
        dict1[psr] = dict2[psr]
      else:
        for noise_term in dict2[psr].keys():
          if noise_term in dict1[psr].keys() and type(dict1[psr][noise_term]) is list:
            dict1[psr][noise_term] = list(set(dict1[psr][noise_term] + dict2[psr][noise_term]))
          else:
            dict1[psr][noise_term] = dict2[psr][noise_term]
    return dict1

def read_tim(tim_file_name, column=1):
  """
  Returning column elements for all lines of .tim files (except commented)
  Column 1 is a file name. For example: t180327_095957.rf.pcm.dzTf8p.
  It is used to see what ToAs are from the same subband.
  """
  elements = list()
  with open(tim_file_name,'r') as tim:
    for line in tim:
      line_elements = line.split(' ')
      if len(line_elements)>2 and line_elements[0]=='':
        elements.append(line_elements[column])
  return np.array(elements)

def get_pta_package(filename):
  """
  Extracting pta_package from parameter name ahead of the rest
  """
  with open(filename) as f:
    return next(
        (line.split(":", 1)[1].strip() 
         for line in f 
         if line.startswith("pta_package:")),
        'enterprise'  # default, if not found
    )
