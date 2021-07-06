import numpy as np
import pandas as pd
import json
import glob
import os
import optparse
import warnings
import hashlib
import pickle

import enterprise.signals.parameter as parameter
from enterprise.signals import signal_base
import enterprise.signals.gp_signals as gp_signals
from enterprise.pulsar import Pulsar
import enterprise.constants as const
from enterprise_extensions import models
from .enterprise_models import StandardModels

try:
  from bilby import sampler as bimpler
except:
  warnings.warn("Warning: failed to import bilby.sampler")

def parse_commandline():
  """
  Parse the command-line options.

  Most important options:

  - --prfile: parameter file, the only option that must be set.
  - --num: index of a pulsar in a data directory (default: 0).

  See other options and their description in the code.
  """
  parser = optparse.OptionParser()

  parser.add_option("-n", "--num", help="Pulsar number",  default=0, type=int)
  parser.add_option("-p", "--prfile", help="Parameter file", type=str)
  parser.add_option("-d", "--drop", \
                    help="Drop pulsar with index --num in a full-PTA run \
                          (0 - No / 1 - Yes)", default=0, type=int)
  parser.add_option("-c", "--clearcache", \
                    help="Clear psrs cache file, associated with the run \
                          (to-do after changes to .par and .tim files)", \
                    default=0, type=int)
  parser.add_option("-m", "--mpi_regime", \
                    help="In MPI, manipulating with files and directories \
                          causes errors. So, we provide 3 regimes: \n \
                          (0) No MPI - run code as usual; \n \
                          (1) MPI preparation - manipulate files and prepare \
                          for the run (should be done outside MPI); \n \
                          (2) MPI run - run the code, assuming all necessary \
                          file manipulations have been performed. \n \
                          PolychordLite sampler in Bilby supports MPI",
                    default=0, type=int)
  parser.add_option("-w", "--wipe_old_output", \
                    help="Wipe contents of the output directory. Otherwise, \
                          the code will attempt to resume the previous run. \
                          Be careful: all subdirectories are removed too!", \
                    default=0, type=int)

  opts, args = parser.parse_args()

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
  """
  def __init__(self, input_file_name, opts=None, custom_models_obj=None,
               init_pulsars=True):
    self.input_file_name = input_file_name
    self.opts = opts
    self.psrs = list()
    self.Tspan = None
    self.custom_models_obj = custom_models_obj
    self.sampler_kwargs = {}
    self.label_attr_map = {
      "paramfile_label:": ["paramfile_label", str],
      "datadir:": ["datadir", str],
      "out:": ["out", str],
      "overwrite:": ["overwrite", str],
      "array_analysis:": ["array_analysis", str],
      "noisefiles:": ["noisefiles", str],
      "noise_model_file:": ["noise_model_file", str],
      "sampler:": ["sampler", str],
      "nsamp:": ["nsamp", int],
      "setupsamp:": ["setupsamp", bool],
      "mcmc_covm_csv:": ["mcmc_covm_csv", str],
      "psrlist:": ["psrlist", str],
      "ssephem:": ["ssephem", str],
      "clock:": ["clock", str],
      "AMweight:": ["AMweight", int],
      "DMweight:": ["DMweight", int],
      "SCAMweight:": ["SCAMweight", int],
      "tm:": ["tm", str],
      "fref:": ["fref", str]
}
    if self.custom_models_obj is not None:
      self.noise_model_obj = self.custom_models_obj
    else:
      self.noise_model_obj = StandardModels
    self.label_attr_map.update( self.noise_model_obj().get_label_attr_map() )
    model_id = None
    self.model_ids = list()
    self.__dict__['models'] = dict()

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
        attr = self.label_attr_map[label][0]
        datatypes = self.label_attr_map[label][1:]
        if len(datatypes)==1 and len(data)>1:
          datatypes = [datatypes[0] for dd in data]

        values = [(datatypes[i](data[i])) if not datatypes[i] is type(None) \
                  else int(data[i]) for i in range(len(data))]

        # Adding sampler kwargs to self.label_attr_map
        if attr == 'sampler' and 'bimpler' in globals():
          if data[0] in bimpler.IMPLEMENTED_SAMPLERS.keys():
            self.sampler_kwargs = bimpler.IMPLEMENTED_SAMPLERS[data[0]].\
                                    default_kwargs
            self.label_attr_map.update( dict_to_label_attr_map(\
                                        self.sampler_kwargs) )
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
    if 'ssephem' not in self.__dict__:
      self.__dict__['ssephem'] = 'DE436'
      print('Setting default Solar System Ephemeris: DE436')
    if 'clock' not in self.__dict__:
      self.__dict__['clock'] = None
      print('Setting a default Enterprise clock convention (check the code)')
    if 'setupsamp' not in self.__dict__:
      self.__dict__['setupsamp'] = False
    if 'psrlist' in self.__dict__:
      self.psrlist = np.loadtxt(self.psrlist, dtype=np.unicode_)
      print('Only using pulsars from psrlist')
    else:
      self.__dict__['psrlist'] = []
      print('Using all available pulsars from .par/.tim directory')
    if 'psrcachefile' not in self.__dict__:
      self.psrcachefile = None
    if 'tm' not in self.__dict__:
      self.tm = 'default'
      print('Setting a default linear timing model')
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
    # Copying default priors from StandardModels/CustomModels object
    # Priors are chosen not to be model-specific because HyperModel
    # (which is the only reason to have multiple models) does not support
    # different priors for different models
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
    if 'noise_model_file' in self.__dict__.keys():
      self.__dict__['noisemodel'] = read_json_dict(self.noise_model_file)
      self.__dict__['common_signals'] = self.noisemodel['common_signals']
      self.__dict__['model_name'] = self.noisemodel['model_name']
      self.__dict__['universal'] = self.noisemodel['universal']
      del self.noisemodel['common_signals']
      del self.noisemodel['universal']
      del self.noisemodel['model_name']
    # Reading model-specific noise model
    for mkey in self.models:
      if 'noise_model_file' in self.models[mkey].__dict__.keys():
        self.models[mkey].__dict__['noisemodel'] = read_json_dict(\
                                  self.models[mkey].noise_model_file)
        self.models[mkey].__dict__['common_signals'] = \
                                  self.models[mkey].noisemodel['common_signals']
        self.models[mkey].__dict__['model_name'] = \
                                  self.models[mkey].noisemodel['model_name']
        self.models[mkey].__dict__['universal'] = \
                                  self.models[mkey].noisemodel['universal']
        del self.models[mkey].noisemodel['common_signals']
        del self.models[mkey].noisemodel['model_name']
        del self.models[mkey].noisemodel['universal']
    self.label_models = '_'.join([self.models[mkey].model_name \
                                                    for mkey in self.models])

  def init_pulsars(self):
      """
      Initiate Enterprise pulsar objects.
      """

      cachedir = self.out+'.psrs_cache/'
      psrs_cache = None
      # Caching is disabled due to problems: Part 1
      #if not os.path.exists(cachedir):
      #  if self.opts.mpi_regime != 2:
      #    os.makedirs(cachedir)
      #
      #if not self.psrcachefile==None or (not self.psrlist==[]):
      #    print('Attempting to load pulsar objects from cache')
      #    if self.psrcachefile is not None:
      #        cached_file = self.psrcachefile
      #    else:
      #        psr_str = ''.join(sorted(self.psrlist)) + self.ssephem
      #        psr_hash = hashlib.sha1(psr_str.encode()).hexdigest()
      #        cached_file = cachedir + psr_hash

      #    if os.path.exists(cached_file):
      #        if bool(self.opts.clearcache):
      #            os.remove(cached_file)
      #            print('Cache file existed, but got removed, following \
      #                   command line options')
      #        else:
      #            with open(cached_file, 'rb') as fin:
      #                print('Loading pulsars from cache')
      #                psrs_cache = pickle.load(fin)
      #    else:
      #        print('Could not load pulsars from cache: file does not exist')
      #        psrs_cache = None
      #else:
      #    psrs_cache = None
      #    print('Condition for loading pulsars from cache is not satisfied')

      if '.pkl' in self.datadir:
        with open(self.datadir, 'rb') as pif:
          pkl_data = pickle.load(pif)
        parfiles = sorted([po.name+'.par' for po in pkl_data])
        timfiles = sorted([po.name+'.tim' for po in pkl_data])
        pkl_data = {pp: psrobj for pp, psrobj in zip(parfiles, pkl_data)}
      else:
        parfiles = sorted(glob.glob(self.datadir + '/*.par'))
        timfiles = sorted(glob.glob(self.datadir + '/*.tim'))
        print('Number of .par files: ',len(parfiles))
        print('Number of .tim files: ',len(timfiles))
      if len(parfiles)!=len(timfiles):
        print('Error: there should be the same number of .par and .tim files.')
        exit()

      if self.array_analysis=='True':
        self.output_dir = self.out + self.label_models + '_' + \
                          self.paramfile_label + '/'
        if psrs_cache == None:
          print('Loading pulsars')
          self.psrlist_new = list()
          for num, (p, t) in enumerate(zip(parfiles, timfiles)):
            pname = p.split('/')[-1].split('_')[0].split('.')[0]
            if (pname in self.psrlist) or self.psrlist==[]:
                if self.opts is not None:
                  if self.opts.drop and self.opts.num==num:
                    print('Dropping pulsar ', pname)
                    self.output_dir += str(num) + '_' + pname + '/'
                    continue
                if '.pkl' in self.datadir:
                  psr = pkl_data[p]
                else:
                  psr = Pulsar(p, t, ephem=self.ssephem, clk=self.clock, \
                               drop_t2pulsar=False)
                psr.__dict__['parfile_name'] = p
                psr.__dict__['timfile_name'] = t
                self.psrs.append(psr)
                self.psrlist_new.append(pname)
          # Caching is disabled due to problems: Part 2
          #print('Writing pulsars to cache.\n')
          #psr_str = ''.join(sorted(self.psrlist_new)) + self.ssephem
          #psr_hash = hashlib.sha1(psr_str.encode()).hexdigest()
          #cached_file = cachedir + psr_hash
          #with open(cached_file, 'wb') as fout:
          #  pickle.dump(self.psrs, fout)
        else:
          print('Using pulsars from cache')
          self.psrs = psrs_cache
        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in self.psrs]
        tmax = [p.toas.max() for p in self.psrs]
        self.Tspan = np.max(tmax) - np.min(tmin)
        #psr = []
        exit_message = "PTA analysis has already been carried out using a given parameter file"

      elif self.array_analysis=='False':
        if '.pkl' in self.datadir:
          self.psrs = psr = pkl_data[parfiles[self.opts.num]]
        else:
          self.psrs = Pulsar(parfiles[self.opts.num], timfiles[self.opts.num], \
                             drop_t2pulsar=False, \
                             ephem=self.ssephem) #, clk=self.clock)
        self.psrs.__dict__['parfile_name'] = parfiles[self.opts.num]
        self.psrs.__dict__['timfile_name'] = timfiles[self.opts.num]
        self.Tspan = self.psrs.toas.max() - self.psrs.toas.min() # observation time in seconds
        self.output_dir = self.out + self.label_models + '_' + \
                          self.paramfile_label + '/' + str(self.opts.num) + \
                          '_' + self.psrs.name + '/'

        parfiles = parfiles[self.opts.num]
        timfiles = timfiles[self.opts.num]
        print('Current .par file: ',parfiles)
        print('Current .tim file: ',timfiles)

        exit_message = "This pulsar has already been processed"
        self.psrs = [self.psrs]

      if self.opts is not None:
        if self.opts.mpi_regime != 2:
          if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
          elif bool(self.opts.wipe_old_output):
            warn_message = 'Warning: removing everything in ' + self.output_dir
            warnings.warn(warn_message)
            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

def init_pta(params_all):
  """
  Initiate enterprise signal models and enterprise.signals.signal_base.PTA.
  """

  ptas = dict.fromkeys(params_all.models)
  for ii, params in params_all.models.items():

    allpsr_model = params_all.noise_model_obj(psr=params_all.psrs,
                                              params=params)

    models = list()
    from_par_file = list()
    ecorrexists = np.zeros(len(params_all.psrs))

    # Including parameters common for all pulsars
    if params.tm=='default':
      tm = gp_signals.TimingModel()
    elif params.tm=='ridge_regression':
      log10_variance = parameter.Uniform(-20, -10)
      basis = scaled_tm_basis()
      prior = ridge_prior(log10_variance=log10_variance)
      tm = gp_signals.BasisGP(prior, basis, name='ridge')

    # Adding common noise terms for all pulsars
    # Only those common signals are added that are listed in the noise model
    # file, getting Enterprise models from the noise model object.
    if 'm_all' in locals():
      del m_all
    for psp, option in params.common_signals.items():
      if 'm_all' in locals():
        m_all += getattr(allpsr_model, psp)(option=option)
      else:
        m_all = tm + getattr(allpsr_model, psp)(option=option)

    # Including single pulsar noise models
    for pnum, psr in enumerate(params_all.psrs):

      singlepsr_model = params_all.noise_model_obj(psr=psr, params=params)

      # Determine if ecorr is mentioned in par file
      try:
        for key,val in psr.t2pulsar.noisemodel.items():
          if key.startswith('ecorr') or key.startswith('ECORR'):
            ecorrexists[pnum]=True
      except Exception as pint_problem:
        print(pint_problem)
        ecorrexists[pnum]=False

      # Add noise models
      if psr.name in params.noisemodel.keys():
        noise_model_dict_psr = params.noisemodel[psr.name]
      else:
        noise_model_dict_psr = params.universal
      for psp, option in noise_model_dict_psr.items():
        if 'm_sep' in locals():
          m_sep += getattr(singlepsr_model, psp)(option=option)
        elif 'm_all' in locals():
          m_sep = m_all + getattr(singlepsr_model, psp)(option=option)
        else:
          m_sep = tm + getattr(singlepsr_model, psp)(option=option)

      models.append(m_sep(psr))
      del m_sep

    pta = signal_base.PTA(models)

    if 'noisefiles' in params.__dict__.keys():
      noisedict = get_noise_dict(psrlist=[p.name for p in params_all.psrs],\
                                 noisefiles=params.noisefiles)
      print('For constant parameters using noise files in PAL2 format')
      pta.set_default_params(noisedict)

    print('Model',ii,'params (',len(pta.param_names),') in order: ', \
          pta.param_names)

    if params.opts is not None:
      if params.opts.mpi_regime != 2:
        np.savetxt(params.output_dir + '/pars.txt', pta.param_names, fmt='%s')
        
    ptas[ii]=pta

  return ptas

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

def get_noise_dict(psrlist,noisefiles):
    """
    Reads in list of pulsar names and returns dictionary
    of {parameter_name: value} for all noise parameters.
    By default the input list is None and we use the 34 pulsars used in
    the stochastic background analysis.
    """

    params = {}
    json_files = sorted(glob.glob(noisefiles + '*.json'))
    for ff in json_files:
        if any([pp in ff for pp in psrlist]):
            with open(ff, 'r') as fin:
                params.update(json.load(fin))
    return params

def get_noise_dict_psr(psrname,noisefiles):
    """
    get_noise_dict for only one pulsar
    """
    params = dict()
    with open(noisefiles+psrname+'_noise.json', 'r') as fin:
        params.update(json.load(fin))
    return params

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
