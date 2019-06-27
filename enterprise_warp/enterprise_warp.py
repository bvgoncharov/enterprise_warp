#!/bin/python

import numpy as np
import json
import glob
import os
import optparse

import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
import enterprise.signals.white_signals as white_signals
import enterprise.signals.gp_signals as gp_signals
from enterprise.pulsar import Pulsar
import enterprise.constants as const
import libstempo as T2
from enterprise_extensions import models

#import LT_custom
#import model_constants as mc

from scipy.special import ndtri

from matplotlib import pyplot as plt

def parse_commandline():
  """@parse the options given on the command-line.
  """
  parser = optparse.OptionParser()

  parser.add_option("-n", "--num", help="Pulsar number",  default=0, type=int)
  parser.add_option("-p", "--prfile", help="Parameter file", type=str)
  parser.add_option("-q", "--prfile2", help="Parameter file 2 for Bayes factor", default=None, type=str)
  parser.add_option("-i", "--images", help="Plots", default=1, type=int)

  parser.add_option("-o", "--onum", help="Other number",  default=0, type=int)
  parser.add_option("-c", "--sn_fcpr", help="Option to fix corner frequency parameter", default=None, type=float)

  opts, args = parser.parse_args()

  return opts

class ModelParams(object):
  def __init__(self,model_id):
    self.model_id = model_id

class Params(object):
  # Load parameters for how to run Enterprise
  def __init__(self, input_file_name, opts=None, custom_models_obj=None):
    self.input_file_name = input_file_name
    self.opts = opts
    self.psrs = list()
    self.Tspan = None
    self.custom_models_obj = custom_models_obj
    self.label_attr_map = {
      "datadir:": ["datadir", str],
      "out:": ["out", str],
      "overwrite:": ["overwrite", str],
      "allpulsars:": ["allpulsars", str],
      "gwb:": ["gwb", str],
      "gwb_lgApr:": ["gwb_lgApr", float, float],
      "gwb_gpr:": ["gwb_gpr", float, float],
      "noisefiles:": ["noisefiles", str],
      "physephem:": ["physephem", str],
      "efac:": ["efac", str],
      "efacpr:": ["efacpr", float, float],
      "efacsel:": ["efacsel", str],
      "equad:": ["equad", str],
      "equadpr:": ["equadpr", float, float],
      "equadsel:": ["equadsel", str],
      "ecorrpr:": ["ecorrpr", float, float],
      "ecorrsel:": ["ecorrsel", str],
      "sn_model:": ["sn_model", str],
      "sn_sincomp:": ["sn_sincomp", int],
      "sn_fourier_comp:": ["sn_fourier_comp", int],
      "sn_lgApr:": ["sn_lgApr", float, float],
      "sn_gpr:": ["sn_gpr", float, float],
      "rs_model:": ["rs_model", str],
      "rs_lgApr:": ["rs_lgApr", float, float],
      "rs_gpr:": ["rs_gpr", float, float],
      "dm_model:": ["dm_model", str],
      "dm_lgApr:": ["dm_lgApr", float, float],
      "dm_gpr:": ["dm_gpr", float, float],
      "dm_fcpr:": ["dm_fcpr", float, float],
      "dm_lgPpr:": ["dm_lgPpr", float, float],
      "dm_apr:": ["dm_apr", float, float],
      "sampler:": ["sampler", str],
      "dlogz:": ["dlogz", float],
      "nsamp:": ["nsamp", int],
      "nwalk:": ["nwalk", int],
      "ntemp:": ["ntemp", int],
      "setupsamp:": ["setupsamp", bool],
      "psrlist:": ["psrlist", str],
      "psrcachedir:": ["psrcachedir", str],
      "ssephem:": ["ssephem", str],
      "clock:": ["clock", str],
      "AMweight:": ["AMweight", int],
      "DMweight:": ["DMweight", int],
      "SCAMweight:": ["SCAMweight", int],
      "custom_commonpsr:": ["custom_commonpsr", str],
      "custom_singlepsr:": ["custom_singlepsr", str],
      "tm:": ["tm", str]
}
    if self.custom_models_obj is not None:
      self.label_attr_map.update(self.custom_models_obj().label_attr_map)
    model_id = None
    self.model_ids = list()
    self.__dict__['models'] = dict()

    with open(input_file_name, 'r') as input_file:
      for line in input_file:
        between_curly_brackets = line[line.find('{')+1 : line.find('}')]
        if between_curly_brackets.isdigit():
          model_id = int(between_curly_brackets)
          self.create_model(model_id)
          continue

        row = line.split()
        label = row[0]
        data = row[1:]  # rest of row is data list
        attr = self.label_attr_map[label][0]
        datatypes = self.label_attr_map[label][1:]
        if len(datatypes)==1 and len(data)>1:
          datatypes = [datatypes[0] for dd in data]

        values = [(datatypes[i](data[i])) for i in range(len(data))]

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
    self.clone_all_params_to_models()

  def override_params_using_opts(self):
    ''' If opts from command line parser has a non-None parameter argument,
    override this parameter for all models'''
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

  def set_default_params(self):
    ''' Setting default parameters here '''
    print('------------------')
    print('Setting default parameters with file ', self.input_file_name)
    #if 'ssephem' not in self.__dict__:
    #  self.__dict__['ssephem'] = 'DE436'
    #  print('Setting default Solar System Ephemeris: DE436')
    #if 'clock' not in self.__dict__:
    #  self.__dict__['clock'] = None
    #  print('Setting a default Enterprise clock convention (check the code)')
    if 'setupsamp' not in self.__dict__:
      self.__dict__['setupsamp'] = False
    if 'psrlist' in self.__dict__:
      self.psrlist = np.loadtxt(self.psrlist, dtype=np.unicode_)
      print('Only using pulsars from psrlist')
    else:
      self.__dict__['psrlist'] = []
      print('Using all available pulsars from .par/.tim directory')
    if 'psrcachedir' not in self.__dict__:
      self.psrcachedir = None
    if 'psrcachefile' not in self.__dict__:
      self.psrcachefile = None
    if 'tm' not in self.__dict__:
      self.tm = 'default'
      print('Setting a default linear timing model')
    if 'inc_events' not in self.__dict__:
      self.inc_events = True
      print('Including transient events to specific pulsar models')
    if 'sn_sincomp' not in self.__dict__:
      self.sn_sincomp = 2
      print('Setting number of Fourier sin-cos components to 2')
    if 'sn_fourier_comp' not in self.__dict__:
      self.sn_fourier_comp = 30
      print('Setting number of Fourier components to 30')

    # Model-dependent parameters
    for mkey in self.models:
      if 'rs_model' not in self.models[mkey].__dict__:
        self.models[mkey].rs_model = None
        print('Not adding red noise with selections for model',mkey)
      else:
        self.models[mkey].rs_model = read_json_dict(self.models[mkey].rs_model)
      if 'custom_commonpsr' not in self.models[mkey].__dict__:
        self.models[mkey].custom_commonpsr = ''
      if 'custom_singlepsr' not in self.models[mkey].__dict__:
        self.models[mkey].custom_singlepsr = ''

    print('------------------')

  def init_pulsars(self):
      directory = self.out
      
      cachedir = directory+'.psrs_cache/'
      if not os.path.exists(cachedir):
          os.makedirs(cachedir)
      
      if not self.psrcachefile==None or (not self.psrcachedir==None and not self.psrlist==[]):
          print('Attempting to load pulsar objects from cache')
          if not self.psrcachedir==None:
              psr_str = ''.join(sorted(self.psrlist)) + self.ssephem
              psr_hash = hashlib.sha1(psr_str.encode()).hexdigest()
              cached_file = self.psrcachedir + psr_hash
          if not self.psrcachefile==None:
              cached_file = self.psrcachefile
          if os.path.exists(cached_file):
              with open(cached_file, 'rb') as fin:
                  print('Loading pulsars from cache')
                  psrs_cache = pickle.load(fin)
          else:
              print('Could not load pulsars from cache: file does not exist')
              psrs_cache = None
      else:
          psrs_cache = None
          print('Condition for loading pulsars from cache is not satisfied')
      
      parfiles = sorted(glob.glob(self.datadir + '/*.par'))
      timfiles = sorted(glob.glob(self.datadir + '/*.tim'))
      print('Number of .par files: ',len(parfiles))
      print('Number of .tim files: ',len(timfiles))
      if len(parfiles)!=len(timfiles):
        print('Error - there should be the same number of .par and .tim files.')
        exit()
      
      if self.allpulsars=='True':
        self.output_dir = self.out
        if psrs_cache == None:
          print('Loading pulsars')
          self.psrlist_new = list()
          for p, t in zip(parfiles, timfiles):
            pname = p.split('/')[-1].split('_')[0].split('.')[0]
            if (pname in self.psrlist) or self.psrlist==[]:
                psr = Pulsar(p, t, ephem=self.ssephem, clk=self.clock,drop_t2pulsar=False)
                self.psrs.append(psr)
                self.psrlist_new.append(pname)
          print('Writing pulsars to cache.\n')
          psr_str = ''.join(sorted(self.psrlist_new)) + self.ssephem
          psr_hash = hashlib.sha1(psr_str.encode()).hexdigest()
          cached_file = cachedir + psr_hash
          with open(cached_file, 'wb') as fout:
            pickle.dump(self.psrs, fout)
        else:
          print('Using pulsars from cache')
          self.psrs = psrs_cache
        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in self.psrs]
        tmax = [p.toas.max() for p in self.psrs]
        self.Tspan = np.max(tmax) - np.min(tmin)
        #psr = []
        exit_message = "PTA analysis has already been carried out using a given parameter file"
      
      elif self.allpulsars=='False':
        self.psrs = Pulsar(parfiles[self.opts.num], timfiles[self.opts.num],drop_t2pulsar=False)#, ephem=params.ssephem, clk=params.clock)
        self.Tspan = self.psrs.toas.max() - self.psrs.toas.min() # observation time in seconds
        self.output_dir = self.out + str(self.opts.num)+'_'+self.psrs.name+'/'

        parfiles = parfiles[self.opts.num]
        timfiles = timfiles[self.opts.num]
        print('Current .par file: ',parfiles)
        print('Current .tim file: ',timfiles)
        directory += str(self.opts.num)+'_'+self.psrs.name+'/'
        exit_message = "This pulsar has already been processed"
        self.psrs = [self.psrs]
      self.directory = directory
      for mkey in self.models:
        self.models[mkey].directory = directory
        if not os.path.exists(directory):
          os.makedirs(directory)

def white_param_interpret(prior,selection,mark):#,m):
  """Take in a noise model "m" and noise-related input parameters
  and return an updated model"""

  #exec("global se; se=selections.Selection(selections.%s)" % (selection))
  se=selections.Selection(selections.__dict__[selection])

  # Interpret input in the prior of a parameter
  if not np.isscalar(prior):
    wp = parameter.Uniform(prior[0],prior[1])
  elif prior<=0:
    wp = parameter.Constant()
  elif prior>0:
    wp = parameter.Constant(prior)

  # Create a noise object
  if mark=='ef':
    w=white_signals.MeasurementNoise(efac=wp,selection=se)
    #if prior<0:
    #    for value in 
  elif mark=='eq':
    w=white_signals.EquadNoise(log10_equad=wp,selection=se)
  elif mark=='ec':
    se_ng=selections.Selection(selections.nanograv_backends)
    w=white_signals.EcorrKernelNoise(log10_ecorr=wp,selection=se_ng)
    #if prior<0:
    #  for value in pta._signal_dict['B1855+09_ecorr_sherman-morrison']._params:
    #        a{'B1855+09_430_PUPPI_efac'}

  return w

#class string(str):
#        def backwards(self):
            

def init_pta(params_all):
  """Initiate model and PTA for Enterprise.
  PTA for our code is just one pulsar, since we only
  do parameter estimation for pulsar noise"""

  ptas = dict.fromkeys(params_all.models)

  for ii, params in params_all.models.items():

    models = list()
    from_par_file = list()
    ecorrexists = np.zeros(len(params_all.psrs))
  
    try:
      params.noisefiles
      noisedict = get_noise_dict(psrlist=[p.name for p in params_all.psrs],noisefiles=params.noisefiles)
    except:
      noisedict = None
  
    # FOR RED NOISE MODEL SELECTION ONLY!!!
    # We do this, so that white and DM noise parameters are consistent for models
    if ii!=0 and noisedict !=None:
      noisedict_dm = get_noise_dict(psrlist=[p.name for p in params_all.psrs],noisefiles=params.noisefiles)
      noisedict_white = noisedict #_dm to make white noise params consistent for 2 models
    elif noisedict!=None:
      noisedict_dm = noisedict
      noisedict_white = noisedict_dm
  
    models = list()
  
    # Including parameters common for all pulsars
    if params.tm=='default':
      tm = gp_signals.TimingModel()
      #m.append('tm')
    elif params.tm=='ridge_regression':
      log10_variance = parameter.Uniform(-20, -10)
      basis = scaled_tm_basis()
      prior = ridge_prior(log10_variance=log10_variance)
      tm = gp_signals.BasisGP(prior, basis, name='ridge')
      #m.append('tm')
    #elif params.tm=='ffdot_separate':
    if params.tm!='none': m = tm
  
    if params.gwb=='none':
      print('GWB signal model is switched off in a parameter file')
    else:
      gwb_log10_A = parameter.Uniform(params.gwb_lgApr[0],params.gwb_lgApr[1])
      if params.gwb=='default':
        gwb_gamma = parameter.Uniform(params.gwb_gpr[0],params.gwb_gpr[1])
      elif params.gwb=='fixedgamma':
        gwb_gamma = parameter.Constant(4.33)
      gwb_pl = utils.powerlaw(log10_A=gwb_log10_A, gamma=gwb_gamma)
      orf = utils.hd_orf()
      gwb = gp_signals.FourierBasisCommonGP(gwb_pl, orf, components=30, name='gwb', Tspan=params.Tspan)
      #m += gwb
      m = m + gwb if params.tm!='none' else gwb
  
    for psp in params.custom_commonpsr.split():
        psp_model = params_all.custom_models_obj(psrname=psr.name,params=params)
        m += getattr(psp_model, psp)()
  
    if params.physephem=='none':
      print('Not fitting for Solar System ephemeris error')
    elif params.physephem=='default':
      eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)
      m += eph
  
    # Including parameters that can be separate for different pulsars
    for ii, psr in enumerate(params_all.psrs):
      # Add constant parameters to PTA from .par files
      #if params.allpulsars == 'True':
      #  from_par_file.append( T2.tempopulsar(parfile=pulpar[ii],timfile=pultim[ii]) )
      #elif params.allpulsars == 'False':
      #  from_par_file.append( T2.tempopulsar(parfile=pulpar,timfile=pultim) )
    
      # Determine if ecorr is mentioned in par file

      for key,val in psr.t2pulsar.noisemodel.items():
        if key.startswith('ecorr') or key.startswith('ECORR'):
          ecorrexists[ii]=True
  
      wnl = list()
      if ecorrexists[ii]:
        ec = white_param_interpret(params.ecorrpr,params.ecorrsel,'ec')#,m)
        wnl.append(ec)
      if params.efac=='default':
        ef = white_param_interpret(params.efacpr,params.efacsel,'ef')#,m)
        wnl.append(ef)
      if params.equad=='default':
        eq = white_param_interpret(params.equadpr,params.equadsel,'eq')#,m)
        wnl.append(eq)  
      for ww, wnm in enumerate(wnl):
        if ww==0:
          if params.tm=='none':
            m_sep = wnm
          else:
            m_sep = m + wnm
        else:
          m_sep += wnm

      for psp in params.custom_singlepsr:
        psp_model = params_all.custom_models_obj(psrname=psr.name,params=params)
        if getattr(psp_model, psp)() is not None:
          m_sep += getattr(psp_model, psp)()
  
      if params.sn_model=='none':
        print('Red/spin noise model is switched off in a parameter file')
      elif params.sn_model=='default':
        log10_A = parameter.Uniform(params.sn_lgApr[0],params.sn_lgApr[1])
        gamma = parameter.Uniform(params.sn_gpr[0],params.sn_gpr[1])
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma, \
            components=params.sn_sincomp)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=params.Tspan)
        m_sep += rn
      elif params.sn_model=='default_informed':
        log10_A = parameter.Uniform(noisedict[psr.name+'_log10_A'][1],noisedict[psr.name+'_log10_A'][2])
        gamma = parameter.Uniform(noisedict[psr.name+'_gamma'][1],noisedict[psr.name+'_gamma'][2])
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=params.Tspan)
        m_sep += rn
  
      elif params.sn_model=='free_spectrum':
        ncomp = 30
        log10_rho = parameter.Uniform(-10, -4, size=ncomp)
        pl = free_spectrum(log10_rho=log10_rho)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=ncomp, Tspan=params.Tspan)
        m_sep += rn
      else:
        print('Warning: red noise model in parameter file is not recognized!')
  
      if params.rs_model is not None:
        if psr.name in params.rs_model:
          rs_log10_A = parameter.Uniform(params.rs_lgApr[0],params.rs_lgApr[1])
          rs_gamma = parameter.Uniform(params.rs_gpr[0],params.rs_gpr[1])
          pl_rs = utils.powerlaw(log10_A=rs_log10_A, gamma=rs_gamma)
          rs_sel = selections.Selection(selections.__dict__\
              [params.rs_model[psr.name]])
          rs = gp_signals.FourierBasisGP(spectrum=pl_rs, components=30, \
              selection=rs_sel, Tspan=params.Tspan, name=params.rs_model[psr.name])
          m_sep += rs
   
      if params.dm_model=='none':
        print('DM noise model is switched off in a parameter file')
      elif params.dm_model=='default':
        dm_log10_A = parameter.Uniform(params.dm_lgApr[0],params.dm_lgApr[1])
        dm_gamma = parameter.Uniform(params.dm_gpr[0],params.dm_gpr[1])
        dm_basis= utils.createfourierdesignmatrix_dm(nmodes=30, Tspan=params.Tspan, fref=(4.15e3)**(0.5))
        dm_pl = utils.powerlaw(log10_A=dm_log10_A, gamma=dm_gamma)
        dm = gp_signals.BasisGP(dm_pl, dm_basis, name='dm_gp')
        m_sep += dm
      elif params.dm_model=='default_informed':
        dm_log10_A = parameter.Uniform(noisedict_dm[psr.name+'_dm_gp_log10_A'][1],noisedict_dm[psr.name+'_dm_gp_log10_A'][2])
        dm_gamma = parameter.Uniform(noisedict_dm[psr.name+'_dm_gp_gamma'][1],noisedict_dm[psr.name+'_dm_gp_gamma'][2])
        dm_basis= utils.createfourierdesignmatrix_dm(nmodes=30, Tspan=params.Tspan)
        dm_pl = utils.powerlaw(log10_A=dm_log10_A, gamma=dm_gamma)
        dm = gp_signals.BasisGP(dm_pl, dm_basis, name='dm_gp')
        m_sep += dm
      else:
        print('Warning: DM noise model in parameter file is not recognized!')
  
      models.append(m_sep(psr))

    pta = signal_base.PTA(models)
  
    # Set some white noise parameters to constants, if required
    constpar = {}
    #constpar.update(get_noise_from_pal2(pulnoise))

    print('Model',ii,'params order: ', pta.param_names)
    np.savetxt(params.directory+'/pars.txt', pta.param_names, fmt='%s')
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

def paramstringcut(mystring):
    # Cut string between " " symbols
    start = mystring.find( '"' )
    end = mystring.find( '":' )
    result = mystring[start+1:end]
    return result

def get_noise_from_pal2(noisefile):
    """ THIS SHOULD BE REPLACED LATER ON, IT IS FOR TESTING ONLY
    https://enterprise.readthedocs.io/en/latest/nano9.html"""
    psrname = noisefile.split('/')[-1].split('_noise.txt')[0]
    fin = open(noisefile, 'r')
    lines = fin.readlines()
    params = {}
    for line in lines:
        ln = line.split()
        if 'efac' in line:
            par = 'efac'
            flag = ln[0].split('efac-')[-1]
        elif 'equad' in line:
            par = 'log10_equad'
            flag = ln[0].split('equad-')[-1]
        elif 'jitter_q' in line:
            par = 'log10_ecorr'
            flag = ln[0].split('jitter_q-')[-1]
        elif 'RN-Amplitude' in line:
            par = 'log10_A'
            flag = ''
        elif 'RN-spectral-index' in line:
            par = 'gamma'
            flag = ''
        else:
            break
        if flag:
            name = [psrname, flag, par]
        else:
            name = [psrname, par]
        pname = '_'.join(name)
        params.update({pname: float(ln[1])})
    return params

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
    ''' get_noise_dict for only one pulsar '''
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
    ''' Load file to dictionary '''
    dictionary = dict()
    with open(filename) as ff:
        for line in ff:
            (key, val) = line.split()
            dictionary[key] = val
    return dictionary

