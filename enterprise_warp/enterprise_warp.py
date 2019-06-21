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
import enterprise.constants as const
import libstempo as T2
import libstempo.toasim as LT
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
  def __init__(self, input_file_name, opts=None, custom_models=None):
    self.input_file_name = input_file_name
    self.opts = opts
    label_attr_map = {
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
      "sn_lgApr:": ["sn_lgApr", float, float],
      "sn_gpr:": ["sn_gpr", float, float],
      "sn_fcpr:": ["sn_fcpr", float, float],
      "sn_lgPpr:": ["sn_lgPpr", float, float],
      "sn_apr:": ["sn_apr", float, float],
      "sn_bpr:": ["sn_bpr", float, float],
      "sn_amp:": ["sn_amp", float, float],
      "sn_cc:": ["sn_cc", float, float],
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
    self.label_attr_map.update(self.custom_models.label_attr_map)
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
        attr = label_attr_map[label][0]
        datatypes = label_attr_map[label][1:]

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

    # Model-dependent parameters
    for mkey in self.models:
      if 'rs_model' not in self.models[mkey].__dict__:
        self.models[mkey].rs_model = None
        print('Not adding red noise with selections for model',mkey)
      else:
        self.models[mkey].rs_model = read_json_dict(self.models[mkey].rs_model)

    print('------------------')

def init_pulsars(params):
    directory = params.out
    
    #all_params = [Params(opts.prfile)]
    #try:
    #    all_params.append(Params(opts.prfile2))
    #    directory = all_params[1].out
    #    print('Working with two parameter files')
    #except:
    #    directory = all_params[0].out
    #    print('Working with one parameter file')
    
    cachedir = directory+'.psrs_cache/'
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    
    if not params.psrcachefile==None or (not params.psrcachedir==None and not params.psrlist==[]):
        print('Attempting to load pulsar objects from cache')
        if not params.psrcachedir==None:
            psr_str = ''.join(sorted(params.psrlist)) + params.ssephem
            psr_hash = hashlib.sha1(psr_str.encode()).hexdigest()
            cached_file = params.psrcachedir + psr_hash
        if not params.psrcachefile==None:
            cached_file = params.psrcachefile
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
    
    parfiles = sorted(glob.glob(params.datadir + '/*.par'))
    timfiles = sorted(glob.glob(params.datadir + '/*.tim'))
    print('Number of .par files: ',len(parfiles))
    print('Number of .tim files: ',len(timfiles))
    if len(parfiles)!=len(timfiles):
      print('Error - there should be the same number of .par and .tim files.')
      exit()
    
    if params.allpulsars=='True':
      if psrs_cache == None:
        print('Loading pulsars')
        psrs = []
        params.psrlist_new = list()
        for p, t in zip(parfiles, timfiles):
          pname = p.split('/')[-1].split('_')[0].split('.')[0]
          if (pname in params.psrlist) or params.psrlist==[]:
            #try:
              psr = Pulsar(p, t, ephem=params.ssephem, clk=params.clock)
              psrs.append(psr)
              #except Exception as failpulsarobj:
              #  print('[x] Could not load pulsar ',pname)
              #  print('Could be due to too many TOAs, for example')
              params.psrlist_new.append(pname)
        print('Writing pulsars to cache.\n')
        psr_str = ''.join(sorted(params.psrlist_new)) + params.ssephem
        psr_hash = hashlib.sha1(psr_str.encode()).hexdigest()
        cached_file = cachedir + psr_hash
        with open(cached_file, 'wb') as fout:
          pickle.dump(psrs, fout)
      else:
        print('Using pulsars from cache')
        psrs = psrs_cache
      # find the maximum time span to set GW frequency sampling
      tmin = [p.toas.min() for p in psrs]
      tmax = [p.toas.max() for p in psrs]
      Tspan = np.max(tmax) - np.min(tmin)
      #psr = []
      exit_message = "PTA analysis has already been carried out using a given parameter file"
    
    elif params.allpulsars=='False':
      psrs = Pulsar(parfiles[opts.num], timfiles[opts.num])#, ephem=params.ssephem, clk=params.clock)
      Tspan = psrs.toas.max() - psrs.toas.min() # observation time in seconds
      raj = psrs._raj # ra in radians (0 - pi)
      decj = psrs._decj # dec in radians
      toaerrmed = np.median(psrs._toaerrs) # median TOA error in sec
      toaerrmean = np.mean(psrs._toaerrs) # mean TOA error in sec
      parfiles = parfiles[opts.num]
      timfiles = timfiles[opts.num]
      print('Current .par file: ',parfiles)
      print('Current .tim file: ',timfiles)
      directory += str(opts.num)+'_'+psrs.name+'/'
      exit_message = "This pulsar has already been processed"
      psrs = [psrs]
    return psrs



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
            

def init_pta(params,Tspan,psrs,pulpar,pultim,ind_par):
  """Initiate model and PTA for Enterprise.
  PTA for our code is just one pulsar, since we only
  do parameter estimation for pulsar noise"""
  models = list()
  from_par_file = list()
  ecorrexists = np.zeros(len(psrs))

  try:
    params.noisefiles
    noisedict = get_noise_dict(psrlist=[p.name for p in psrs],noisefiles=params.noisefiles)
  except:
    noisedict = None

  # FOR RED NOISE MODEL SELECTION ONLY!!!
  # We do this, so that white and DM noise parameters are consistent for models
  if ind_par!=0 and noisedict !=None:
    noisedict_dm = get_noise_dict(psrlist=[p.name for p in psrs],noisefiles=params.noisefiles)
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
    gwb = gp_signals.FourierBasisCommonGP(gwb_pl, orf, components=30, name='gwb', Tspan=Tspan)
    #m += gwb
    m = m + gwb if params.tm!='none' else gwb

  for psp in params.custom_commonpsr.split():
      psp_model = CustomModels(psrname=psr.name,params=params)
      m_sep += getattr(psp_model, psp)()

  if params.physephem=='none':
    print('Not fitting for Solar System ephemeris error')
  elif params.physephem=='default':
    eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)
    m += eph

  # Including parameters that can be separate for different pulsars
  for ii, psr in enumerate(psrs):
    # Add constant parameters to PTA from .par files
    if params.allpulsars == 'True':
      from_par_file.append( T2.tempopulsar(parfile=pulpar[ii],timfile=pultim[ii]) )
    elif params.allpulsars == 'False':
      from_par_file.append( T2.tempopulsar(parfile=pulpar,timfile=pultim) )
  
    # Determine if ecorr is mentioned in par file
    for key,val in from_par_file[ii].noisemodel.items():
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

    for psp in params.custom_singlepsr.split():
      psp_model = CustomModels(psrname=psr.name,params=params)
      m_sep += getattr(psp_model, psp)()

    if params.sn_model=='none':
      print('Red/spin noise model is switched off in a parameter file')
    elif params.sn_model=='default':
      log10_A = parameter.Uniform(params.sn_lgApr[0],params.sn_lgApr[1])
      gamma = parameter.Uniform(params.sn_gpr[0],params.sn_gpr[1])
      pl = utils.powerlaw(log10_A=log10_A, gamma=gamma, \
          components=params.sn_sincomp)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
      m_sep += rn
    elif params.sn_model=='default_informed':
      log10_A = parameter.Uniform(noisedict[psr.name+'_log10_A'][1],noisedict[psr.name+'_log10_A'][2])
      gamma = parameter.Uniform(noisedict[psr.name+'_gamma'][1],noisedict[psr.name+'_gamma'][2])
      pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
      m_sep += rn
    elif params.sn_model=='melatos':
      amp = parameter.Uniform(params.sn_amp[0],params.sn_amp[1])
      cc = parameter.Uniform(params.sn_cc[0],params.sn_cc[1])
      pl = powerlaw_melatos(amp=amp, cc=cc)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
      m_sep += rn
    elif params.sn_model=='powerlaw_v1':
      log10_A = parameter.Uniform(params.sn_lgApr[0],params.sn_lgApr[1])
      gamma = parameter.Uniform(params.sn_gpr[0],params.sn_gpr[1])
      if not np.isscalar(params.sn_fcpr):
        fc = parameter.Uniform(params.sn_fcpr[0],params.sn_fcpr[1])
      else:
        fc = parameter.Constant(params.sn_fcpr)
      pl = powerlaw_v1(log10_A=log10_A, gamma=gamma, fc=fc, \
          components=params.sn_sincomp)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan,\
        name='redv1')
      m_sep += rn
    elif params.sn_model=='powerlaw_v1_1':
      log10_A = parameter.Uniform(params.sn_lgApr[0],params.sn_lgApr[1])
      gamma = parameter.Uniform(params.sn_gpr[0],params.sn_gpr[1])
      if not np.isscalar(params.sn_fcpr):
        fc = parameter.Uniform(params.sn_fcpr[0],params.sn_fcpr[1])
      else:
        fc = parameter.Constant(params.sn_fcpr)
      pl = powerlaw_v1_1(log10_A=log10_A, gamma=gamma, fc=fc)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan,\
        name='redv1_1')
      m_sep += rn
    elif params.sn_model=='powerlaw_v2':
      log10_P0 = parameter.Uniform(params.sn_lgPpr[0],params.sn_lgPpr[1])
      alpha = parameter.Uniform(params.sn_apr[0],params.sn_apr[1])
      if not np.isscalar(params.sn_fcpr):
        fc = parameter.Uniform(params.sn_fcpr[0],params.sn_fcpr[1])
      else:
        fc = parameter.Constant(params.sn_fcpr)
      pl = powerlaw_v2(log10_P0=log10_P0, alpha=alpha, fc=fc)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
      m_sep += rn
    elif params.sn_model=='powerlaw_v2_informed':
      log10_P0 = parameter.Uniform(noisedict[psr.name+'_log10_P0'][1],noisedict[psr.name+'_log10_P0'][2])
      alpha = parameter.Uniform(noisedict[psr.name+'_alpha'][1],noisedict[psr.name+'_alpha'][2])
      fc = parameter.Uniform(noisedict[psr.name+'_fc'][1],noisedict[psr.name+'_fc'][2])
      pl = powerlaw_v2(log10_P0=log10_P0, alpha=alpha, fc=fc)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
      m_sep += rn
    elif params.sn_model=='powerlaw_v2_2':
      log10_P0 = parameter.Uniform(params.sn_lgPpr[0],params.sn_lgPpr[1])
      alpha = parameter.Uniform(params.sn_apr[0],params.sn_apr[1])
      fc = parameter.Uniform(params.sn_fcpr[0],params.sn_fcpr[1])
      beta = parameter.Uniform(params.sn_bpr[0],params.sn_bpr[1])
      pl = powerlaw_v2_2(log10_P0=log10_P0, alpha=alpha, fc=fc, beta=beta)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
      m_sep += rn
    elif params.sn_model=='powerlaw_v3':
      log10_A = parameter.Uniform(params.sn_lgApr[0],params.sn_lgApr[1])
      gamma = parameter.Uniform(params.sn_gpr[0],params.sn_gpr[1])
      pl = powerlaw_v3(log10_A=log10_A, gamma=gamma)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
      m_sep += rn
    elif params.sn_model=='powerlaw_v3_1':
      log10_A = parameter.Uniform(params.sn_lgApr[0],params.sn_lgApr[1])
      gamma = parameter.Uniform(params.sn_gpr[0],params.sn_gpr[1])
      fc = parameter.Uniform(params.sn_fcpr[0],params.sn_fcpr[1])
      pl = powerlaw_v3_1(log10_A=log10_A, gamma=gamma, fc=fc)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
      m_sep += rn
    elif params.sn_model=='free_spectrum':
      ncomp = 30
      log10_rho = parameter.Uniform(-10, -4, size=ncomp)
      pl = free_spectrum(log10_rho=log10_rho)
      rn = gp_signals.FourierBasisGP(spectrum=pl, components=ncomp, Tspan=Tspan)
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
            selection=rs_sel, Tspan=Tspan, name=params.rs_model[psr.name])
        m_sep += rs
 
    if params.dm_model=='none':
      print('DM noise model is switched off in a parameter file')
    elif params.dm_model=='default':
      dm_log10_A = parameter.Uniform(params.dm_lgApr[0],params.dm_lgApr[1])
      dm_gamma = parameter.Uniform(params.dm_gpr[0],params.dm_gpr[1])
      dm_basis= utils.createfourierdesignmatrix_dm(nmodes=30, Tspan=Tspan, fref=(4.15e3)**(0.5))
      dm_pl = utils.powerlaw(log10_A=dm_log10_A, gamma=dm_gamma)
      dm = gp_signals.BasisGP(dm_pl, dm_basis, name='dm_gp')
      m_sep += dm
    elif params.dm_model=='default_informed':
      dm_log10_A = parameter.Uniform(noisedict_dm[psr.name+'_dm_gp_log10_A'][1],noisedict_dm[psr.name+'_dm_gp_log10_A'][2])
      dm_gamma = parameter.Uniform(noisedict_dm[psr.name+'_dm_gp_gamma'][1],noisedict_dm[psr.name+'_dm_gp_gamma'][2])
      dm_basis= utils.createfourierdesignmatrix_dm(nmodes=30, Tspan=Tspan)
      dm_pl = utils.powerlaw(log10_A=dm_log10_A, gamma=dm_gamma)
      dm = gp_signals.BasisGP(dm_pl, dm_basis, name='dm_gp')
      m_sep += dm
    elif params.dm_model=='powerlaw_v2':
      log10_P0 = parameter.Uniform(params.dm_lgPpr[0],params.dm_lgPpr[1])
      alpha = parameter.Uniform(params.dm_apr[0],params.dm_apr[1])
      fc = parameter.Uniform(params.dm_fcpr[0],params.dm_fcpr[1])
      dm_basis= utils.createfourierdesignmatrix_dm(nmodes=30, Tspan=Tspan)
      dm_pl = powerlaw_v2(log10_P0=log10_P0, alpha=alpha, fc=fc)
      dm = gp_signals.BasisGP(dm_pl, dm_basis, name='dm_gp')
      m_sep += dm
    else:
      print('Warning: DM noise model in parameter file is not recognized!')

    models.append(m_sep(psr))

  #  models = [model(psr) for psr in psrs]

  #if params.allpulsars=='True':
  #  pta = signal_base.PTA(models)
  #elif params.allpulsars=='False':
  #  pta = signal_base.PTA(model(psrs[0]))
  #pta = signal_base.PTA([model(psr)])
  pta = signal_base.PTA(models)

  # Set some white noise parameters to constants, if required
  constpar = {}
  #constpar.update(get_noise_from_pal2(pulnoise))

  # Try-catch block for using noise files in PAL2 format, or .par file data
  # First, try-condition for loading constant params from PAL2 noise files
  if not noisedict==None:
    print('For constant parameters using noise files in PAL2 format')
    pta.set_default_params(noisedict_white)
    # Next, except condition for loading constant params from .par files, as no
    # direction to PAL2 noise files is provided
  else:
    print('Warning: constant parameters (if used) are taken from .par files')
    print('This only works for one pulsar analysis')
    if checkifconstpar(params) and params.allpulsars=='True':
      for jj, psr in enumerate(psrs):
        constpar=readconstpar(params.efacpr,from_par_file[jj].noisemodel,'efac',psr.name,constpar)
        constpar=readconstpar(params.equadpr,from_par_file[jj].noisemodel,'equad',psr.name,constpar)
        constpar=readconstpar(params.ecorrpr,from_par_file[jj].noisemodel,'ecorr',psr.name,constpar)        
    elif checkifconstpar(params) and params.allpulsars=='False':
        constpar=readconstpar(params.efacpr,from_par_file[jj].noisemodel,'efac',psrs.name,constpar)
        constpar=readconstpar(params.equadpr,from_par_file[jj].noisemodel,'equad',psrs.name,constpar)
        constpar=readconstpar(params.ecorrpr,from_par_file[jj].noisemodel,'ecorr',psrs.name,constpar)
  
    if not constpar=={}:
      pta.set_default_params(constpar)
      # Setting efacs per backend to one
      # Setting log_equads per backend to -5 (or equads to 0.00001)
      for sig in pta._signal_dict:
        if 'efac' in pta._signal_dict[sig].name.lower():
          for backend in pta._signal_dict[sig]._params:
            if pta._signal_dict[sig]._params[backend].value == None:
              print('[!!!] Setting ',backend,' to one. No parameter was provided for it in a parameter file.')
              pta._signal_dict[sig]._params[backend].value = 1
        if 'equad' in pta._signal_dict[sig].name.lower():
          for backend in pta._signal_dict[sig]._params:
            if 'log10' in backend.lower():
              if pta._signal_dict[sig]._params[backend].value == None:
                print('[!!!] Setting ',backend,' to -10. No parameter was provided for it in a parameter file.')
                pta._signal_dict[sig]._params[backend].value = -10
              elif pta._signal_dict[sig]._params[backend].value > 0:
                print('[!!!] Converting ',backend,' from microsec to sec, then to log10-space')
                pta._signal_dict[sig]._params[backend].value = np.log10(pta._signal_dict[sig]._params[backend].value*1e-6)
            else:
              if pta._signal_dict[sig]._params[backend].value == None:
                print('[!!!] Setting ',backend,' to 0. No parameter was provided for it in a parameter file.')
                pta._signal_dict[sig]._params[backend].value = 0
    # Condition for loading constant params from PAL2 noise files

  return pta

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

class Nestle_PT(object):
   def __init__(self,pta_params):
      self.pta_params = np.asarray(pta_params)
      # Prepare a transform for all params here, in advance !
      # Then vectorize
      self.unif = dict()
      self.unif['id'] = []
      self.unif['pmax'] = np.array([])
      self.unif['pmin'] = np.array([])
      self.norm = dict()
      self.norm['id'] = []
      self.norm['mu'] = np.array([])
      self.norm['sigma'] = np.array([])
      # Loop through parameters
      for ii, prm in enumerate(self.pta_params):
         if prm.size == None:
             size = 1
         else:
             size = prm.size
         # Loop through elements of parameters that have size
         for jj in range(0,size):
             if '_pmax' in dir(prm):
               self.unif['id'].append(ii+jj)
               self.unif['pmax'] = np.append(self.unif['pmax'], prm._pmax)
               self.unif['pmin'] = np.append(self.unif['pmin'], prm._pmin)
             elif '_mu' in dir(prm):
               self.norm['id'].append(ii+jj)
               self.norm['mu'] = np.append(self.norm['mu'], prm._mu)
               self.norm['sigma'] = np.append(self.norm['sigma'], prm._sigma)
             else:
               print('Error: one of priors is not recognized (Nestle_PT)')

   def nestle_uniform_prior_transform(self,input_data):
      out = np.empty(input_data.shape)

      out[self.unif['id']] = input_data[self.unif['id']] * (
            self.unif['pmax'] - self.unif['pmin']) + self.unif['pmin']
      out[self.norm['id']] = self.norm['mu'] + (
            self.norm['sigma'] * ndtri(input_data[self.norm['id']]) )
      return out

def load_to_dict(filename):
    ''' Load file to dictionary '''
    dictionary = dict()
    with open(filename) as ff:
        for line in ff:
            (key, val) = line.split()
            dictionary[key] = val
    return dictionary

def red_psd(ff,AA,gamma):
    norm = AA**2 * const.yr**3 / (12 * np.pi**2)
    return norm * (ff*const.yr)**(-gamma)

def red_v1_psd(ff,AA,gamma,fc):
    norm = AA**2 * const.yr**3 / (12 * np.pi**2)
    return norm * ((ff+fc)*const.yr)**(-gamma)

def dm_psd(ff,AA,gamma,max_rad_freq_hz,DMk=4.15e3):
    return red_psd(ff,AA,gamma)*DMk/max_rad_freq_hz**2

def lorenzian_red_psd(ff,PP,fc,alpha):
    return PP / (1 + (ff/fc)**2)**(alpha/2)

def plot_noise_psd_from_dict(psr, psd_params, backends, ff):
   for backend in backends:
       label = 'RMS white noise in '+backend
       wpsd = psd_params[backend]['rms_toaerr']*1e-6
       wpsd = np.repeat(wpsd,len(ff))
       plt.loglog(ff,wpsd,label=label)

   if 'red' in psd_params.keys():
       if 'A' in psd_params['red']:
           label = 'Red noise, lgA='+\
               str(round(np.log10(psd_params['red']['A']),2))+\
               ', gamma='+str(round(psd_params['red']['gamma'],2))
           plt.loglog(ff, red_psd(ff,psd_params['red']['A'],\
               psd_params['red']['gamma']),label=label)
       elif 'P' in psd_params['red']:
           label = 'Red noise, lgP='+\
               str(round(np.log10(psd_params['red']['P']),2))+\
               ', alpha='+str(round(psd_params['red']['alpha'],2))+\
               ', lg(fc)='+str(round(np.log10(psd_params['red']['fc']),2))
           plt.loglog(ff, lorenzian_red_psd(ff,psd_params['red']['P'],\
               psd_params['red']['fc'],psd_params['red']['alpha']),label=label)

   if 'dm' in psd_params.keys():
       if 'A' in psd_params['dm']:
           label = 'DM noise at max freq, lgA='+\
               str(round(np.log10(psd_params['dm']['A']),2))+\
               ', gamma='+str(round(psd_params['dm']['gamma'],2))
           #plt.loglog(ff, dm_psd(ff,psd_params['dm']['A'],\
           #    psd_params['dm']['gamma'],max(psr.freqs)*1e9),label=label)
           print('Function dm_psd in core.py: not clear how to include DM constant and radio frequencies')
           print('Not plotting DM noise PSD')

def add_noise(t2pulsar,noise_dict,sim_dm=True,sim_white=True,sim_red=True,seed=None):
   ''' Recognize noise from noise parameter name, and add to t2pulsar '''

   added_noise_psd_params = dict()
   flagid=list()
   if 'f' in t2pulsar.flags():
      backends = np.unique(t2pulsar.flagvals('f'))
      flagid.append('f')
   if 'g' in t2pulsar.flags(): #PPTA
      backends = np.append(backends, np.unique(t2pulsar.flagvals('g')) )
      flagid.append('g')
   if 'sys' in t2pulsar.flags() and not 'group' in t2pulsar.flags():
      backends = np.unique(t2pulsar.flagvals('sys'))
      flagid.append('sys')
   if 'sys' in t2pulsar.flags() and 'group' in t2pulsar.flags():
      backends = np.unique(t2pulsar.flagvals('group'))
      flagid.append('group')
   if flagid==[]:
      backends = []
      raise Exception('Backend convention is not recognized')

   used_backends=list()
   for noise_param, noise_val in noise_dict.items():
      if not np.isscalar(noise_val):
         noise_val = noise_val[0]
         noise_dict[noise_param] = noise_dict[noise_param][0]
      backend_name = ''
      param_name = ''
      for bcknd in backends:
         if bcknd in noise_param:
            used_backends.append(bcknd)
            backend_name = bcknd
            for fid in flagid: # for 2 flags in PPTA
              flagid_bcknd = ''
              if backend_name in t2pulsar.flagvals(fid):
                flagid_bcknd = fid
            added_noise_psd_params.setdefault(backend_name,dict())
            toaerr_bcknd = t2pulsar.toaerrs[ np.where(t2pulsar.flagvals(flagid_bcknd)==backend_name)[0] ]
            added_noise_psd_params[backend_name]['rms_toaerr'] = (np.sum(toaerr_bcknd**2)/len(toaerr_bcknd))**(0.5)
            added_noise_psd_params[backend_name]['mean_toaerr'] = np.mean(toaerr_bcknd)

      if 'efac' in noise_param.lower() and sim_white:
         param_name = 'efac'
         if not backend_name == '':
            #LT.add_efac(t2pulsar, efac=noise_val, seed=seed, flags=backend_name, flagid=flagid_bcknd)
            added_noise_psd_params[backend_name]['efac'] = noise_val
            print('Added efac: ',noise_val,backend_name)
         elif noise_param==t2pulsar.name+'_efac':
            #LT.add_efac(t2pulsar, efac=noise_val, seed=seed)
            added_noise_psd_params['efac'] = noise_val
            print('Added efac: ',noise_val)
         else:
            raise Exception('Efac is not recognized. Neither signle, nor per backend. Parameter name from noise file: ',noise_param,'. Backends: ',backends)
   
      elif 'log10_equad' in noise_param.lower() and sim_white:
         param_name = 'log10_equad'
         if not backend_name == '':
            #LT.add_equad(t2pulsar, equad=10**noise_val, seed=seed, flags=backend_name, flagid=flagid_bcknd)
            added_noise_psd_params[backend_name]['equad'] = 10**noise_val
            print('Added equad: ',10**noise_val,backend_name)
         elif noise_param==t2pulsar.name+'_log10_equad':
            #LT.add_equad(t2pulsar, equad=10**noise_val, seed=seed)
            added_noise_psd_params['equad'] = 10**noise_val
            print('Added efac: ',10**noise_val)
         else:
            raise Exception('Equad is not recognized. Neither signle, nor per backend. Parameter name from noise file: ',noise_param,'. Backends: ',backends)
   
      elif 'log10_ecorr' in noise_param.lower() and sim_white:
         param_name = 'log10_ecorr'
         if not backend_name == '':
            #LT.add_jitter(t2pulsar, ecorr=10**noise_val, seed=seed, flags=backend_name, flagid=flagid_bcknd)
            added_noise_psd_params[backend_name]['ecorr'] = 10**noise_val
            print('Added ecorr: ',10**noise_val,backend_name)
         elif noise_param==t2pulsar.name+'_log10_ecorr':
            #LT.add_jitter(t2pulsar, ecorr=10**noise_val, seed=seed)
            added_noise_psd_params['ecorr'] = 10**noise_val
            print('Added efac: ',noise_val)
         else:
            raise Exception('Ecorr is not recognized. Neither signle, nor per backend. Parameter name from noise file: ',noise_param,'. Backends: ',backends)
   
      elif 'dm_gp_log10_A' in noise_param and sim_dm:
         param_name = 'dm_gp_log10_A'
         #LT.add_dm(t2pulsar,A=10**noise_val,gamma=noise_dict[t2pulsar.name+'_dm_gp_gamma'],seed=seed,components=30)
         added_noise_psd_params.setdefault('dm',dict())
         added_noise_psd_params['dm']['A'] = 10**noise_val
         print('Added DM noise, A=',10**noise_val,' gamma=',noise_dict[t2pulsar.name+'_dm_gp_gamma'])
      elif 'dm_gp_gamma' in noise_param and sim_dm:
         added_noise_psd_params.setdefault('dm',dict())
         added_noise_psd_params['dm']['gamma'] = noise_val
         param_name = 'dm_gp_gamma'
         pass

      elif noise_param==t2pulsar.name+'_log10_A' and sim_red:
         param_name = 'log10_A'
         #LT.add_rednoise(t2pulsar,A=10**noise_val,gamma=noise_dict[t2pulsar.name+'_gamma'],seed=seed,components=30)
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['A'] = 10**noise_val
         print('Added red noise, A=',10**noise_val,' gamma=',noise_dict[t2pulsar.name+'_gamma'])
      elif noise_param==t2pulsar.name+'_gamma':
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['gamma'] = noise_val
         param_name = 'gamma'
         pass

      elif 'log10_P0' in noise_param:
         param_name = 'log10_P0'
         #LT_custom.add_lorenzianrednoise(t2pulsar,P=10**noise_val,fc=10**noise_dict[t2pulsar.name+'_fc'],alpha=noise_dict[t2pulsar.name+'_alpha'],seed=seed)
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['P'] = 10**noise_val
         print('Added red noise, P=',10**noise_val,' fc=',noise_dict[t2pulsar.name+'_fc'], ' alpha=',noise_dict[t2pulsar.name+'_alpha'])
      elif 'alpha' in noise_param:
         param_name = 'alpha'
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['alpha'] = noise_val
         pass
      elif 'fc' in noise_param:
         param_name = 'fc'
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['fc'] = 10**noise_val
         pass

      else:
         print('Warning: parameter ',noise_param,' is not recognized or switched off. It was not used to simulate any data.')

   if sim_white:
      vector_vals, vector_bckd = added_noise_psd_to_vector( \
              added_noise_psd_params,param='efac')
      LT.add_efac(t2pulsar, efac=vector_vals, seed=seed, flags=vector_bckd, flagid=flagid_bcknd)
      vector_vals, vector_bckd = added_noise_psd_to_vector( \
              added_noise_psd_params,param='equad')
      LT.add_equad(t2pulsar, equad=vector_vals, seed=seed, flags=vector_bckd, flagid=flagid_bcknd)

   if sim_red:
      Ared = added_noise_psd_params['red']['A']
      gred = added_noise_psd_params['red']['gamma']
      LT.add_rednoise(t2pulsar,A=Ared,gamma=gred,seed=seed,components=30)

   if sim_dm:
      Adm = added_noise_psd_params['dm']['A']
      gdm = added_noise_psd_params['dm']['gamma']
      LT.add_dm(t2pulsar,A=Adm,gamma=gdm,seed=seed,components=30)

   used_backends = np.unique(used_backends)
   backends = np.unique(backends)

   if len(backends)!=len(used_backends):
      print('[!] Warning, number of not used backends: ',len(backends)-len(used_backends))

   return t2pulsar, used_backends, added_noise_psd_params

def added_noise_psd_to_vector(added_noise_psd_params,param='efac'):
    ''' Conversion of dict to vector to simulate efac/equad in libstempo'''
    vector_vals = list()
    vector_bckd = list()
    for key, val in added_noise_psd_params.items():
        if param in val:
            vector_vals.append( val[param] )
            vector_bckd.append( key )
    return vector_vals, vector_bckd
