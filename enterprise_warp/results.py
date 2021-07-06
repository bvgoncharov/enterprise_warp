import matplotlib
matplotlib.use('Agg')
#from matplotlib import rcParams
#rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'
import matplotlib.pyplot as plt

import os
import re
import json
import shutil
import pickle
import optparse
import itertools
import numpy as np
import scipy as sp
import pandas as pd
from corner import corner
from datetime import datetime
from bilby import result as br
from chainconsumer import ChainConsumer
from dateutil.parser import parse as pdate

from enterprise_extensions.frequentist.optimal_statistic import OptimalStatistic as OptStat
from enterprise.signals import signal_base

from . import enterprise_warp

def parse_commandline():
  """
  Parsing command line arguments for action on results
  """

  parser = optparse.OptionParser()

  parser.add_option("-r", "--result", help="Output directory or a parameter \
                    file. In case of individual pulsar analysis, specify a \
                    directory that contains subdirectories with individual \
                    pulsar results. In case of an array analysis, specify a \
                    directory with result files.", \
                    default=None, type=str)

  parser.add_option("-i", "--info", help="Print information about all results. \
                    In case \"-n\" is specified, print an information about \
                    results for a specific pulsar.", \
                    default=0, type=int)

  parser.add_option("-n", "--name", help="Pulsar name or number (or \"all\")", \
                    default="all", type=str)

  parser.add_option("-c", "--corner", help="Plot corner (0 - no corner, 1 - \
                    corner, 2 - chainconsumer), ", default=0, type=int)

  parser.add_option("-p", "--par", help="Include only model parameters that \
                    contain \"par\" (more than one could be added)",
                    action="append", default=None, type=str)

  parser.add_option("-a", "--chains", help="Plot chains (1/0)", \
                    default=0, type=int)

  parser.add_option("-b", "--logbf", help="Display log Bayes factors (1/0)", \
                    default=0, type=int)

  parser.add_option("-f", "--noisefiles", help="Make noisefiles (1/0)", \
                    default=0, type=int)

  parser.add_option("-l", "--credlevels", help="Credible levels (1/0)", \
                    default=0, type=int)

  parser.add_option("-m", "--covm", help="Collect PTMCMCSampler covariance \
                    matrices (1/0)", default=0, type=int)

  parser.add_option("-u", "--separate_earliest", help="Separate the first MCMC \
                    samples (fraction). Optional: add --par to also separate \
                    the chain with only --par columns.", default=0., type=float)

  parser.add_option("-s", "--load_separated", help="Attempt to load separated \
                    chain files with names chain_DATETIME(14-symb)_PARS.txt. \
                    If --par are supplied, load only files with --par \
                    columns.", default=0, type=int)


  parser.add_option("-o", "--optimal_statistic", help="Calculate optimal \
                    statistic and make key plots (1/0)", default = 0,
                    type = int)

  parser.add_option("-g", "--optimal_statistic_orf", help = "Set overlap reduction \
                    function form for optimal statistic analysis. Allowed \
                    options: all, hd (Hellings-Downs), quadrupole, dipole, monopole",
                    default = "hd,dipole,monopole", type = str)

  parser.add_option("-s", "--optimal_statistic_nsamples", help = "Set integer number \
                    of samples for noise-marginalised optimal statistic analysis.",
                    default = 1000, type = int)

  parser.add_option("-y", "--bilby", help="Load bilby result", \
                    default=0, type=int)


  opts, args = parser.parse_args()

  return opts

def get_HD_curve(zeta):

  coszeta = np.cos(zeta*np.pi/180.)
  xip = (1.-coszeta) / 2.
  HD = 3.*( 1./3. + xip * ( np.log(xip) -1./6.) )

  return HD/2.0

def get_dipole_curve(zeta):
  coszeta = np.cos(zeta*np.pi/180.)
  return coszeta


def get_monopole_curve(zeta):
  return zeta * 0.0

def dist_mode_position(values, nbins=50):
  """
  Parameters
  ----------
  values: float
    Values of a distribution
  method: int
    Approximating a distribution with a histogram with this number of bins

  Returns
  -------
  value : float
    Position of the largest frequency bin
  """
  nb, bins, patches = plt.hist(values, bins=nbins)
  plt.close()
  return bins[np.argmax(nb)]

def suitable_estimator(levels, errorbars_cdf = [16,84]):
  """
  Returns maximum-posterior value (posterior mode position) if it is within
  credible levels, otherwise returns 50%-CDF value.
  The function complements estimate_from_distribution().
  """
  if levels['maximum'] < levels[str(errorbars_cdf[1])] and \
     levels['maximum'] > levels[str(errorbars_cdf[0])]:
    return levels['maximum'], 'maximum'
  else:
    return levels['50'], '50'

def estimate_from_distribution(values, method='mode', errorbars_cdf = [16,84]):
  """
  Return estimate of a value from a distribution (i.e., an MCMC posterior)

  Parameters
  ----------
  values: float
    Values of a distribution
  method: str
    Currently available: mode or median

  Returns
  -------
  value : float
    Position of a mode or a median of a distribution, along the "values" axis
  """
  if method == 'median':
    return np.median(values)
  elif method == 'mode':
    return dist_mode_position(values)
  elif method == 'credlvl':
    levels = dict()
    levels['median'] = np.median(values)
    levels['maximum'] = dist_mode_position(values)
    levels[str(errorbars_cdf[0])] = \
           np.percentile(values, errorbars_cdf[0], axis=0)
    levels[str(errorbars_cdf[1])] = \
           np.percentile(values, errorbars_cdf[1], axis=0)
    levels[str(50)] = np.percentile(values, 50, axis=0)
    return levels

def make_noise_dict(psrname, chain, pars, method='mode', suffix = 'noise', \
  recompute = True):
  """
  Create noise dictionary for a given MCMC or nested sampling chain.
  This is a dict that assigns a characteristic value (mode/median)
  to a parameter from the distribution of parameter values in a chain.
  Can be used for outputting a noise file or for use in further
  analysis (e.g. optimal statistic)
  """

  result_filename = outdir + '/' + psrname + '_' + suffix + '.json'

  if not recompute:
    if os.path.exists(result_filename, 'r'):
      xx = json.load(open(result_fileneame))
      return(xx)

  xx = {}
  for ct, par in enumerate(pars):
    xx[par] = estimate_from_distribution(chain[:,ct], method=method)
  return xx

def make_noise_files(psrname, chain, pars, outdir='noisefiles/',
                     method='mode', suffix='noise'):
  """
  Create noise files from a given MCMC or nested sampling chain.
  Noise file is a dict that assigns a characteristic value (mode/median)
  to a parameter from the distribution of parameter values in a chain.
  """

  xx = make_noise_dict(psrname, chain, pars, method = method)

  os.system('mkdir -p {}'.format(outdir))
  with open(outdir + '/' + psrname + '_' + suffix + '.json', 'w') as fout:
    json.dump(xx, fout, sort_keys=True, indent=4, separators=(',', ': '))


def check_if_psr_dir(folder_name):
  """
  Check if the folder name (without path) is in the enterprise_warp format:
  integer, underscore, pulsar name.
  """
  return bool(re.match(r'^\d{1,}_[J,B]\d{2,4}[+,-]\d{4,4}[A,B]{0,1}$',
  folder_name))



class OptimalStatisticResult(OptStat):

  def __init__(self, OptimalStatistic, params, xi, rho, sig, OS, OS_err):
    self.OptimalStatistic = OptimalStatistic #OptimalStatistic object
    self.params = params #optimal statistic parameters
    self.xi = xi
    self.rho = rho
    self.sig = sig
    self.OS = OS #the actual A^2 optimal statistic
    self.OS_err = OS_err #optimal statistic error

  def add_marginalised(self, marginalised_os, marginalised_os_err):
    self.marginalised_os = marginalised_os
    self.marginalised_os_err = marginalised_os_err

  def weightedavg(self, _rho, _sig):

    return np.average(_rho, _sig**-2.0), np.sqrt(np.sum(_sig**-2.0))

  def bin_crosscorr(self, zeta):

    idx = np.argsort(self.xi)
    xi_sorted = self.xi[idx]
    rho_sorted = self.rho[idx]
    sig_sorted = self.sig[idx]
    rho_avg, sig_avg = np.zeros(len(zeta)), np.zeros(len(zeta))


    for i,z in enumerate(zeta[:-1]):
      _rhos, _sigs = [], []
      for x,r,s in zip(xi,rho,sig):
        if x >= z and x < (z+10.):
          _rhos.append(r)
          _sigs.append(s)
          rho_avg[i], sig_avg[i] = self.weightedavg(_rhos, _sigs)

    return rho_avg, sig_avg

  def avg_ostat_bins(self):
    # sort the cross-correlations by xi
    idx = np.argsort(self.xi)

    xi_sorted = self.xi[idx]
    rho_sorted = self.rho[idx]
    sig_sorted = self.sig[idx]

    # bin the cross-correlations so that there are the same number of \
    #pairs per bin
    n_psr = len(self.psrs)
    npairs = int(n_psr*(n_psr - 1.0)/2.0)

    xi_avg = []
    xi_err = []

    rho_avg = []
    sig_avg = []

    i = 0
    while i < len(xi_sorted):

      xi_avg.append(np.mean(xi_sorted[i:npairs+i]))
      xi_err.append(np.std(xi_sorted[i:npairs+i]))

      r, s = self.weightedavg(rho_sorted[i:npairs+i], \
        sig_sorted[i:npairs+i])
      rho_avg.append(r)
      sig_avg.append(s)

      i += npairs

    xi_avg = np.array(xi_avg)
    xi_err = np.array(xi_err)
    rho_avg = np.array(rho_avg)
    sig_avg = np.array(sig_avg)

    #do we want to return these or add them as class attributes?
    self.xi_avg = xi_avg
    self.xi_err = xi_err
    self.rho_mean = rho_avg
    self.sig_avg = sig_avg
    #return xi_mean, xi_err, rho_avg, sig_avg


class EnterpriseWarpResult(object):

  def __init__(self, opts):
    self.opts = opts
    self.interpret_opts_result()
    self.get_psr_dirs()


  def main_pipeline(self):
    self._reset_covm()
    for psr_dir in self.psr_dirs:

      self.psr_dir = psr_dir
      success = self._scan_psr_output()
      if not success:
        continue

      self._get_covm()

      if not (self.opts.noisefiles or self.opts.logbf or self.opts.corner or \
              self.opts.chains):
        continue

      success = self.load_chains()
      if not success:
        continue

      self._separate_earliest()
      self._make_noisefiles()
      self._get_credible_levels()
      self._print_logbf()
      self._make_corner_plot()
      self._make_chain_plot()

    self._save_covm()

  def _scan_psr_output(self):

    self.outdir = self.outdir_all + '/' + self.psr_dir + '/'
    if self.opts.name is not 'all' and self.opts.name not in self.psr_dir:
      return False
    print('Processing ', self.psr_dir)

    self.get_pars()
    self.get_chain_file_name()

    return True

  def interpret_opts_result(self):
    """ Determine output directory from the --results argument """
    if os.path.isdir(self.opts.result):
      self.outdir_all = self.opts.result
    elif os.path.isfile(self.opts.result):
      self.params = enterprise_warp.Params(self.opts.result, init_pulsars=False)
      self.outdir_all = self.params.out + self.params.label_models + '_' + \
                        self.params.paramfile_label + '/'
    else:
      raise ValueError('--result seems to be neither a file, not a directory')


  def get_psr_dirs(self):
    """ Check if we need to loop over pulsar directories, or not """
    out_subdirs = np.array(os.listdir(self.outdir_all))
    psr_dir_mask = [check_if_psr_dir(dd) for dd in out_subdirs]
    self.psr_dirs = out_subdirs[psr_dir_mask]
    if self.psr_dirs.size == 0:
      self.psr_dirs = np.array([''])


  def get_chain_file_name(self):
    if self.opts.load_separated:
      outdirfiles = next(os.walk(self.outdir))[2]
      self.chain_file = list()
      for ff in outdirfiles:
        if len(ff.split('_')) < 2: continue
        timestr = ff.split('_')[1]
        if self.par_out_label=='' and timestr[-4:]=='.txt':
          timestr = timestr[:-4]
        elif self.par_out_label!='':
          pass
        else:
          continue
        if not (timestr.isdigit() and len(timestr)==14):
          continue
        #if self.par_out_label=='':
        #  if ff.split('_')[2]!=self.par_out_label:
        #    continue
        self.chain_file.append(self.outdir + ff)
      if not self.chain_file:
        self.chain_file = None
        print('Could not find chain file in ',self.outdir)

    else:
      if os.path.isfile(self.outdir + '/chain_1.0.txt'):
        self.chain_file = self.outdir + '/chain_1.0.txt'
      elif os.path.isfile(self.outdir + '/chain_1.txt'):
        self.chain_file = self.outdir + '/chain_1.txt'
      else:
        self.chain_file = None
        print('Could not find chain file in ',self.outdir)

    if self.opts.info and self.chain_file is not None:
      print('Available chain file ', self.chain_file, '(',
            int(np.round(os.path.getsize(self.chain_file)/1e6)), ' Mb)')


  def get_pars(self):
    self.par_out_label = '' if self.opts.par is None \
                            else '_'.join(self.opts.par)
    if not os.path.exists(self.outdir + '/pars_' + self.par_out_label + '.txt'):
      self.par_out_label = ''
    if self.opts.load_separated and self.par_out_label!='':
      self.pars = np.loadtxt(self.outdir + '/pars_' + self.par_out_label + \
                             '.txt', dtype=np.unicode_)
    else:
      self.pars = np.loadtxt(self.outdir + '/pars.txt', dtype=np.unicode_)
    self._get_par_mask()
    if self.opts.info and (self.opts.name != 'all' or self.psr_dir == ''):
      print('Parameter names:')
      for par in self.pars:
        print(par)


  def load_chains(self):
    """ Loading PTMCMC chains """
    if self.opts.load_separated:
      self.chain = np.empty((0,len(self.pars)))
      for ii, cf in enumerate(self.chain_file):
        if ii==0:
          self.chain = np.loadtxt(cf)
        else:
          self.chain = np.concatenate([self.chain, np.loadtxt(cf)])
    else:
      try:
        self.chain = np.loadtxt(self.chain_file)
      except:
        print('Could not load file ', self.chain_file)
        return False
      if len(self.chain)==0:
        print('Empty chain file in ', self.outdir)
        return False
    burn = int(0.25*self.chain.shape[0])
    self.chain_burn = self.chain[burn:,:-4]

    if 'nmodel' in self.pars:
      self.ind_model = list(self.pars).index('nmodel')
      self.unique, self.counts = np.unique(np.round( \
                                 self.chain_burn[:, self.ind_model]), \
                                 return_counts=True)
      self.dict_real_counts = dict(zip(self.unique.astype(int),
                                       self.counts.astype(float)))
    else:
      self.ind_model = 0
      self.unique, self.counts, self.dict_real_counts = [None], None, None

    return True


  def _get_par_mask(self):
    """ Get an array mask to select only parameters chosen with --par """
    if self.opts.par is not None:
      masks = list()
      for pp in self.opts.par:
        masks.append( [True if pp in label else False for label in self.pars] )
      self.par_mask = np.sum(masks, dtype=bool, axis=0)
    else:
      self.par_mask = np.repeat(True, len(self.pars))

  def _make_noisefiles(self):
    if self.opts.noisefiles:
      make_noise_files(self.psr_dir, self.chain_burn, self.pars,
                       outdir = self.outdir_all + '/noisefiles/')

  def _get_credible_levels(self):
    if self.opts.credlevels:
      make_noise_files(self.psr_dir, self.chain_burn, self.pars,
                       outdir = self.outdir_all + '/noisefiles/',
                       suffix = 'credlvl', method='credlvl')

  def _reset_covm(self):
    self.covm = np.empty((0,0))
    self.covm_pars = np.array([])
    self.covm_repeating_pars = np.array([])

  def _save_covm(self):
    if self.opts.covm:
      out_dict = {
                 'covm': self.covm,
                 'covm_pars': self.covm_pars,
                 'covm_repeating_pars': self.covm_repeating_pars,
                 }
      with open(self.outdir_all+'covm_all.pkl', 'wb') as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

      df = pd.DataFrame(self.covm, index=self.covm_pars, columns=self.covm_pars)
      df.to_csv(self.outdir_all+'covm_all.csv')

  def _get_covm(self):
    """
    Save available PTMCMCSampler covariance matrices in Pandas format,
    exclude repeating parameters.
    """
    if self.opts.covm:
      covm_pars_add = self.pars
      covm_add = np.load(self.outdir_all + self.psr_dir + '/cov.npy')
      common_pars = set(self.covm_pars) & set(self.pars)
      for cp in common_pars:
        if cp not in self.covm_repeating_pars:
          self.covm_repeating_pars = np.append(self.covm_repeating_pars, cp)
      for rp in self.covm_repeating_pars:
        if rp in covm_pars_add:
          mask_delete = covm_pars_add==rp
          covm_pars_add = covm_pars_add[~mask_delete]
          covm_add = covm_add[~mask_delete,:][:,~mask_delete]
        if rp in self.covm_pars:
          mask_delete = self.covm_pars==rp
          self.covm_pars = self.covm_pars[~mask_delete]
          self.covm = self.covm[~mask_delete,:][:,~mask_delete]
      self.covm_pars = np.concatenate([self.covm_pars, covm_pars_add])
      self.covm = sp.linalg.block_diag(self.covm, covm_add)

  def _separate_earliest(self):
    if self.opts.separate_earliest:
      chain_shape = self.chain.shape
      earliest_nlines = int(np.round(chain_shape[0] * \
                                     self.opts.separate_earliest))
      earliest_chain = self.chain[0:earliest_nlines,:]
      time_now = datetime.now().strftime("%Y%m%d%H%M%S")
      earliest_name = 'chain_' + time_now + '.txt'
      np.savetxt(self.outdir + earliest_name, earliest_chain)

      if self.opts.par is not None:
        earliest_name_par = 'chain_' + time_now + '_' + self.par_out_label + \
                            '.txt'
        mask_full_chain = np.append(self.par_mask, [True,True,True,True])
        earliest_chain_par = earliest_chain[:,mask_full_chain]
        np.savetxt(self.outdir + earliest_name_par, earliest_chain_par)
        np.savetxt(self.outdir + 'pars_' + self.par_out_label + '.txt', \
                   self.pars[self.par_mask], fmt="%s")

      shutil.copyfile(self.chain_file, self.chain_file+'.bckp')
      np.savetxt(self.chain_file, self.chain[earliest_nlines:,:])

      print('Earlier chain fraction (', self.opts.separate_earliest*100, \
            ' %) is separated, exiting.')
      exit()

  def _print_logbf(self):
    """ Print log Bayes factors (product-space) from PTMCMC on the screen """
    if self.opts.logbf:
      print('=====', self.psr_dir, ' model selection results', '=====')
      print('Samples in favor of models: ', self.dict_real_counts)
      if len(self.unique) > 1:
        count_by_pairs = list(itertools.combinations(sorted(self.unique), 2))
        for combination in count_by_pairs:
          logbf = np.log(self.dict_real_counts[combination[1]] / \
                         self.dict_real_counts[combination[0]])
          print('logBF for ', int(combination[1]), 'over ', \
                int(combination[0]),': ', logbf)


  def _make_corner_plot(self):
    """ Corner plot for a posterior distribution from the result """
    if self.opts.corner == 1:
      for jj in self.unique:
        if jj is not None:
          model_mask = np.round(self.chain_burn[:,self.ind_model]) == jj
        else:
          model_mask = np.repeat(True, self.chain_burn.shape[0])
        chain_plot = self.chain_burn[model_mask,:]
        chain_plot = chain_plot[:,self.par_mask]
        figure = corner(chain_plot, 30, labels=self.pars[self.par_mask])
        plt.savefig(self.outdir_all + '/' + self.psr_dir + '_corner_' + \
                    str(jj) + '_' + self.par_out_label + '.png')
        plt.close()
    elif self.opts.corner == 2:
      cobj = ChainConsumer()
      pars = self.pars.astype(str)
      pars = np.array(['$'+pp+'$' for pp in pars],dtype=str)
      for jj in self.unique:
        if jj is not None:
          model_mask = np.round(self.chain_burn[:,self.ind_model]) == jj
        else:
          model_mask = np.repeat(True, self.chain_burn.shape[0])
        chain_plot = self.chain_burn[model_mask,:]
        chain_plot = chain_plot[:,self.par_mask]
        cobj.add_chain(chain_plot, name=str(jj),
                       parameters=pars[self.par_mask].tolist())
      cobj.configure(serif=True, label_font_size=12, tick_font_size=12,
                     legend_color_text=False, legend_artists=True)
      corner_name = self.outdir_all + '/' + self.psr_dir + '_' + \
                    self.par_out_label + '_corner.png'
      fig = cobj.plotter.plot(filename=corner_name)
      plt.close()

  def _make_chain_plot(self):
    """ MCMC chain plots (evolution in time) """
    if self.opts.chains:
       thin_factor = 200
       x_tiles = np.floor(len(self.pars)**0.5)
       y_tiles = np.ceil(len(self.pars)/x_tiles)
       plt.figure(figsize=[6.4*x_tiles,4.8*y_tiles])
       for pp, par in enumerate(self.pars):
          plt.subplot(x_tiles, y_tiles, pp + 1)
          cut_chain = self.chain[::int(self.chain[:,pp].size/thin_factor),pp]
          plt.plot(cut_chain,label=par.replace('_','\n'))
          plt.legend()
          plt.xlabel('Thinned MCMC iterations')
          plt.ylabel('Value')
       plt.subplots_adjust(wspace=0.)
       plt.tight_layout()
       plt.savefig(self.outdir_all + '/' + self.psr_dir + '_samples_trace_' + \
                   '.png')
       plt.close()

class EnterpriseWarpOptimalStatistic(EnterpriseWarpResult):
  def __init__(self, opts):
    super(EnterpriseWarpOptimalStatistic).__init__(opts)
    self.interpret_opts_result()
    self.optstat_orfs = list(self.opts.optimal_statistic_orfs.split(','))
    self.optstat_nsamp = self.opts.optimal_statistic_nsamples
    self._get_pta()

  def main_pipeline(self):

    for psr_dir in self.psr_dirs:

      self.psr_dir = psr_dir
      success = self._scan_psr_output()
      if not success:
        continue

      success = self.load_chains()
      if not success:
        continue

      self._get_pta()
      self._compute_optimalstatistic(method = 'mode')
      self._marginalise_ostat()
      self._avg_ostat_bins()
      self.plot_noisemarg_os()
      self.plot_os_orf()

      return True

  def load_chains(self):
    """ Loading PTMCMC chains """
    if self.opts.load_separated:
      self.chain = np.empty((0,len(self.pars)))
      for ii, cf in enumerate(self.chain_file):
        if ii==0:
          self.chain = np.loadtxt(cf)
        else:
          self.chain = np.concatenate([self.chain, np.loadtxt(cf)])
    else:
      try:
        self.chain = np.loadtxt(self.chain_file)
      except:
        print('Could not load file ', self.chain_file)
        return False
      if len(self.chain)==0:
        print('Empty chain file in ', self.outdir)
        return False
    burn = int(0.25*self.chain.shape[0])
    self.chain_burn = self.chain[burn:,:-4]

    if 'nmodel' in self.pars:
      self.ind_model = list(self.pars).index('nmodel')
      self.unique, self.counts = np.unique(np.round( \
                                 self.chain_burn[:, self.ind_model]), \
                                 return_counts=True)
      self.dict_real_counts = dict(zip(self.unique.astype(int),
                                       self.counts.astype(float)))
    else:
      self.ind_model = 0
      self.unique, self.counts, self.dict_real_counts = [None], None, None

    if 'gw_log10_A' not in self.pars:
      raise AttributeError('Uncorrelated common noise amplitude must be\
                            in parameter list!')
    self.ind_gw_log10_A = list(self.pars).index('gw_log10_A')
    self.gw_log10_A = self.chain_burn[:, self.ind_gw_log10_A]

    return True

  def interpret_opts_result(self):
    if os.path.isdir(self.opts.result):
      raise ValueError("--result should be a parameter file for \
                        optimal statistic")
    elif os.path.isfile(self.opts.result):
      self.params = enterprise_warp.Params(self.opts.result, \
                                           init_pulsars=True)
      #might want to include custom models support here
      self.outdir_all = params.out + params.label_models + '_' \
                                   + params.paramfile_label + '/'

  def _get_pta(self):
    self.pta = enterprise_warp.init_pta(self.params)

  def _add_optimalstatistics(self, method='mode', chain_idx = 0):
    optstat_dict = {}

    if method == 'samp':
      #make noise dict from chain sample
      os_params = dict(zip(self.pars, self.chain_burn[chain_idx]))
    elif method == 'mode' or method == 'median':
      #make noise dict from max post or max likelihood / whatever method
      os_params = make_noise_dict(self.psr_dir,self.chain_burn,self.pars,\
                                  method = method, recompute = False)

    for orf in self.optstat_orfs:
      _os = OptStat.OptimalStatistic(psrs, pta = pta, orf = orf)
      _xi, _rho, _sig, _OS, _OS_sig = _os.compute_os(params=os_params)

      result = OptimalStatisticResult(_os, os_params, _xi, _rho, _sig, _OS, \
                                      _OS_sig)
      optstat_dict['orf'] = result

      if method == 'samp':
        return optstat_dict #this is probably useless
      else:
        self.OptimalStatisticResults = optstat_dict #this is a representative optimal statistic
        return True


  def _marginalise_ostat(self):

    for orf in self.optstat_orfs:

      _osr = self.OptimalStatisticResults[orf]
      _os = _osr.OptimalStatistic

      _noisemarg_os = np.zeros(self.optstat_nsamp)
      _noisemarg_os_err = np.zeros(self.optstat_nsamp)

      samp_indices = np.random.randint(0, \
                                       self.chain_burn.shape[0], \
                                       size = self.optstat_nsamp \
                                      )
      for ii in range(self.optstat_nsamp):
          chain_idx = samp_indices[ii]
          os_params = dict(zip(self.pars, self.chain_burn[chain_idx]))
          _xi, _rho, _sig, _OS, _OS_sig = _os.compute_os(params=os_params)
          _ostat_dict = self._compute_optimalstatistic(method = 'samp', \
                                                       chain_idx = \
                                                       samp_indices[ii] \
                                                      )
          _noisemarg_os[ii] = _OS
          _noisemarg_os_err[ii] = _OS_sig

      _osr.add_marginalised(_noisemarg_os, _noisemarg_os_err)

  def _avg_ostat_bins(self):
    for orf, _osr in self.OptimalStatisticResults.items():
      _osr.avg_ostat_bins()

  def plot_os_orf(self):

    orf_funcs = {'hd': get_HD_curve, \
                 'dipole': get_dipole_curve, \
                 'monopole': get_monopole_curve
                }


    color_dict = {'hd': '#E53935', \
                  'dipole': '#39E635', \
                  'monopole': '#3539E6'
                 }

    default_linewidth = 0.8
    highlight_linewidth = 1.8

    for _orf, _osr in self.OptimalStatisticResults.items():
      _xi_avg = _osr.xi_avg
      _rho_avg = _osr.rho_avg
      _xi_err = _osr.xi_err
      _sig_avg = _osr.sig_avg
      _OS = _osr.OS
      fig, ax = plt.subplots(1, 1, figsize = (2.008, 3.25))
      (_, caps, _) = ax.errorbar(_xi_avg,\
                                 _rho_avg,\
                                 xerr = _xi_err,\
                                 yerr = _sig_avg,\
                                 marker = 'o',\
                                 ls = '',\
                                 color = '#4FC3F7',\
                                 fmt = 'o',\
                                 capsize = 4,\
                                 elinewidth = 1.2\
                                )


      zeta = np.linspace(0.001, np.pi, 200)

      for __orf, curve in orf_funcs.items():
        orf_curve = curve(zeta + 1.0)
        if __orf == _orf:
          linewidth = highlight_linewidth
        else:
          linewidth = default_linewidth


        ax.plot(zeta, _OS*orf_curve, \
                linestyle = '--', \
                color = color_dict['orf'], \
                linewidth = linewidth\
                )

      ax.set_xlim(0, np.pi)
      ax.set_ylim(-4E-30, 4E-30)
      ax.set_xlabel(r'$\zeta$ (rad)')
      ax.set_ylabel(r'$\hat{{A}}^2 \Gamma_{{ab}}(\zeta)$')
      fig.tight_layout()
      fig.savefig(
                  self.outdir_all + '/' + self.psr_dir + '_os_orf_' +  _orf + '_' + \
                  self.par_out_label + '.png', \
                  dpi = 300, \
                  bbox_inches = 'tight' \
                  )
      fig.close()

  def plot_noisemarg_os(self):
    #plot OS S/N
    color_dict = {'hd': '#E53935', \
                  'dipole': '#39E635', \
                  'monopole': '#3539E6'\
                 }

    fig1, ax1 = plt.subplots(1, 1, figsize = (2.008, 3.25))
    fig2, ax2 = plt.subplots(1, 1, figsize = (2.008, 3.25))
    for orf, _osr in self.OptimalStatisticResults.items():
      _noisemarg_os = _osr.marginalised_os
      _noisemarg_os_err = _osr.marginalised_os_err
      _color = color_dict[orf]
      ax1.hist(_noisemarg_os/_noisemarg_os_err, \
               histtype = 'step', \
               color = _color, \
               label = orf \
               )

      ax1.axvline(np.mean(_noisemarg_os/_noisemarg_os_er), \
                         linestyle = '--', color = _color)

      ax2.hist(_noisemarg_os, histtype = 'step', color = _color, label = orf)
      ax2.axvline(np.mean(_noisemarg_os), linestyle = '--', color = _color)

    ax1.legend(fontsize = 11)
    ax1.set_xlabel('SNR')
    ax1.set_ylabel('Counts')
    fig1.savefig(self.outdir_all + '/' + self.psr_dir + '_os_SNR_' +  '_' +\
                 self.par_out_label + '.png', dpi = 300, bbox_inches = 'tight')
    fig1.close()

    ax2.hist(self.gw_log10_A**2.0, histtype = 'step', color = '0.5', label = 'uncorrelated')
    ax2.axvline(np.mean(self.gw_log10_A**2.0), linestyle = '--', color = '0.5')
    ax2.legend(fontsize = 11)
    ax2.set_xlabel('$\hat{{A}}^{{2}}$')
    ax2.set_ylabel('Counts')
    fig2.savefig(self.outdir_all + '/' + self.psr_dir + '_os_A2_' + \
                 '_' + self.par_out_label + '.png', dpi = 300, \
                 bbox_inches = 'tight')
    fig2.close()


class BilbyWarpResult(EnterpriseWarpResult):

  def __init__(self, opts):
    super(BilbyWarpResult, self).__init__(opts)

  def get_chain_file_name(self):
      label = os.path.basename(os.path.normpath(self.outdir))
      if os.path.isfile(self.outdir + '/' + label + '_result.json'):
        self.chain_file = self.outdir + '/' + label + '_result.json'
      else:
        self.chain_file = None
        print('Could not find chain file in ',self.outdir)

      if self.opts.info and self.chain_file is not None:
        print('Available chain file ', self.chain_file, '(',
              int(np.round(os.path.getsize(self.chain_file)/1e6)), ' Mb)')

  def load_chains(self):
    """ Loading Bilby result """
    try:
      self.result = br.read_in_result(filename=self.chain_file)
    except:
      print('Could not load file ', self.chain_file)
      return False
    self.chain = self.result.posterior
    self.chain_burn = self.chain
    return True

  def get_pars(self):
    return

  def _make_corner_plot(self):
    """ Corner plot for a posterior distribution from the result """
    if self.opts.corner == 1:
      self.result.plot_corner()

def main():
  """
  The pipeline script
  """

  opts = parse_commandline()

  if opts.bilby:
    result_obj = BilbyWarpResult(opts)
  elif opts.optimal_statistic:
    print('running OS analysis')
    result_obj = EnterpriseWarpOptimalStatistic(opts)
  else:
    result_obj = EnterpriseWarpResult(opts)

  result_obj.main_pipeline()

if __name__=='__main__':
  main()
