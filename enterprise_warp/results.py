"""
Result view and plotting for enterprise warp. Example:

>>> python -m enterprise_warp.results --result parameter_file_dat_or_directory --info 1
>>> python -m enterprise_warp.results --result enterprise_dir --result discovery_dir --corner 2 --hists 1 --chains 1

See `python -m enterprise_warp.results -h` for a list of available options:
    --corner 1: for corner plot (--corner 2 for ChainConsumer);
    --par gw: to only include parameter with "gw" in its name into the corner plot;
    --name J0437: to only plot results for pulsars which names contain string "J0437".
    --chains 1: to make a plot of MCMC chain for all parameters.
"""

import os
import re
import json
import glob
import copy
import shutil
import pickle
import optparse
import warnings
import itertools
import numpy as np
import scipy as sp


from datetime import datetime
from dateutil.parser import parse as pdate

from . import enterprise_warp

try:
  from enterprise.signals import signal_base
except Exception as ex:
  print(ex)
  warnings.warn('enterprise is not available')

try:
  from bilby import result as br
except Exception as ex:
  print(ex)
  warnings.warn('bilby is not available')

try:
    import matplotlib
    matplotlib.use('Agg')
    #from matplotlib import rcParams
    #rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'
    import matplotlib.pyplot as plt
    from corner import corner
except Exception as ex:
    print(ex)
    warnings.warn('matplotlib or corner are not available for making plots')

try:
  from chainconsumer import ChainConsumer, Chain, PlotConfig
except Exception as ex:
  print(ex)
  warnings.warn('[PLOTTING] chainconsumer is not available')

try:
  import pandas as pd
except Exception as ex:
  print(ex)
  warnings.warn('pandas not available, required for ChainConsumer, EnterpriseWarpResult._save_covm()')

try:
    from enterprise_extensions.frequentist.optimal_statistic import OptimalStatistic as OptStat
except Exception as ex:
    print(ex)
    warnings.warn('enterprise_extensions is not available, required for Optimal Statistic')

def add_result_commandline_options(parser=None):
  """
  Add command line arguments for action on results.
  """

  if parser is None:
    parser = optparse.OptionParser()

  parser.add_option("-r", "--result", help="Output directory or a parameter \
                    file. In case of individual pulsar analysis, specify a \
                    directory that contains subdirectories with individual \
                    pulsar results. In case of an array analysis, specify a \
                    directory with result files. Repeat --result to compare \
                    multiple runs.", action="append", default=None, type=str)

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
  parser.add_option("-A", "--par_equal", help="Treat parameters as equivalent \
                    across compared results, e.g. cw_log10_h0=crn_log10_hcw. \
                    The left-hand side is used as the plotted parameter name. \
                    Repeat to add more equivalences.",
                    action="append", default=None, type=str)

  parser.add_option("-t", "--truths", help="Truths for corner plots",
                    default=None, type=str)

  parser.add_option("-a", "--chains", help="Plot chains (1/0)", \
                    default=0, type=int)
                
  parser.add_option("-H", "--hists", help="Plot marginal posteriors (1/0)", \
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

  parser.add_option("-T", "--thin", help="Save a new chain file with N times \
                    less samples after 25% burn in, N=thin.", default=None, \
                    type=int)


  parser.add_option("-o", "--optimal_statistic", help="Calculate optimal \
                    statistic and make key plots (1/0)", default = 0,
                    type = int)

  parser.add_option("-g", "--optimal_statistic_orfs", help = "Set overlap \
                    reduction function form for optimal statistic analysis. \
                    Allowed options: \
                    all, hd (Hellings-Downs), quadrupole, dipole, monopole",
                    default = "hd,dipole,monopole", type = str)

  parser.add_option("-N", "--optimal_statistic_nsamples", help = "Set integer \
                    number of samples for noise-marginalised optimal statistic \
                    analysis.",
                    default = 1000, type = int)

  parser.add_option("-L", "--load_optimal_statistic_results", help = "load \
                    results from optimal statistic analysis. Do not recalculate\
                    any results. (1/0)",
                    default = 0, type = int)


  parser.add_option("-y", "--bilby", help="Load bilby result", \
                    default=0, type=int)
  parser.add_option("-d", "--discovery", help="Load Discovery result", \
                    default=0, type=int)

  parser.add_option("-P", "--custom_models_py", help = "Full path to a .py \
                    file with custom enterprise_warp model object, derived \
                    from enterprise_warp.StandardModels. It is only needed to \
                    correctly load a parameter file with unknown parameters. \
                    An alteriative: just use full path for --result, not \
                    the parameter file.",
                    default = None, type = str)

  parser.add_option("-M", "--custom_models", help = "Name of the custom \
                    enterprise_warp model object in --custom_models_py.",
                    default = None, type = str)

  parser.add_option("-R", "--realization", help = "For full-PTA run, \
                    whether to use 0/ or other folder (--num).",
                    default = 0, type = int)

  return parser


class ResultsParser(object):
  """
  Parser class for enterprise_warp result actions.
  """
  def __init__(self):
    self.parser = add_result_commandline_options()

  def parse_args(self):
    opts, args = self.parser.parse_args()
    return opts


def parse_commandline():
  """
  Parsing command line arguments for action on results.
  """
  return ResultsParser().parse_args()

class FakeResultOpts:
  """
  Fake output of parse_commandline()
  """
  def __init__(self, outdir):
    self.result = outdir
    self.info = 1
    self.name = "all"
    self.corner = 0
    self.par = None
    self.par_equal = None
    self.truths = None
    self.chains = 0
    self.hists = 0
    self.logbf = 0
    self.noisefiles = 0
    self.credlevels = 0
    self.covm = 0
    self.separate_earliest = 0
    self.load_separated = 0
    self.thin = None
    self.optimal_statistic = 0
    self.bilby = 0
    self.custom_models_py = None
    self.realization = 0

def normalize_result_args(result_arg):
  if result_arg is None:
    return []
  if isinstance(result_arg, (list, tuple, np.ndarray)):
    return list(result_arg)
  return [result_arg]

def clone_opts_with_result(opts, result_arg):
  opts_copy = copy.copy(opts)
  opts_copy.result = result_arg
  return opts_copy

def get_result_label(result_arg):
  norm_path = os.path.normpath(result_arg)
  label = os.path.basename(norm_path)
  return label if label else norm_path

def get_unique_result_labels(result_args):
  norm_paths = [os.path.normpath(result_arg) for result_arg in result_args]
  split_paths = [path.split(os.sep) for path in norm_paths]
  labels = [parts[-1] if parts[-1] else path
            for parts, path in zip(split_paths, norm_paths)]

  while len(labels) != len(set(labels)):
    duplicates = {label for label in labels if labels.count(label) > 1}
    for ii, label in enumerate(labels):
      if label not in duplicates:
        continue
      parts = split_paths[ii]
      if len(parts) > 1:
        split_paths[ii] = parts[:-1]
        parent = parts[-2]
        labels[ii] = os.path.join(parent, label)
      else:
        labels[ii] = '{}__{}'.format(label, ii + 1)

  return labels

def sanitize_filename_component(label):
  safe_label = str(label)
  if os.sep:
    safe_label = safe_label.replace(os.sep, '__')
  if os.altsep:
    safe_label = safe_label.replace(os.altsep, '__')
  safe_label = re.sub(r'[^A-Za-z0-9._+-]+', '_', safe_label)
  safe_label = safe_label.strip('._')
  return safe_label if safe_label else 'result'

def get_par_out_label(par_filters):
  return '' if par_filters is None else '_'.join(par_filters)

def get_plot_stride(size, target_samples):
  if size <= 0:
    return 1
  return max(1, int(size/target_samples))

def filter_parameter_names(par_names, par_filters=None):
  filtered_names = list()
  for par in par_names:
    par_name = str(par)
    if par_filters is None or any(flt in par_name for flt in par_filters):
      filtered_names.append(par_name)
  return filtered_names

def parse_equal_par_specs(par_equal_specs):
  if par_equal_specs is None:
    return {}

  canonical_map = {}
  for spec in par_equal_specs:
    if '=' not in spec:
      raise ValueError('Invalid --par_equal specification: {}'.format(spec))
    lhs, rhs = spec.split('=', 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs or not rhs:
      raise ValueError('Invalid --par_equal specification: {}'.format(spec))
    canonical_map.setdefault(lhs, [])
    for par_name in [lhs, rhs]:
      if par_name not in canonical_map[lhs]:
        canonical_map[lhs].append(par_name)
  return canonical_map

def get_canonical_par_name(par_name, equal_par_map=None):
  if equal_par_map is None:
    return str(par_name)
  par_name = str(par_name)
  for canonical_name, aliases in equal_par_map.items():
    if par_name in aliases:
      return canonical_name
  return par_name

def get_union_parameter_names(results, par_filters=None, equal_par_map=None):
  par_union = list()
  seen = set()
  for result_obj in results:
    for par in np.asarray(result_obj.pars, dtype=str):
      canonical_par = get_canonical_par_name(par, equal_par_map=equal_par_map)
      if canonical_par not in seen:
        par_union.append(canonical_par)
        seen.add(canonical_par)
  return filter_parameter_names(par_union, par_filters=par_filters)

def get_common_psr_dirs(results, name_filter='all'):
  if not results:
    return []
  common_dirs = set(np.asarray(results[0].psr_dirs, dtype=str).tolist())
  for result_obj in results[1:]:
    common_dirs &= set(np.asarray(result_obj.psr_dirs, dtype=str).tolist())
  common_dirs = sorted(common_dirs)
  if name_filter != 'all':
    common_dirs = [psr_dir for psr_dir in common_dirs if name_filter in psr_dir]
  return common_dirs

def get_HD_curve(zeta):
  coszeta = np.cos(zeta)
  xip = (1.-coszeta) / 2.
  HD = 3.*( 1./3. + xip * ( np.log(xip) -1./6.) )

  return HD/2.0

def get_dipole_curve(zeta):
  coszeta = np.cos(zeta)
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
  outdir = 'noisefiles/', recompute = True):
  """
  Create noise dictionary for a given MCMC or nested sampling chain.
  This is a dict that assigns a characteristic value (mode/median)
  to a parameter from the distribution of parameter values in a chain.
  Can be used for outputting a noise file or for use in further
  analysis (e.g. optimal statistic)
  """
  result_filename = outdir + '/' + psrname + '_' + suffix + '.json'

  if not recompute:
    if os.path.exists(result_filename):
      xx = json.load(open(result_fileneame, 'r'))
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
  return bool(re.match(r'^\d{1,}_[J,B]\d{2,4}[+,-]\d{2,4}[A,B]{0,1}$',
  folder_name))



class OptimalStatisticResult(object):

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

    weights, avg = 0., 0.
    for r,s in zip(_rho, _sig):
      weights += 1./(s*s)
      avg += r/(s*s)

    return avg/weights, np.sqrt(1./weights)
    #return np.average(_rho, _sig**-2.0), np.sqrt(np.sum(_sig**-2.0))

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

  def avg_ostat_bins(self, n_psr):
    # sort the cross-correlations by xi
    idx = np.argsort(self.xi)

    xi_sorted = self.xi[idx]
    rho_sorted = self.rho[idx]
    sig_sorted = self.sig[idx]

    # bin the cross-correlations so that there are the same number of \
    #pairs per bin
    # n_psr = len(self.psrs)
    npairs = int(n_psr*(n_psr - 1.0)/2.0)

    xi_avg = []
    xi_err = []

    rho_avg = []
    sig_avg = []

    ii = 0
    while ii < len(xi_sorted):

      xi_avg.append(np.mean(xi_sorted[ii:int(npairs/8)+ii]))
      xi_err.append(np.std(xi_sorted[ii:int(npairs/8)+ii]))

      r, s = self.weightedavg(rho_sorted[ii:int(npairs/8)+ii], \
        sig_sorted[ii:npairs+ii])
      rho_avg.append(r)
      sig_avg.append(s)

      ii += int(npairs/8)

    xi_avg = np.array(xi_avg)
    xi_err = np.array(xi_err)
    rho_avg = np.array(rho_avg)
    sig_avg = np.array(sig_avg)

    #do we want to return these or add them as class attributes?
    self.xi_avg = xi_avg
    self.xi_err = xi_err
    self.rho_avg = rho_avg
    self.sig_avg = sig_avg
    #return xi_mean, xi_err, rho_avg, sig_avg


class EnterpriseWarpResult(object):

  def __init__(self, opts, custom_models_obj=None):
    self.opts = opts
    self.custom_models_obj = custom_models_obj
    self.truth_path = None
    self.truth_values = None
    self._truth_cache = {}
    self._auto_truth_paths_seen = set()
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
              self.opts.chains or self.opts.hists or self.opts.thin):
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
      self._make_histograms()

    self._save_covm()

  def _scan_psr_output(self):

    self.outdir = self.outdir_all + '/' + self.psr_dir + '/'
    if self.opts.name != 'all' and self.opts.name not in self.psr_dir:
      return False
    print('Processing ', self.psr_dir)

    self.get_pars()
    self.get_chain_file_name()
    self._detect_truth_file()
    self._load_truth_values()

    return True

  def _detect_truth_file(self):
    if self.opts.truths is not None:
      self.truth_path = self.opts.truths
      return
    candidates = []
    if self.outdir:
      candidates.append(os.path.join(self.outdir, 'truth.json'))
    if self.outdir_all:
      candidates.append(os.path.join(self.outdir_all, 'truth.json'))
    self.truth_path = None
    for candidate in candidates:
      if os.path.isfile(candidate):
        self.truth_path = candidate
        if candidate not in self._auto_truth_paths_seen:
          print('Using truth file ', candidate)
          self._auto_truth_paths_seen.add(candidate)
        break

  def _load_truth_values(self):
    self.truth_values = None
    if self.truth_path is None:
      return
    if self.truth_path in self._truth_cache:
      self.truth_values = self._truth_cache[self.truth_path]
      return
    try:
      with open(self.truth_path, 'r') as truth_file:
        truths = json.load(truth_file)
    except Exception as exc:
      warnings.warn('Could not load truths from {}: {}'.format(
                    self.truth_path, exc))
      return
    if not isinstance(truths, dict):
      warnings.warn('Truth file {} does not contain a JSON object, ignoring.'
                    .format(self.truth_path))
      return
    self.truth_values = truths
    self._truth_cache[self.truth_path] = truths

  def _get_truths_for_pars(self, par_subset):
    if self.truth_values is None:
      return None
    try:
      return [self.truth_values[str(par)] for par in par_subset]
    except KeyError as exc:
      warnings.warn('Truth value for {} not found in {}'.format(
                    exc, self.truth_path))
      return None

  def _get_truth_value(self, par_name):
    if self.truth_values is None:
      return None
    return self.truth_values.get(str(par_name))

  def get_display_label(self):
    return get_result_label(self.opts.result)

  def get_filtered_par_names(self):
    pars = np.asarray(self.pars, dtype=str)
    if 'par_mask' in self.__dict__:
      return pars[self.par_mask]
    return pars

  def get_samples_for_par(self, par_name, burned=True):
    pars = np.asarray(self.pars, dtype=str)
    matches = np.where(pars == str(par_name))[0]
    if matches.size == 0:
      return None
    chain_source = self.chain_burn if burned else self.chain
    return chain_source[:, matches[0]]

  def get_samples_for_equal_par(self, par_name, equal_par_map=None, burned=True):
    alias_names = [str(par_name)]
    if equal_par_map is not None and str(par_name) in equal_par_map:
      alias_names = [str(alias) for alias in equal_par_map[str(par_name)]]

    matched_indices = list()
    pars = np.asarray(self.pars, dtype=str)
    for alias_name in alias_names:
      matches = np.where(pars == alias_name)[0]
      if matches.size > 0:
        matched_indices.extend(matches.tolist())

    if len(matched_indices) == 0:
      return None
    if len(matched_indices) > 1:
      warnings.warn('More than one equivalent parameter found for {} in {}. '
                    'Ignoring this result for that plotted parameter.'.format(
                    par_name, self.get_display_label()))
      return None

    chain_source = self.chain_burn if burned else self.chain
    return chain_source[:, matched_indices[0]]

  def get_truth_value_for_equal_par(self, par_name, equal_par_map=None):
    truth_val = self._get_truth_value(par_name)
    if truth_val is not None:
      return truth_val
    if equal_par_map is None or str(par_name) not in equal_par_map:
      return None
    for alias_name in equal_par_map[str(par_name)]:
      truth_val = self._get_truth_value(alias_name)
      if truth_val is not None:
        return truth_val
    return None

  def interpret_opts_result(self):
    """ Determine output directory from the --results argument """
    if os.path.isdir(self.opts.result):
      self.outdir_all = self.opts.result
    elif os.path.isfile(self.opts.result):
      self.params = enterprise_warp.Params(self.opts.result, \
                      init_pulsars=False, \
                      custom_models_obj=self.custom_models_obj)
      if self.params.array_analysis:
        self.outdir_all = self.params.out + self.params.label_models + '_' + \
                          self.params.paramfile_label + '/' + str(self.opts.realization) + '/'
      else:
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
    if self.opts.thin is not None:
      new_chain_fname = self.chain_file.replace("chain","thin_chain")
      np.savetxt(new_chain_fname, self.chain[burn::self.opts.thin,:])
      print("Saved thinned chain:", new_chain_fname)
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
    self.logbf = {}
    if self.opts.logbf:
      print('=====', self.psr_dir, ' model selection results', '=====')
      print('Samples in favor of models: ', self.dict_real_counts)
      if len(self.unique) > 1:
        count_by_pairs = list(itertools.combinations(sorted(self.unique), 2))
        self.logbf = {}
        for combination in count_by_pairs:
          logbf = np.log(self.dict_real_counts[combination[1]] / \
                         self.dict_real_counts[combination[0]])
          print('logBF for ', int(combination[1]), 'over ', \
                int(combination[0]),': ', logbf)
          self.logbf[combination] = logbf


  def _make_corner_plot(self):
    """ Corner plot for a posterior distribution from the result """
    if self.opts.corner == 1:
      truths = self._get_truths_for_pars(self.pars[self.par_mask])
      for jj in self.unique:
        if jj is not None:
          model_mask = np.round(self.chain_burn[:,self.ind_model]) == jj
        else:
          model_mask = np.repeat(True, self.chain_burn.shape[0])
        chain_plot = self.chain_burn[model_mask,:]
        chain_plot = chain_plot[:,self.par_mask]
        figure = corner(chain_plot, 30, labels=self.pars[self.par_mask], \
                        truths=truths)
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
        pd_chain = pd.DataFrame(chain_plot, \
                   columns=pars[self.par_mask].tolist())
        cobj.add_chain(Chain(samples=pd_chain, name=str(jj)))
      cobj.set_plot_config(PlotConfig(serif=True, label_font_size=12, 
                     tick_font_size=12, legend_color_text=False, 
                     legend_artists=True))
      corner_name = self.outdir_all + '/' + self.psr_dir + '_' + \
                    self.par_out_label + '_corner.png'
      fig = cobj.plotter.plot(filename=corner_name)
      plt.close()

  def _make_histograms(self):
    """ Histograms for the posterior distribution for all parameters """
    if self.opts.hists:
       thin_factor = 1000
       x_tiles = int(np.floor(len(self.pars)**0.5))
       y_tiles = int(np.ceil(len(self.pars)/x_tiles))
       plt.figure(figsize=[6.4*x_tiles,4.8*y_tiles])
       for pp, par in enumerate(self.pars):
          plt.subplot(x_tiles, y_tiles, pp + 1)
          stride = get_plot_stride(self.chain[:,pp].size, thin_factor)
          cut_chain = self.chain[::stride,pp]
          plt.hist(cut_chain,label=par.replace('_','\n'),bins=50)
          truth_val = self._get_truth_value(par)
          if truth_val is not None:
            plt.axvline(truth_val, color='r', linestyle='--', linewidth=1.5)
          plt.legend()
          plt.xlabel('Parameter')
          plt.ylabel('Density')
       plt.subplots_adjust(wspace=0.)
       plt.tight_layout()
       plt.savefig(self.outdir_all + '/' + self.psr_dir + '_hist_pars_' + \
                   '.png')
       plt.close()


  def _make_chain_plot(self):
    """ MCMC chain plots (evolution in time) """
    if self.opts.chains:
       thin_factor = 200
       x_tiles = int(np.floor(len(self.pars)**0.5))
       y_tiles = int(np.ceil(len(self.pars)/x_tiles))
       plt.figure(figsize=[6.4*x_tiles,4.8*y_tiles])
       for pp, par in enumerate(self.pars):
          plt.subplot(x_tiles, y_tiles, pp + 1)
          stride = get_plot_stride(self.chain[:,pp].size, thin_factor)
          cut_chain = self.chain[::stride,pp]
          plt.plot(cut_chain,label=par.replace('_','\n'))
          plt.legend()
          plt.xlabel('Thinned MCMC iterations')
          plt.ylabel('Value')
       plt.subplots_adjust(wspace=0.)
       plt.tight_layout()
       plt.savefig(self.outdir_all + '/' + self.psr_dir + '_samples_trace_' + \
                   '.png')
       plt.close()

class OptimalStatisticWarp(EnterpriseWarpResult):
  def __init__(self, opts):
    super(OptimalStatisticWarp, self).__init__(opts)
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

      if self.opts.load_optimal_statistic_results == 1:
        self.load_results()
      else:
        self._add_optimalstatistics(method = 'mode')
        self._marginalise_ostat()
        self._avg_ostat_bins()
        self.dump_results()

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
      self.params = enterprise_warp.Params(self.opts.result, init_pulsars=True)
      self.psrs = self.params.psrs
      #might want to include custom models support here
      self.outdir_all = self.params.out + self.params.label_models + '_' + \
                        self.params.paramfile_label + '/'

  def _get_pta(self):
    #hard-coding in choice of model 0 here
    self.pta = enterprise_warp.init_pta(self.params)[0]

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
      print('Computing optimal statistic for {} ORF'.format(orf))
      _os = OptStat(self.params.psrs, pta = self.pta, orf = orf)
      _xi, _rho, _sig, _OS, _OS_sig = _os.compute_os(params=os_params)

      result = OptimalStatisticResult(_os, os_params, _xi, _rho, _sig, _OS, \
                                      _OS_sig)
      optstat_dict[orf] = result

    if method == 'samp':
        return optstat_dict #this is probably useless
    else:
        self.OptimalStatisticResults = optstat_dict
        #this is a representative optimal statistic
        return True


  def _marginalise_ostat(self):

    for orf in self.optstat_orfs:
      print(self.OptimalStatisticResults)
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
          #_ostat_dict = self._compute_optimalstatistic(method = 'samp', \
                                                      #  chain_idx = \
                                                      #  samp_indices[ii] \
                                                      # )
          _noisemarg_os[ii] = _OS
          _noisemarg_os_err[ii] = _OS_sig

      _osr.add_marginalised(_noisemarg_os, _noisemarg_os_err)

  def _avg_ostat_bins(self):
    for orf, _osr in self.OptimalStatisticResults.items():
      _osr.avg_ostat_bins(len(self.params.psrs))

  def plot_os_orf(self):

    orf_funcs = {'hd': get_HD_curve, \
                 'dipole': get_dipole_curve, \
                 'monopole': get_monopole_curve
                }


    color_dict = {'hd': 'C3', \
                  'dipole': 'C2', \
                  'monopole': 'C0'\
                 }

    default_linewidth = 0.8
    highlight_linewidth = 1.8

    _orf = self.optstat_orfs[0]
    _osr = self.OptimalStatisticResults[_orf]

    _xi_avg = _osr.xi_avg
    _rho_avg = _osr.rho_avg
    _xi_err = _osr.xi_err
    _sig_avg = _osr.sig_avg
    _OS = _osr.OS
    fig, ax = plt.subplots(1, 1, figsize = (3.25, 2.008))
    (_, caps, _) = ax.errorbar(_xi_avg,\
                               _rho_avg,\
                               xerr = _xi_err,\
                               yerr = _sig_avg,\
                               marker = 'o',\
                               ls = '', \
                               color = 'k', ##4FC3F7' \
                               fmt = 'o',\
                               capsize = 4,\
                               elinewidth = 1.2\
                              )


    zeta = np.linspace(0.001, np.pi, 200)

    for __orf in self.optstat_orfs:
      curve = orf_funcs[__orf]
      orf_curve = curve(zeta)
      __OS = self.OptimalStatisticResults[__orf].OS

      # if __orf == '_orf':
      #   linewidth = highlight_linewidth
      # else:
      #   linewidth = default_linewidth
      linewidth = highlight_linewidth

      ax.plot(zeta, __OS*orf_curve, \
              linestyle = '--', \
              color = color_dict[__orf], \
              linewidth = linewidth\
             )

    ax.set_xlim(0, np.pi)
    ylo, yhi = ax.get_ylim()
    m = np.amax([np.abs(ylo), np.abs(yhi)])
    ax.set_ylim(-m, m)
    ax.set_xlabel(r'$\zeta$ (rad)')
    ax.set_ylabel(r'$\hat{{A}}^2 \Gamma_{{ab}}(\zeta)$')
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(
                self.outdir_all + '/' + self.psr_dir + '_os_orf_' + \
                self.par_out_label + '.png', dpi = 300, \
                bbox_inches = 'tight' \
                )
    plt.close(fig)

  def plot_noisemarg_os(self):
    from astropy.visualization import hist as a_hist
    #plot OS S/N
    color_dict = {'hd': 'C3', \
                  'dipole': 'C2', \
                  'monopole': 'C0'\
                 }

    fig1, ax1 = plt.subplots(1, 1, figsize = (3.25, 2.008))
    fig2, ax2 = plt.subplots(1, 1, figsize = (3.25, 2.008))
    for orf, _osr in self.OptimalStatisticResults.items():
      _noisemarg_os = _osr.marginalised_os
      _noisemarg_os_err = _osr.marginalised_os_err
      _color = color_dict[orf]
      _os = _osr.OS
      _os_err = _osr.OS_err
      a_hist(_noisemarg_os/_noisemarg_os_err, \
             histtype = 'step', \
             color = _color, \
             label = orf, \
             density = True, \
             ax = ax1, \
             bins = 'knuth' \
            )

      ax1.axvline(np.mean(_noisemarg_os/_noisemarg_os_err), \
                  linestyle = '--', \
                  color = _color, \
                  linewidth = 0.8 \
                 )
      
      ax1.axvline(_os/_os_err, \
                  linestyle = '-.', \
                  color = _color, \
                  linewidth = 0.8 \
                 )
      
      if orf == 'monopole':
        bins = 'blocks'
      else:
        bins = 'knuth'

      a_hist(_noisemarg_os, \
             histtype = 'step', \
             color = _color, \
             label = orf, \
             density = True, \
             ax = ax2, \
             bins = bins \
            )
       
      ax2.axvline(np.mean(_noisemarg_os), \
                  linestyle = '--', \
                  color = _color, \
                  linewidth = 0.8, \
                 )
      
      ax2.axvline(_os, \
                  linestyle = '-.', \
                  color = _color, \
                  linewidth = 0.8
                 )

    ax1.legend(fontsize = 9)
    ax1.set_xlabel('SNR')
    ax1.set_ylabel('Probability density')
    ax1.minorticks_on()
    fig1.savefig(self.outdir_all + '/' + self.psr_dir + '_os_SNR_' +  '_' +\
                 self.par_out_label + '.png', dpi = 300, bbox_inches = 'tight')
    plt.close(fig1)

    a_hist((10.0**(self.gw_log10_A))**2.0, \
           histtype = 'step', \
           color = '0.5', \
           label = 'uncorrelated', \
           density = True, \
           ax = ax2, \
           bins = 'knuth' \
          )
    
    ax2.axvline((10.0**(np.mean(self.gw_log10_A)))**2.0, linestyle = '--', \
                color = '0.5', linewidth = 0.8)
    ax2.legend(fontsize = 9)
    ax2.set_xlabel('$\hat{{A}}^{{2}} \& {{A}}^{{2}}_{{\mathrm{{CP}}}}$')
    ax2.set_ylabel('Probability density')
    ax2.set_xlim(-2.0E-29, 8E-29)
    ax2.minorticks_on()
    fig2.savefig(self.outdir_all + '/' + self.psr_dir + '_os_A2_' + \
                 '_' + self.par_out_label + '.png', dpi = 300, \
                 bbox_inches = 'tight')
    plt.close(fig2)

  def dump_results(self):
    fname = self.outdir_all + '/' + self.psr_dir + '_os_results.pkl'
    with open(fname, 'wb') as _file:
      dump_struct = dict()
      for _orf in self.optstat_orfs:
        
        _result = self.OptimalStatisticResults[_orf]
        orf_dump = {'params': _result.params, \
                    'xi':    _result.xi, \
                    'rho':   _result.rho, \
                    'sig':   _result.sig, \
                    'OS':    _result.OS, \
                    'OS_err': _result.OS_err, \
                    'marginalised_os': _result.marginalised_os, \
                    'marginalised_os_err': _result.marginalised_os_err, \
                    'xi_avg': _result.xi_avg, \
                    'xi_err': _result.xi_err, \
                    'rho_avg': _result.rho_avg, \
                    'sig_avg': _result.sig_avg
        }
      
        dump_struct[_orf] = orf_dump
      
      pickle.dump(dump_struct, _file)
      
    return True

  def load_results(self):
    fname = self.outdir_all + '/' + self.psr_dir + '_os_results.pkl'
    _file = open(fname, 'r')
    _OptimalStatisticResults = pickle.load(_file) #dumped pickle is not an optimalstatisticresult. this is broken - need to fix
    self.OptimalStatisticResults = _OptimalStatisticResults
    #need to add functionalitu
    return True


class DiscoveryWarpResult(EnterpriseWarpResult):
  """
  Result handler for Discovery/NumPyro runs that save CSV chains.
  """

  def get_chain_file_name(self):
    csv_candidates = sorted(glob.glob(os.path.join(self.outdir, "*_chain.csv")))
    if not csv_candidates:
      csv_candidates = sorted(glob.glob(os.path.join(self.outdir, "*.csv")))
    self.chain_file = csv_candidates[0] if csv_candidates else None
    if self.chain_file is None:
      print('Could not find CSV chain file in ', self.outdir)
    elif self.opts.info:
      size_mb = int(np.round(os.path.getsize(self.chain_file) / 1e6))
      print('Available CSV chain file ', self.chain_file, '(', size_mb, ' Mb)')

  def load_chains(self):
    """Load NumPyro CSV chains."""
    if self.chain_file is None:
      print('No chain file selected for ', self.outdir)
      return False
    try:
      df = pd.read_csv(self.chain_file)
    except Exception as exc:
      print('Could not load CSV ', self.chain_file, ':', exc)
      return False
    if df.empty:
      print('Empty CSV chain file in ', self.outdir)
      return False

    # Discovery CSV headers contain the true parameter ordering (including vector entries),
    # so replace the pars/mask derived from pars.txt to keep shapes consistent.
    self.pars = np.asarray(df.columns, dtype=str)
    self._get_par_mask()

    self.chain = df.to_numpy()
    burn = int(0.1 * self.chain.shape[0])
    burn = min(self.chain.shape[0], max(burn, 0))
    self.chain_burn = self.chain[burn:, :]

    self.ind_model = 0
    self.unique, self.counts, self.dict_real_counts = [None], None, None
    return True


class ResultCollection(object):
  """
  Compare multiple result directories using their loaded sample arrays.
  """

  def __init__(self, opts, result_cls, custom_models_obj=None):
    self.opts = opts
    self.result_cls = result_cls
    self.custom_models_obj = custom_models_obj
    self.result_args = normalize_result_args(self.opts.result)
    self.result_objs = list()
    for result_arg in self.result_args:
      member_opts = clone_opts_with_result(self.opts, result_arg)
      result_obj = self.result_cls(member_opts,
                                   custom_models_obj=self.custom_models_obj)
      self.result_objs.append(result_obj)
    self.result_labels = get_unique_result_labels(self.result_args)
    self.filename_labels = [sanitize_filename_component(label)
                            for label in self.result_labels]
    self.equal_par_map = parse_equal_par_specs(getattr(self.opts, 'par_equal', None))
    self.outdir_all = self.result_objs[0].outdir_all
    self.psr_dirs = get_common_psr_dirs(self.result_objs, self.opts.name)
    self.par_out_label = get_par_out_label(self.opts.par)

  def _build_output_prefix(self):
    return '__vs__'.join(self.filename_labels)

  def _get_truth_value(self, par_name):
    if not self.loaded_results:
      return None
    return self.loaded_results[0].get_truth_value_for_equal_par(
        par_name, equal_par_map=self.equal_par_map)

  def _get_plot_pars(self):
    return get_union_parameter_names(self.loaded_results,
                                     par_filters=self.opts.par,
                                     equal_par_map=self.equal_par_map)

  def _load_results_for_psr_dir(self, psr_dir):
    loaded_results = list()
    for result_obj in self.result_objs:
      result_obj.psr_dir = psr_dir
      success = result_obj._scan_psr_output()
      if not success:
        return None
      if not (self.opts.corner or self.opts.chains or self.opts.hists):
        loaded_results.append(result_obj)
        continue
      success = result_obj.load_chains()
      if not success:
        return None
      loaded_results.append(result_obj)
    return loaded_results

  def main_pipeline(self):
    if not self.psr_dirs:
      warnings.warn('No common result directories found among the supplied --result inputs.')
      return

    for psr_dir in self.psr_dirs:
      self.psr_dir = psr_dir
      self.loaded_results = self._load_results_for_psr_dir(psr_dir)
      if self.loaded_results is None:
        continue
      self.output_prefix = self._build_output_prefix()
      self._make_corner_plot()
      self._make_chain_plot()
      self._make_histograms()

  def _make_corner_plot(self):
    if not self.opts.corner:
      return
    if self.opts.corner == 1:
      raise ValueError('ResultCollection only supports --corner 2 for comparison plots.')

    par_names = self._get_plot_pars()
    if not par_names:
      warnings.warn('No parameters selected for comparison corner plot.')
      return

    cobj = ChainConsumer()
    chain_columns = ['$'+par+'$' for par in par_names]
    for result_index, result_obj in enumerate(self.loaded_results):
      result_label = self.result_labels[result_index]
      available = [par for par in par_names if
                   result_obj.get_samples_for_equal_par(
                       par, equal_par_map=self.equal_par_map) is not None]
      if not available:
        continue
      pd_chain = pd.DataFrame({('$'+par+'$'): result_obj.get_samples_for_equal_par(
                                  par, equal_par_map=self.equal_par_map)
                               for par in available},
                              columns=['$'+par+'$' for par in available])
      cobj.add_chain(Chain(samples=pd_chain, name=result_label))

    cobj.set_plot_config(PlotConfig(serif=True, label_font_size=12,
                   tick_font_size=12, legend_color_text=False,
                   legend_artists=True))
    corner_name = self.outdir_all + '/' + self.psr_dir + '_comparison_corner_' + \
                  self.output_prefix + '_' + self.par_out_label + '.png'
    cobj.plotter.plot(columns=chain_columns, filename=corner_name)
    plt.close()

  def _make_histograms(self):
    if not self.opts.hists:
      return

    par_names = self._get_plot_pars()
    if not par_names:
      warnings.warn('No parameters selected for comparison histogram plot.')
      return

    x_tiles = int(np.floor(len(par_names)**0.5))
    y_tiles = int(np.ceil(len(par_names)/x_tiles))
    plt.figure(figsize=[6.4*x_tiles,4.8*y_tiles])
    for pp, par in enumerate(par_names):
      plt.subplot(x_tiles, y_tiles, pp + 1)
      available_samples = list()
      for result_obj in self.loaded_results:
        par_samples = result_obj.get_samples_for_equal_par(
            par, equal_par_map=self.equal_par_map, burned=True)
        if par_samples is not None:
          available_samples.append(np.asarray(par_samples))
      if not available_samples:
        continue

      combined_samples = np.concatenate(available_samples)
      bins = np.histogram_bin_edges(combined_samples, bins=50)
      for rr, result_obj in enumerate(self.loaded_results):
        par_samples = result_obj.get_samples_for_equal_par(
            par, equal_par_map=self.equal_par_map, burned=True)
        if par_samples is None:
          continue
        plt.hist(par_samples, bins=bins, alpha=0.35, density=True,
                 label=self.result_labels[rr], color='C{}'.format(rr % 10))

      truth_val = self._get_truth_value(par)
      if truth_val is not None:
        plt.axvline(truth_val, color='r', linestyle='--', linewidth=1.5)
      plt.legend()
      plt.xlabel('Parameter')
      plt.ylabel('Density')
      plt.title(par.replace('_','\n'))
    plt.subplots_adjust(wspace=0.)
    plt.tight_layout()
    plt.savefig(self.outdir_all + '/' + self.psr_dir + '_comparison_hist_pars_' + \
                self.output_prefix + '_' + self.par_out_label + '.png')
    plt.close()

  def _make_chain_plot(self):
    if not self.opts.chains:
      return

    par_names = self._get_plot_pars()
    if not par_names:
      warnings.warn('No parameters selected for comparison chain plot.')
      return

    thin_factor = 200
    x_tiles = int(np.floor(len(par_names)**0.5))
    y_tiles = int(np.ceil(len(par_names)/x_tiles))
    plt.figure(figsize=[6.4*x_tiles,4.8*y_tiles])
    for pp, par in enumerate(par_names):
      plt.subplot(x_tiles, y_tiles, pp + 1)
      for rr, result_obj in enumerate(self.loaded_results):
        par_samples = result_obj.get_samples_for_equal_par(
            par, equal_par_map=self.equal_par_map, burned=False)
        if par_samples is None:
          continue
        stride = get_plot_stride(par_samples.size, thin_factor)
        plt.plot(par_samples[::stride], label=self.result_labels[rr],
                 alpha=0.7, color='C{}'.format(rr % 10))
      plt.legend()
      plt.xlabel('Thinned MCMC iterations')
      plt.ylabel('Value')
      plt.title(par.replace('_','\n'))
    plt.subplots_adjust(wspace=0.)
    plt.tight_layout()
    plt.savefig(self.outdir_all + '/' + self.psr_dir + '_comparison_samples_trace_' + \
                self.output_prefix + '_' + self.par_out_label + '.png')
    plt.close()


class BilbyWarpResult(EnterpriseWarpResult):

  def __init__(self, opts):
    super(BilbyWarpResult, self).__init__(opts)

  def get_chain_file_name(self):
      if 'params' in self.__dict__:
        label = os.path.basename(os.path.normpath(self.params.out))
        if os.path.isfile(self.outdir + '/' + label + '_result.json'):
          self.chain_file = self.outdir + '/' + label + '_result.json'
        else:
          self.chain_file = None
          print('Could not find chain file in ',self.outdir)
      else:
        self.chain_file = glob.glob(self.outdir+"*_result.json")[0]

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
    self.chain = np.array(self.result.posterior)
    self.chain_burn = self.chain
    self.pars = self.result.parameter_labels
    return True

  def get_pars(self):
    return

  def _make_corner_plot(self):
    """ Corner plot for a posterior distribution from the result """
    if self.opts.corner == 1:
      self.result.plot_corner(filename=self.outdir_all+'/'+self.psr_dir+'_corner.png')
    else:
      raise ValueError('Only --corner 1 is supported for Bilby.')

def main():
  """
  The pipeline script
  """

  opts = parse_commandline()
  result_args = normalize_result_args(opts.result)
  if not result_args:
    raise ValueError('Please supply at least one --result.')
  opts.result = result_args if len(result_args) > 1 else result_args[0]

  if opts.custom_models is not None and opts.custom_models_py is not None:
    import importlib
    spec = importlib.util.spec_from_file_location("custom_models_obj", \
                                                  opts.custom_models_py)
    cmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmod)
    custom_models_obj = cmod.__dict__[opts.custom_models]
  elif opts.custom_models is None and opts.custom_models_py is None:
    custom_models_obj = None
  else:
    raise ValueError('Please set both --custom_models and --custom_models_obj')

  if opts.bilby and len(result_args) > 1:
    raise ValueError('Multiple --result inputs are not supported with --bilby.')
  if opts.optimal_statistic and len(result_args) > 1:
    raise ValueError('Multiple --result inputs are not supported with --optimal_statistic.')

  if opts.bilby:
    result_obj = BilbyWarpResult(opts)
  elif opts.optimal_statistic:
    print('running OS analysis')
    result_obj = OptimalStatisticWarp(opts)
  elif len(result_args) > 1:
    result_cls = DiscoveryWarpResult if opts.discovery else EnterpriseWarpResult
    result_obj = ResultCollection(opts, result_cls,
                                  custom_models_obj=custom_models_obj)
  elif opts.discovery:
    result_obj = DiscoveryWarpResult(opts, custom_models_obj=custom_models_obj)
  else:
    result_obj = EnterpriseWarpResult(opts, custom_models_obj=custom_models_obj)

  result_obj.main_pipeline()

if __name__=='__main__':
  main()
