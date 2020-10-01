import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

import os
import re
import json
import optparse
import itertools
import numpy as np
from corner import corner

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

  opts, args = parser.parse_args()

  return opts


def estimate_from_distribution(values, method='mode'):
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
    nb, bins, patches = plt.hist(values, bins=50)
    plt.close()
    return bins[np.argmax(nb)]


def make_noise_files(psrname, chain, pars, outdir='noisefiles/',
                     method='mode'):
  """
  Create noise files from a given MCMC or nested sampling chain.
  Noise file is a dict that assigns a characteristic value (mode/median)
  to a parameter from the distribution of parameter values in a chain.
  """
  x = {}
  for ct, par in enumerate(pars):
    x[par] = estimate_from_distribution(chain[:,ct], method=method)

  os.system('mkdir -p {}'.format(outdir))
  with open(outdir + '/{}_noise.json'.format(psrname), 'w') as fout:
      json.dump(x, fout, sort_keys=True, indent=4, separators=(',', ': '))


def check_if_psr_dir(folder_name):
  """
  Check if the folder name (without path) is in the enterprise_warp format: 
  integer, underscore, pulsar name.
  """
  return bool(re.match(r'^\d{1,}_[J,B]\d{2,4}[+,-]\d{4,4}[A,B]{0,1}$', 
                       folder_name))

class EnterpriseWarpResult(object):

  def __init__(self, opts):
    self.opts = opts
    self.iterpret_opts_result()
    self.get_psr_dirs()

  def main_pipeline(self):
    for psr_dir in self.psr_dirs:

      self.psr_dir = psr_dir
      success = self._scan_psr_output()
      if not success:
        continue

      if not (self.opts.noisefiles or self.opts.logbf or self.opts.corner or \
              self.opts.chains):
        continue

      success = self.load_chains()
      if not success:
        continue

      self._get_par_mask()
      self._make_noisefiles()
      self._print_logbf()
      self._make_corner_plot()
      self._make_chain_plot()

  def _scan_psr_output(self):

    self.outdir = self.outdir_all + '/' + self.psr_dir + '/'
    if self.opts.name is not 'all' and self.opts.name not in self.psr_dir:
      return False
    print('Processing ', self.psr_dir)

    self.get_chain_file_name()
    self.get_pars()

    return True

  def iterpret_opts_result(self):
    """ Determine output directory from the --results argument """
    if os.path.isdir(self.opts.result):
      self.outdir_all = self.opts.result
    elif os.path.isfile(self.opts.result):
      params = enterprise_warp.Params(self.opts.result, init_pulsars=False)
      self.outdir_all = params.out + params.label_models + '_' + \
                        params.paramfile_label + '/'
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
    self.pars = np.loadtxt(self.outdir + '/pars.txt', dtype=np.unicode_)
    if self.opts.info and (self.opts.name != 'all' or self.psr_dir == ''):
      print('Parameter names:')
      for par in self.pars:
        print(par)


  def load_chains(self):
    """ Loading PTMCMC chains """
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

    self.ind_model = list(self.pars).index('nmodel')
    self.unique, self.counts = np.unique(np.round( \
                               self.chain_burn[:, self.ind_model]), \
                               return_counts=True)
    self.dict_real_counts = dict(zip(self.unique.astype(int),
                                     self.counts.astype(float)))
    return True


  def _get_par_mask(self):
    """ Get an array mask to select only parameters chosen with --par """
    if self.opts.par is not None:
      masks = list()
      for pp in self.opts.par:
        masks.append( [True if pp in label else False for label in self.pars] )
      self.par_mask = np.sum(masks, dtype=bool, axis=0)
      self.par_out_label = '_'.join(self.opts.par)
    else:
      self.par_mask = np.repeat(True, len(self.pars))
      self.par_out_label = ''


  def _make_noisefiles(self):
    if self.opts.noisefiles:
      make_noise_files(self.psr_dir, self.chain_burn, self.pars,
                       outdir = self.outdir_all + '/noisefiles/')

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
        model_mask = np.round(self.chain_burn[:,self.ind_model]) == jj
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
        model_mask = np.round(self.chain_burn[:,self.ind_model]) == jj
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
       plt.figure(figsize=[6.4,4.8*len(self.pars)])
       for pp, par in enumerate(self.pars):
          plt.subplot(len(self.pars), 1, pp+1)
          cut_chain = self.chain[::int(self.chain[:,pp].size/thin_factor),pp]
          plt.plot(cut_chain,label=par)
          plt.legend()
          plt.xlabel('Thinned MCMC iterations')
          plt.ylabel('Value')
       plt.tight_layout()
       plt.savefig(self.outdir_all + '/' + self.psr_dir + '_samples_trace_' + \
                   '.png')
       plt.close()

def main():
  """
  The pipeline script
  """

  opts = parse_commandline()

  result_obj = EnterpriseWarpResult(opts)
  result_obj.main_pipeline()

if __name__=='__main__':
  main()
