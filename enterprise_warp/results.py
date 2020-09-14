import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import re
import optparse
import itertools
import numpy as np

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

  parser.add_option("-c", "--corner", help="Plot corner (1/0)", \
                    default=0, type=int)

  parser.add_option("-p", "--par", help="Include only specific model \
                    parameters for a corner plot (more than one could and \
                    should be added)", action="append", default=None, type=str)

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
    nb, bins, patches = plt.hist(chain[:,ct], bins=50)
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
  return bool(re.match(r'^\d{1,}_[J,B]\d{2,4}[+,-]\d{4,4}$', folder_name))


def main():
  """
  The pipeline script
  """

  opts = parse_commandline()

  # Determine output directory from the --results argument
  if os.path.isdir(opts.result):
    outdir_all = opts.result
  elif os.path.isfile(opts.result):
    params = enterprise_warp.Params(opts.result, init_pulsars=False)
    outdir_all = params.out + params.label_models + '_' + \
                 params.paramfile_label + '/'
  else:
    raise ValueError('--result seems to be neither a file, not a directory')

  # Check if we need to loop over pulsar directories, or not
  out_subdirs = np.array(os.listdir(outdir_all))
  psr_dir_mask = [check_if_psr_dir(dd) for dd in out_subdirs]
  psr_dirs = out_subdirs[psr_dir_mask]
  if psr_dirs.size == 0:
    psr_dirs = np.array([''])

  for psr_dir in psr_dirs:

    outdir = outdir_all + '/' + psr_dir + '/'
    if opts.name is not 'all' and opts.name not in psr_dir:
      continue
    print('Processing ', psr_dir)

    if os.path.isfile(outdir + '/chain_1.0.txt'):
      chain_file = outdir + '/chain_1.0.txt'
    elif os.path.isfile(outdir + '/chain_1.txt'):
      chain_file = outdir + '/chain_1.txt'
    else:
      chain_file = None
      print('Could not find chain file in ',outdir)

    if opts.info:
      if chain_file is not None:
        print('Available chain file ', chain_file, '(', 
              int(np.round(os.path.getsize(chain_file)/1e6)), ' Mb)')
      else:
        print('Could not find chain file in ', outdir)

    pars = np.loadtxt(outdir + '/pars.txt', dtype=np.unicode_)
    if opts.info and opts.name != 'all':
      print('Parameter names:')
      for par in pars:
        print(par)

    if not (opts.noisefiles or opts.logbf or opts.corner or opts.chains):
      continue

    # Loading PTMCMC chains
    chain = np.loadtxt(chain_file)
    burn = int(0.25*chain.shape[0])
    chain_burn = chain[burn:,:]

    ind_model = list(pars).index('nmodel')
    unique, counts = np.unique(np.round(chain_burn[:, ind_model]),
                               return_counts=True)
    dict_real_counts = dict(zip(unique.astype(int), counts.astype(float)))

    # Noise files part
    if opts.noisefiles:
      make_noise_files(psr_dir, chain_burn, pars,
                       outdir = outdir_all + '/noisefiles/')

    # Bayes Factors
    if opts.logbf:
      print('=====', psr_dir, ' model selection results', '=====')
      print('Samples in favor of models: ', dict_real_counts)
      if len(unique) > 1:
        count_by_pairs = list(itertools.combinations(sorted(unique), 2))
        for combination in count_by_pairs:
          logbf = np.log(dict_real_counts[combination[1]] / \
                         dict_real_counts[combination[0]])
          print('logBF for ', int(combination[1]), 'over ', \
                int(combination[0]),': ', logbf)

    # Corner plots
    if opts.corner:
      for jj in unique:
        model_mask = np.round(chain_burn[:,ind_model]) == jj
        figure = corner(chain_burn[model_mask,:-4], 30, labels=pars)
        plt.savefig(outdir_all + '/' + psr_name + '_' + 'corner_' + str(jj) + \
                    '.png')
        plt.close()

    # MCMC chain plots (evolution in time)
    if opts.chains:
       thin_factor = 200
       plt.figure(figsize=[6.4,4.8*len(pars)])
       for pp, par in enumerate(pars):
          plt.subplot(len(pars), 1, pp+1)
          cut_chain = chain[::int(chain[:,pp].size/thin_factor),pp]
          plt.plot(cut_chain,label=par)
          plt.legend()
          plt.xlabel('Thinned MCMC iterations')
          plt.ylabel('Value')
       plt.tight_layout()
       plt.savefig(outdir_all + '/' + psr_name + '_mcmc_samples_trace_' + \
                   '.png')
       plt.close()


if __name__=='__main__':
  main()
