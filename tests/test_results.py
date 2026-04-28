import os
import sys
import glob
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from enterprise_warp import results


def make_opts(result, discovery=0, corner=0, hists=0, chains=0, par=None,
              par_equal=None):
  return SimpleNamespace(
      result=result,
      info=0,
      name='all',
      corner=corner,
      par=par,
      par_equal=par_equal,
      truths=None,
      chains=chains,
      hists=hists,
      logbf=0,
      noisefiles=0,
      credlevels=0,
      covm=0,
      separate_earliest=0.0,
      load_separated=0,
      thin=None,
      optimal_statistic=0,
      optimal_statistic_orfs='hd,dipole,monopole',
      optimal_statistic_nsamples=1000,
      load_optimal_statistic_results=0,
      bilby=0,
      discovery=discovery,
      custom_models_py=None,
      custom_models=None,
      realization=0,
  )


class ResultsTestCase(unittest.TestCase):

  def setUp(self):
    self.tmpdir = tempfile.mkdtemp(prefix='ew_results_test_')

  def tearDown(self):
    shutil.rmtree(self.tmpdir)

  def _write_enterprise_result(self, dirname, pars, truth=None):
    outdir = os.path.join(self.tmpdir, dirname)
    os.makedirs(outdir)
    np.savetxt(os.path.join(outdir, 'pars.txt'), np.asarray(pars), fmt='%s')
    samples = np.zeros((40, len(pars) + 4))
    for idx in range(len(pars)):
      samples[:, idx] = np.linspace(idx, idx + 1.0, samples.shape[0])
    np.savetxt(os.path.join(outdir, 'chain_1.txt'), samples)
    if truth is not None:
      with open(os.path.join(outdir, 'truth.json'), 'w') as fout:
        import json
        json.dump(truth, fout)
    return outdir

  def _write_discovery_result(self, dirname, pars, truth=None):
    outdir = os.path.join(self.tmpdir, dirname)
    os.makedirs(outdir)
    np.savetxt(os.path.join(outdir, 'pars.txt'), np.asarray(pars), fmt='%s')
    data = {}
    for idx, par in enumerate(pars):
      data[par] = np.linspace(idx, idx + 1.0, 40)
    pd.DataFrame(data).to_csv(os.path.join(outdir, dirname + '_chain.csv'),
                              index=False)
    if truth is not None:
      with open(os.path.join(outdir, 'truth.json'), 'w') as fout:
        import json
        json.dump(truth, fout)
    return outdir

  def test_parse_commandline_repeated_result(self):
    argv = ['results.py', '--result', 'run_a', '--result', 'run_b', '--corner', '2']
    with mock.patch.object(sys, 'argv', argv):
      opts = results.parse_commandline()
    self.assertEqual(opts.result, ['run_a', 'run_b'])
    self.assertEqual(opts.corner, 2)

  def test_parse_commandline_par_equal(self):
    argv = ['results.py', '--result', 'run_a',
            '--par_equal', 'cw_log10_h0=crn_log10_hcw',
            '--par_equal', 'crn_log10_fcw=cw_log10_f']
    with mock.patch.object(sys, 'argv', argv):
      opts = results.parse_commandline()
    self.assertEqual(opts.par_equal,
                     ['cw_log10_h0=crn_log10_hcw',
                      'crn_log10_fcw=cw_log10_f'])

  def test_main_uses_result_collection_for_discovery_multiresult(self):
    opts = make_opts(['run_a', 'run_b'], discovery=1)
    fake_collection = mock.Mock()
    with mock.patch.object(results, 'parse_commandline', return_value=opts), \
         mock.patch.object(results, 'ResultCollection', return_value=fake_collection) as collection_cls:
      results.main()
    collection_cls.assert_called_once()
    self.assertIs(collection_cls.call_args[0][1], results.DiscoveryWarpResult)
    fake_collection.main_pipeline.assert_called_once_with()

  def test_main_uses_result_collection_for_enterprise_multiresult(self):
    opts = make_opts(['run_a', 'run_b'], discovery=0)
    fake_collection = mock.Mock()
    with mock.patch.object(results, 'parse_commandline', return_value=opts), \
         mock.patch.object(results, 'ResultCollection', return_value=fake_collection) as collection_cls:
      results.main()
    collection_cls.assert_called_once()
    self.assertIs(collection_cls.call_args[0][1], results.EnterpriseWarpResult)
    fake_collection.main_pipeline.assert_called_once_with()

  def test_union_and_filter_parameter_names(self):
    res_a = SimpleNamespace(pars=np.asarray(['a', 'b', 'shared']))
    res_b = SimpleNamespace(pars=np.asarray(['shared', 'c']))
    union_all = results.get_union_parameter_names([res_a, res_b])
    union_filtered = results.get_union_parameter_names([res_a, res_b],
                                                       par_filters=['sh', 'c'])
    self.assertEqual(union_all, ['a', 'b', 'shared', 'c'])
    self.assertEqual(union_filtered, ['shared', 'c'])

  def test_union_parameter_names_with_equal_pars(self):
    res_a = SimpleNamespace(pars=np.asarray(['cw_log10_h0', 'shared']))
    res_b = SimpleNamespace(pars=np.asarray(['crn_log10_hcw', 'shared']))
    equal_map = results.parse_equal_par_specs(['cw_log10_h0=crn_log10_hcw'])
    union_all = results.get_union_parameter_names([res_a, res_b],
                                                  equal_par_map=equal_map)
    self.assertEqual(union_all, ['cw_log10_h0', 'shared'])

  def test_unique_result_labels_with_duplicate_basenames(self):
    labels = results.get_unique_result_labels([
        '/tmp/run_a/6',
        '/tmp/run_b/6',
        '/tmp/run_c/other',
    ])
    self.assertEqual(labels, ['run_a/6', 'run_b/6', 'other'])

  def test_sanitize_filename_component(self):
    safe = results.sanitize_filename_component('run_a/6__vs__model.dat')
    self.assertEqual(safe, 'run_a__6__vs__model.dat')

  def test_equal_par_sample_and_truth_lookup(self):
    opts = make_opts('dummy')
    result_obj = results.EnterpriseWarpResult.__new__(results.EnterpriseWarpResult)
    result_obj.opts = opts
    result_obj.pars = np.asarray(['crn_log10_hcw', 'other'])
    result_obj.chain = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    result_obj.chain_burn = np.asarray([[3.0, 4.0]])
    result_obj.truth_values = {'crn_log10_hcw': -12.3}
    equal_map = results.parse_equal_par_specs(['cw_log10_h0=crn_log10_hcw'])
    samples = result_obj.get_samples_for_equal_par('cw_log10_h0',
                                                   equal_par_map=equal_map,
                                                   burned=False)
    truth_val = result_obj.get_truth_value_for_equal_par('cw_log10_h0',
                                                         equal_par_map=equal_map)
    self.assertTrue(np.allclose(samples, np.asarray([1.0, 3.0])))
    self.assertEqual(truth_val, -12.3)

  def test_common_psr_dir_intersection(self):
    res_a = SimpleNamespace(psr_dirs=np.asarray(['001_J1234+5678', '002_J2345+6789']))
    res_b = SimpleNamespace(psr_dirs=np.asarray(['001_J1234+5678', '003_J3456+7890']))
    res_c = SimpleNamespace(psr_dirs=np.asarray(['001_J1234+5678']))
    common_dirs = results.get_common_psr_dirs([res_a, res_b, res_c])
    self.assertEqual(common_dirs, ['001_J1234+5678'])

  def test_enterprise_result_collection_smoke(self):
    out_a = self._write_enterprise_result('ent_a', ['shared', 'only_a'],
                                          truth={'shared': 0.4})
    out_b = self._write_enterprise_result('ent_b', ['shared', 'only_b'])
    opts = make_opts([out_a, out_b], corner=2, hists=1, chains=1)
    collection = results.ResultCollection(opts, results.EnterpriseWarpResult)
    collection.main_pipeline()
    self.assertTrue(glob.glob(os.path.join(out_a, '*comparison_corner*png')))
    self.assertTrue(glob.glob(os.path.join(out_a, '*comparison_hist_pars*png')))
    self.assertTrue(glob.glob(os.path.join(out_a, '*comparison_samples_trace*png')))

  def test_discovery_result_collection_smoke(self):
    out_a = self._write_discovery_result('disc_a', ['shared', 'only_a'],
                                         truth={'shared': 0.4})
    out_b = self._write_discovery_result('disc_b', ['shared', 'only_b'])
    opts = make_opts([out_a, out_b], discovery=1, corner=2, hists=1, chains=1)
    collection = results.ResultCollection(opts, results.DiscoveryWarpResult)
    collection.main_pipeline()
    self.assertTrue(glob.glob(os.path.join(out_a, '*comparison_corner*png')))
    self.assertTrue(glob.glob(os.path.join(out_a, '*comparison_hist_pars*png')))
    self.assertTrue(glob.glob(os.path.join(out_a, '*comparison_samples_trace*png')))


if __name__ == '__main__':
  unittest.main()
