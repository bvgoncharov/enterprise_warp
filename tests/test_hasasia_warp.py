import unittest
from unittest import mock
from types import SimpleNamespace
import sys

import numpy as np

from enterprise_warp import hasasia_warp


class HasasiaWarpTestCase(unittest.TestCase):

  def test_hasasia_parser_inherits_result_options(self):
    argv = ['ew_hasasia.py', '--result', 'run.dat', '--info', '1',
            '--discovery', '1', '--hasasia', '1', '--hasasia_psr', '9']
    with mock.patch.object(sys, 'argv', argv):
      opts = hasasia_warp.HasasiaParser().parse_args()
    self.assertEqual(opts.result, ['run.dat'])
    self.assertEqual(opts.info, 1)
    self.assertEqual(opts.discovery, 1)
    self.assertEqual(opts.hasasia, 1)
    self.assertEqual(opts.hasasia_psr, '9')
    self.assertEqual(opts.hasasia_average_toas, 0)

  def test_extract_model_nfreqs(self):
    class FakeNoiseModel(object):
      def __init__(self, psr=None, params=None):
        pass

      def option_nfreqs(self, option, sel_func_name=None):
        return option['n_freqs']

    obj = hasasia_warp.HasasiaEnterpriseWarp.__new__(
        hasasia_warp.HasasiaEnterpriseWarp)
    obj.params = SimpleNamespace(noise_model_obj=FakeNoiseModel)
    model_params = SimpleNamespace(
        common_signals={'common_gp': {'psd': 'astro_free_spectrum_cw',
                                      'n_freqs': 15}},
        noisemodel={'J1234+5678': {'spin_noise': {
            'psd': 'regularized_powerlaw', 'n_freqs': 40}}},
        to_remaining_psrs={'spin_noise': {'psd': 'regularized_powerlaw',
                                          'n_freqs': 30}},
        to_each_psr={},
    )
    self.assertEqual(obj._common_nfreq(model_params), 15)
    self.assertEqual(obj._red_nfreq(model_params,
                                    SimpleNamespace(name='J1234+5678')), 40)
    self.assertEqual(obj._red_nfreq(model_params,
                                    SimpleNamespace(name='J0000+0000')), 30)

  def test_extract_legacy_string_nfreqs(self):
    obj = hasasia_warp.HasasiaEnterpriseWarp.__new__(
        hasasia_warp.HasasiaEnterpriseWarp)
    model_params = SimpleNamespace(
        common_signals={'gwb': 'fixed_gamma_20_nfreqs'},
        noisemodel={},
        to_remaining_psrs={'spin_noise': 'powerlaw_30_nfreqs'},
        to_each_psr={},
    )
    self.assertEqual(obj._common_nfreq(model_params), 20)
    self.assertEqual(obj._red_nfreq(model_params,
                                    SimpleNamespace(name='J1')), 30)

  def test_posterior_mode_uses_histogram_center(self):
    samples = np.concatenate([np.linspace(0.0, 0.1, 20),
                              np.linspace(1.0, 1.1, 100)])
    mode = hasasia_warp.posterior_mode(samples, bins=10)
    self.assertGreater(mode, 0.9)
    self.assertLess(mode, 1.2)

  def test_log10rho_to_psd(self):
    psd = hasasia_warp.log10rho_psd(np.asarray([-8.0, -7.0]), 100.0)
    self.assertTrue(np.allclose(psd, np.asarray([1e-14, 1e-12])))

  def test_find_log10rho_modes_bracket_and_underscore(self):
    obj = hasasia_warp.HasasiaEnterpriseWarp.__new__(
        hasasia_warp.HasasiaEnterpriseWarp)
    obj.pars = np.asarray(['crn_log10_rho[0]', 'crn_log10_rho_1'])
    obj.chain_burn = np.asarray([[-8.0, -7.0],
                                 [-8.1, -7.1],
                                 [-8.0, -7.0],
                                 [-9.0, -6.0]])
    prefix, values = obj._rho_modes(2)
    self.assertEqual(prefix, 'crn')
    self.assertEqual(values.shape, (2,))

  def test_white_noise_covariance_with_ecorr(self):
    psr = SimpleNamespace(
        name='J0000+0000',
        toas=np.asarray([0.0, 10.0, 2 * 86400.0]),
        toaerrs=np.asarray([1.0, 2.0, 3.0]),
        flags={'f': np.asarray(['be1', 'be1', 'be2'])},
    )
    noise = {
        'J0000+0000_be1_efac': 2.0,
        'J0000+0000_be1_log10_t2equad': 0.0,
        'J0000+0000_be1_log10_ecorr': 0.0,
        'J0000+0000_be2_efac': 1.0,
    }

    class HsenStub(object):
      @staticmethod
      def quantize_fast(toas, toaerrs, flags=None, dt=1):
        return None, None, None, None, [[0, 1], [2]]

    corr = hasasia_warp.build_white_noise_covariance(psr, noise, hsen=HsenStub())
    self.assertTrue(np.allclose(np.diag(corr), np.asarray([6.0, 18.0, 9.0])))
    self.assertEqual(corr[0, 1], 1.0)
    self.assertEqual(corr[1, 0], 1.0)
    self.assertEqual(corr[0, 2], 0.0)


if __name__ == '__main__':
  unittest.main()
