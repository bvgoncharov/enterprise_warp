import unittest
from unittest import mock
from types import SimpleNamespace
import sys
import os
import json
import tempfile

import numpy as np

from enterprise_warp import hasasia_warp


class HasasiaWarpTestCase(unittest.TestCase):

  def test_hasasia_parser_inherits_result_options(self):
    argv = ['ew_hasasia.py', '--result', 'run.dat', '--info', '1',
            '--discovery', '1', '--hasasia', '1', '--hasasia_psr', '9',
            '--snr', '3.0',
            '--pta', 'cw', '--hasasia_directional_theta', '1.2',
            '--hasasia_directional_phi', '2.3',
            '--hasasia_directional_freq', '4e-8']
    with mock.patch.object(sys, 'argv', argv):
      opts = hasasia_warp.HasasiaParser().parse_args()
    self.assertEqual(opts.result, ['run.dat'])
    self.assertEqual(opts.info, 1)
    self.assertEqual(opts.discovery, 1)
    self.assertEqual(opts.hasasia, 1)
    self.assertEqual(opts.hasasia_psr, '9')
    self.assertEqual(opts.snr, 3.0)
    self.assertEqual(opts.pta, 'cw')
    self.assertEqual(opts.hasasia_average_toas, 0)
    self.assertAlmostEqual(opts.hasasia_directional_theta, 1.2)
    self.assertAlmostEqual(opts.hasasia_directional_phi, 2.3)
    self.assertAlmostEqual(opts.hasasia_directional_freq, 4e-8)

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

  def test_white_noise_diagnostics_from_quantized_epochs(self):
    psr = SimpleNamespace(
        name='J0000+0000',
        toas=np.asarray([0.0, 10.0, 2 * 86400.0, 2 * 86400.0 + 10.0]),
        toaerrs=np.asarray([1.0, 1.0, 1.0, 1.0]),
        residuals=np.asarray([1.0, 3.0, 5.0, 7.0]),
        flags={'f': np.asarray(['be1', 'be1', 'be1', 'be1'])},
    )
    noise = {
        'J0000+0000_be1_efac': 1.0,
        'J0000+0000_be1_log10_ecorr': 0.0,
    }

    class HsenStub(object):
      @staticmethod
      def quantize_fast(toas, toaerrs, flags=None, dt=1):
        return None, None, None, None, [[0, 1], [2, 3]]

    diagnostics = hasasia_warp.build_white_noise_diagnostics(
        psr, noise, freqs=np.asarray([1.0e-8, 2.0e-8]), hsen=HsenStub())
    self.assertAlmostEqual(diagnostics['delta_t_eff'], 2.0 * 86400.0)
    self.assertAlmostEqual(diagnostics['tspan'], 2.0 * 86400.0)
    self.assertAlmostEqual(diagnostics['mean_sigma_epoch_sqr'], 1.5)
    self.assertAlmostEqual(diagnostics['white_psd'], 259200.0)
    self.assertAlmostEqual(diagnostics['white_arith_psd'], 518400.0)
    self.assertAlmostEqual(diagnostics['wrms_s'], 2.0)
    self.assertAlmostEqual(diagnostics['wrms_psd'], 1382400.0)

  def test_white_noise_diagnostics_without_residuals_omit_wrms_curve(self):
    psr = SimpleNamespace(
        name='J0000+0000',
        toas=np.asarray([0.0, 10.0, 2 * 86400.0, 2 * 86400.0 + 10.0]),
        toaerrs=np.asarray([1.0, 1.0, 1.0, 1.0]),
        flags={'f': np.asarray(['be1', 'be1', 'be1', 'be1'])},
    )

    class HsenStub(object):
      @staticmethod
      def quantize_fast(toas, toaerrs, flags=None, dt=1):
        return None, None, None, None, [[0, 1], [2, 3]]

    diagnostics = hasasia_warp.build_white_noise_diagnostics(
        psr, {}, freqs=np.asarray([1.0e-8, 2.0e-8]), hsen=HsenStub())
    self.assertIsNotNone(diagnostics['white_hc'])
    self.assertIsNone(diagnostics['wrms_s'])
    self.assertIsNone(diagnostics['wrms_hc'])

  def test_residual_psd_to_hc_conversion(self):
    freqs = np.asarray([1.0e-8, 2.0e-8])
    psd = np.asarray([3.0, 4.0])
    hc = hasasia_warp._residual_psd_to_hc(freqs, psd)
    expected = np.sqrt(12.0 * np.pi**2 * freqs**3 * psd)
    self.assertTrue(np.allclose(hc, expected))

  def test_transmission_function_from_tm_basis_matches_full_g_basis(self):
    toas = np.asarray([0.0, 1.0, 3.0, 6.0])
    designmatrix = np.column_stack([
        np.ones_like(toas),
        toas,
    ])
    freqs = np.asarray([0.1, 0.25])
    tf_fast = hasasia_warp._transmission_function_from_tm_basis(
        designmatrix, toas, freqs, chunk_size=1)
    u_full = np.linalg.svd(designmatrix, full_matrices=True)[0]
    g_basis = u_full[:, designmatrix.shape[1]:]
    phases = np.exp(1j * 2.0 * np.pi * freqs[:, None] * toas[None, :])
    tf_direct = np.real(np.sum(
        np.abs(np.matmul(phases, g_basis))**2, axis=1) / float(toas.size))
    self.assertTrue(np.allclose(tf_fast, tf_direct))

  def test_astro_common_powerlaw_modes_derive_log10A_from_h2c(self):
    obj = hasasia_warp.HasasiaEnterpriseWarp.__new__(
        hasasia_warp.HasasiaEnterpriseWarp)
    obj.pars = np.asarray(['crn_log10_h2c'])
    obj.chain_burn = np.asarray([[-30.0], [-30.0], [-29.5], [-31.0]])
    prefix, amp, gamma, source = obj._astro_common_powerlaw_modes()
    self.assertEqual(prefix, 'crn')
    self.assertAlmostEqual(amp, -15.0, places=2)
    self.assertAlmostEqual(gamma, 13.0 / 3.0)
    self.assertIn('log10_h2c', source)

  def test_projected_rrf_spectrum_keeps_geometry_metadata(self):
    class HsenStub(object):
      @staticmethod
      def resid_response(freqs):
        return np.ones_like(freqs)

    spectrum = hasasia_warp.ProjectedRRFSpectrum(
        name='J0000+0000',
        toas=np.asarray([1.0, 2.0]),
        toaerrs=np.asarray([0.1, 0.2]),
        freqs=np.asarray([1.0e-8, 2.0e-8]),
        ncalinv=np.asarray([3.0, 4.0]),
        hsen=HsenStub(),
        phi=1.25,
        theta=2.5,
        pdist=(1.0, 0.1))
    self.assertEqual(spectrum.phi, 1.25)
    self.assertEqual(spectrum.theta, 2.5)
    self.assertEqual(spectrum.pdist, (1.0, 0.1))

  def test_pta_snr_scaled_curves(self):
    sensitivity = SimpleNamespace(
        S_eff=np.asarray([4.0, 9.0]),
        h_c=np.asarray([2.0, 3.0]),
    )
    gwb_seff, gwb_hc = hasasia_warp._pta_snr_scaled_curves(
        'gwb', sensitivity, 4.0)
    self.assertTrue(np.allclose(gwb_seff, np.asarray([16.0, 36.0])))
    self.assertTrue(np.allclose(gwb_hc, np.asarray([4.0, 6.0])))

    cw_seff, cw_hc = hasasia_warp._pta_snr_scaled_curves(
        'cw', sensitivity, 4.0)
    self.assertTrue(np.allclose(cw_seff, np.asarray([64.0, 144.0])))
    self.assertTrue(np.allclose(cw_hc, np.asarray([8.0, 12.0])))

  def test_find_latest_compatible_checkpoint_uses_matching_settings(self):
    obj = hasasia_warp.HasasiaEnterpriseWarp.__new__(
        hasasia_warp.HasasiaEnterpriseWarp)
    obj.opts = SimpleNamespace(hasasia_spectrum='rrf_projected',
                               hasasia_average_toas=0)
    with tempfile.TemporaryDirectory() as tmpdir:
      obj.hasasia_result_dir = tmpdir
      hasasia_dir = os.path.join(tmpdir, 'hasasia')
      older = os.path.join(hasasia_dir, '20260101_000000_J0000+0000')
      newer = os.path.join(hasasia_dir, '20260102_000000_J0000+0000')
      os.makedirs(older)
      os.makedirs(newer)
      for path in [older, newer]:
        open(os.path.join(path, 'pulsar.pkl'), 'wb').close()
        open(os.path.join(path, 'spectrum.pkl'), 'wb').close()
      with open(os.path.join(older, 'settings.json'), 'w') as fout:
        json.dump({
            'selected_pulsar': 'J0000+0000',
            'spectrum': 'rrf_projected',
            'curve_nf': 2,
            'curve_fmin': 1.0,
            'curve_fmax': 2.0,
            'average_toas': 0,
        }, fout)
      with open(os.path.join(newer, 'settings.json'), 'w') as fout:
        json.dump({
            'selected_pulsar': 'J0000+0000',
            'spectrum': 'spectrum',
            'curve_nf': 2,
            'curve_fmin': 1.0,
            'curve_fmax': 2.0,
            'average_toas': 0,
        }, fout)
      checkpoint = obj._find_latest_compatible_checkpoint(
          'J0000+0000', np.asarray([1.0, 2.0]))
    self.assertEqual(checkpoint, older)

  def test_directional_selection_uses_nearest_pixel_and_requested_frequency(self):
    obj = hasasia_warp.HasasiaEnterpriseWarp.__new__(
        hasasia_warp.HasasiaEnterpriseWarp)
    obj.opts = SimpleNamespace(hasasia_directional_theta=1.01,
                               hasasia_directional_phi=0.02,
                               hasasia_directional_freq=1.9e-8)
    sensitivity = SimpleNamespace(
        theta_gw=np.asarray([0.1, 1.0, 2.2]),
        phi_gw=np.asarray([0.0, 0.0, 0.0]),
        freqs=np.asarray([1.0e-8, 2.0e-8, 3.0e-8]),
        S_eff_mean=np.asarray([3.0, 2.0, 1.0]),
        fidx=lambda freq: np.asarray([np.argmin(np.abs(
            np.asarray([1.0e-8, 2.0e-8, 3.0e-8]) - freq))]),
    )
    sky_idx, freq_idx = obj._directional_selection(sensitivity)
    self.assertEqual(sky_idx, 1)
    self.assertEqual(freq_idx, 1)


if __name__ == '__main__':
  unittest.main()
