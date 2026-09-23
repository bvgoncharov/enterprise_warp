import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from enterprise_warp import gp_reconstruction


class GPReconstructionTestCase(unittest.TestCase):

  def test_parser_inherits_result_options(self):
    argv = ['ew_gp_reconstruction.py', '--result', 'run.dat',
            '--discovery', '1', '--num', '3', '--gp_model', '2',
            '--gp_terms', 'red_noise,crn', '--gp_psr', 'J1713',
            '--gp_grid_points', '128', '--gp_plot_toas', '0',
            '--gp_outdir', '/tmp/gp']
    with mock.patch.object(sys, 'argv', argv):
      opts = gp_reconstruction.GPReconstructionParser().parse_args()
    self.assertEqual(opts.result, ['run.dat'])
    self.assertEqual(opts.discovery, 1)
    self.assertEqual(opts.num, 3)
    self.assertEqual(opts.realization, 3)
    self.assertEqual(opts.gp_model, 2)
    self.assertEqual(opts.gp_terms, 'red_noise,crn')
    self.assertEqual(opts.gp_psr, 'J1713')
    self.assertEqual(opts.gp_grid_points, 128)
    self.assertEqual(opts.gp_plot_toas, 0)
    self.assertEqual(opts.gp_outdir, '/tmp/gp')

  def test_detect_gp_terms(self):
    psrs = [SimpleNamespace(name='J1'), SimpleNamespace(name='J2')]
    model_params = SimpleNamespace(
        common_signals={
            'common_gp': {'psd': 'astro_free_spectrum_cw', 'n_freqs': 15},
            'global_gp': {'psd': 'powerlaw', 'orf': 'hd_orf',
                          'n_freqs': 10},
        },
        noisemodel={'J1': {'spin_noise': {'psd': 'regularized_powerlaw',
                                          'n_freqs': 30}}},
        to_remaining_psrs={},
        to_each_psr={},
    )
    inventory = gp_reconstruction.detect_gp_terms(model_params, psrs)
    terms = sorted(item['term'] for item in inventory)
    self.assertEqual(terms, ['crn', 'gw', 'red_noise'])
    self.assertEqual([item['psr_name'] for item in inventory
                      if item['term'] == 'red_noise'], ['J1'])

  def test_chain_provider_mode_and_vector_fallbacks(self):
    provider = gp_reconstruction.ChainParameterProvider(
        pars=np.asarray(['crn_log10_rho[0]', 'crn_log10_rho[1]',
                         'J1_red_noise_log10_A']),
        chain_burn=np.asarray([[-8.0, -7.0, -15.0],
                               [-8.0, -7.0, -14.9],
                               [-9.0, -6.5, -14.8],
                               [-8.0, -7.0, -14.7]]),
        noisedict={'J1_backend_efac': 1.2},
        defaults={'crn_gamma': 13.0 / 3.0})
    params = provider.parameter_dict(
        ['crn_log10_rho(2)', 'J1_red_noise_log10_A', 'crn_gamma',
         'J1_backend_efac'])
    self.assertEqual(params['crn_log10_rho'].shape, (2,))
    self.assertAlmostEqual(params['crn_gamma'], 13.0 / 3.0)
    self.assertAlmostEqual(params['J1_backend_efac'], 1.2)

  def test_solve_linear_gp_system_recovers_simple_coefficients(self):
    class WhiteModel(object):
      def apply_inverse(self, values):
        return np.asarray(values, dtype=float)

    basis = np.asarray([[1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0]])
    true_coeff = np.asarray([0.4, -0.2])
    residuals = basis.dot(true_coeff)

    class Phi(object):
      params = []

      @staticmethod
      def make_inv():
        def inv(params):
          return np.eye(2) / 1.0e6, 0.0
        inv.params = []
        return inv

    component = SimpleNamespace(Phi=Phi())
    desc = gp_reconstruction.GPDescriptor(
        term='red_noise', psr_name='J1', psr_index=0, basis=basis,
        prior_group='red_noise_J1', prior_component=component)
    mean, cov = gp_reconstruction.solve_linear_gp_system(
        [desc], {0: residuals}, {0: WhiteModel()}, {})
    self.assertTrue(np.allclose(mean, true_coeff, atol=1e-5))
    self.assertEqual(cov.shape, (2, 2))

  def test_basis_from_tspan_matches_discovery_order(self):
    toas = np.asarray([0.0, 0.25, 0.5])
    basis = gp_reconstruction._basis_from_tspan(toas, 4, 1.0)
    expected = np.column_stack([
        np.sin(2.0 * np.pi * toas),
        np.cos(2.0 * np.pi * toas),
        np.sin(4.0 * np.pi * toas),
        np.cos(4.0 * np.pi * toas),
    ])
    self.assertTrue(np.allclose(basis, expected))


if __name__ == '__main__':
  unittest.main()
