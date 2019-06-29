.. enterprise_warp documentation master file, created by
   sphinx-quickstart on Thu Jun 27 22:53:00 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to enterprise_warp's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   paramfile

enterprise_warp
===============

enterprise_warp includes a wrapper and a set of tools for `Nanograv's Enterprise<https://github.com/nanograv/enterprise/>`__.

Look how easy it is to run from the command line: `python run_example_paramfile.py --prfile example_paramfile.dat --num 0`

Features
--------

- Wrapper for Enterprise that allows to run it from a configuration file. Custom models can be included too.
  + Easy to paralellize separate pulsar analysis on a supercomputer
  + Use the same configuration file to keep track of your input and output directories, plot results
- Wrap Enterprise likelihood and priors in `Bilby<https://git.ligo.org/lscsoft/bilby>`
  + use one interfance for multiple samplers: Nestle, Dynesty, Emcee, etc.
  + pre-defined priors, including an option with periodic boundaries
  + hyper PE
  + easy access to a variety of Bayesian data analysis tools: P-P plots, prior volume, Occam factor, covariance matrix, etc.
  + see Bilby's website for more options

Installation
------------

Install by:

1. git clone https://github.com/bvgoncharov/enterprise_warp.git
2. cd enterprise_warp
3. python setup.py install --user

License
-------

The project is licensed under the MIT license.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
