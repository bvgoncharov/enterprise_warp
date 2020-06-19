===================
enterprise_warp API
===================

Below is the information about different modules of enterprise_warp.

* :ref:`enterprise_warp`
* :ref:`enterprise_models`
* :ref:`bilby_warp`
* :ref:`tempo2_warp`
* :ref:`libstempo_warp`

.. _enterprise_warp:

enterprise_warp
---------------

This is the main module that allows to load parameter file, noise model files, python noise model functions and possible noise files for fixed white noise parameters into Enterprise.

.. automodule:: enterprise_warp.enterprise_warp
       :members:

.. _enterprise_models:

enterprise_models
-----------------

This module contains basic models for pulsar timing analyses and serves as a base class for custom noise models. Timing model is set up in enterprise_warp.py.

.. automodule:: enterprise_warp.enterprise_models
          :members:

.. _bilby_warp:

bilby_warp
----------

This module contains a wrapper for Bilby (the Bayesian Inference LiBrarY), a software developed in the LIGO collaboration. Bilby provides access to multiple likelihood samplers and other useful tools for data analysis.

.. automodule:: enterprise_warp.bilby_warp
             :members:

.. _tempo2_warp:

tempo2_warp
----------

This module contains a function to run tempo2 in Python, in order to produce maximum-likelihood time-correlated (red) noise realizations.

.. automodule:: enterprise_warp.tempo2_warp
                :members:

.. _libstempo_warp:

libstempo_warp
--------------

This model contains useful tools for pulsar timing array data simulation with libstempo. It extends libstempo simulations to realistic pulsar data with multiple observing backends, based on real pulsar data.

.. automodule:: enterprise_warp.libstempo_warp
                   :members:
