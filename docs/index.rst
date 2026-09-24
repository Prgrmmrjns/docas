DOCAS
=====

Dose–response curve alignment via synthetic augmentation.

DOCAS aligns the treatment response of any ``fit`` / ``predict`` regressor with a
stated target dose–response curve (DRC). The model, its features, and its loss
stay unchanged. DOCAS only adds synthetic training rows.

A model trained on observational data can predict well and still respond to the
treatment with the wrong sign. Ceteris paribus plots (ICE curves and partial
dependence plots) show this. The alignment error quantifies it: the RMSE between
the centred ICE curves and the target DRC.

The target DRC is a stated reference, for example a population curve, and not an
individual ground truth. The response is only learned within the range covered
by the synthetic rows.

Install
-------

.. code-block:: bash

   pip install docas

Quick start
-----------

.. code-block:: python

   import numpy as np
   from sklearn.ensemble import HistGradientBoostingRegressor

   from docas import Aligner, audit_curve

   def train(X, y, sample_weight=None):
       model = HistGradientBoostingRegressor(max_depth=3, random_state=0)
       model.fit(X, y, sample_weight=sample_weight)
       return model

   def target(u):
       return -20.0 * np.asarray(u, float)

   result = Aligner(train, treatment_idx=1, target=target, span=1.0, synth_ratio=2.0).fit(X, y)
   print(result.alignment_error)

``span`` is the physical dose at fractional dose ``u = 1``. Half of the probe
doses are drawn from the doses observed in training (``emp_frac``); the rest are
drawn uniformly between zero and that maximum. Each synthetic label keeps the
observed outcome and adds the target DRC difference between the probe dose and
the observed dose.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api
