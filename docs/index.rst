.. IS-P2-01 documentation master file
   ==================================

Welcome to the IS-P2-01 documentation!
=====================================

Contents:

.. toctree::
   :maxdepth: 2

   modules


API Reference
=============

Automatic API documentation for the Python modules in the `src` package. If the
modules and their classes/methods include docstrings, they will be extracted
and rendered here by Sphinx/autodoc.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   app
   api_routes_predictions
   data_processing
   entrenar_ensemble_completo
   ensemble_model
   finetuning
   guardar_modelo_completo
   incidence_detector
   main
   model.registro_incidencias
   model.sysGestion
   model.usuario
   predict_realtime
   routes.api_routes
   routes.web_routes
   train_ensemble
   training_template
   usar_ensemble
   usomodelo
   socket_events

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
