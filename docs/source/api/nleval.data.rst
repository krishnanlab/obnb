obnb.data
===========

.. currentmodule:: obnb.data

.. autosummary::
   :nosignatures:

   {% for cls in obnb.data.classes %}
     {{ cls }}
   {% endfor %}

Base data objects
-----------------

.. autoclass:: obnb.data.base.BaseData
   :members:

.. autoclass:: obnb.data.network.base.BaseNDExData
   :members:

.. autoclass:: obnb.data.annotated_ontology.base.BaseAnnotatedOntologyData
   :members:

Data objects
------------

.. automodule:: obnb.data
   :members:
   :exclude-members: download, process

Experimental features
---------------------

.. automodule:: obnb.data.experimental
   :members:
