nleval.data
===========

.. currentmodule:: nleval.data

.. autosummary::
   :nosignatures:

   {% for cls in nleval.data.classes %}
     {{ cls }}
   {% endfor %}

Base data objects
-----------------

.. autoclass:: nleval.data.base.BaseData
   :members:

.. autoclass:: nleval.data.network.base.BaseNDExData
   :members:

.. autoclass:: nleval.data.annotated_ontology.base.BaseAnnotatedOntologyData
   :members:

Data objects
------------

.. automodule:: nleval.data
   :members:
   :exclude-members: download, process

Experimental features
---------------------

.. automodule:: nleval.data.experimental
   :members:
