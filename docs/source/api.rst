API
===

DeepFinder
----------

Each step of the DeepFinder workflow is coded as a class. The parameters of each method are stored as class attributes
and are given default values in the constructor. These parameters can easily be given custom values as follows::
    from deepfinder.training import Train
    trainer = Train(Ncl=5, dim_in=56) # initialize training task, where default batch_size=25
    trainer.batch_size = 16 # customize batch_size value

Each class has a main method called 'launch' to execute the procedure. These classes all inherit from a mother class
'DeepFinder' that possesses features useful for communicating with the GUI.


Training
++++++++
.. autoclass:: deepfinder.training.TargetBuilder
   :members: 
.. autoclass:: deepfinder.training.Train
   :members:
   
Inference
+++++++++
.. autoclass:: deepfinder.inference.Segment
   :members:
.. autoclass:: deepfinder.inference.Cluster
   :members:

   
Utilities
---------
Common utils
++++++++++++
.. automodule:: deepfinder.utils.common
   :members:
Object list utils
+++++++++++++++++
.. automodule:: deepfinder.utils.objl
   :members:
Scoremap utils
++++++++++++++
.. automodule:: deepfinder.utils.smap
   :members: