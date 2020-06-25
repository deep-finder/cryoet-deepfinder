.. _cmd_line_tools:

Command line tools
==================

This page gives instructions on how to launch DeepFinder steps from the terminal. 

Set up
------
First, add DeepFinder to your path with following command: :code:`export PATH="/path/to/deep-finder/bin:$PATH"`

You can add this command to your :code:`~/.bash_profile` 

.. note::
   Running these commands without any argument will launch the graphical user interface.

Annotation
----------

Usage::

	annotate -t /path/to/tomogram.mrc
		 -o /path/to/output/object_list.xml

Target generation
-----------------
For :code:`generate_target` and :code:`train`, it is not possible to pass all necessary parameters as terminal arguments. Therefore, they have to be passed as an xml file.

Usage::

	generate_target -p /path/to/parameters.xml
	
Parameter file::

	<paramsGenerateTarget>
	  <path_objl path="/path/to/objl.xml"/>
	  <path_initial_vol path=""/>
	  <tomo_size>
	    <X size="400"/>
	    <Y size="400"/>
	    <Z size="200"/>
	  </tomo_size>
	  <strategy strategy="spheres"/>
	  <radius_list>
	    <class1 radius="1"/>
	    <class2 radius="2"/>
	    <class3 radius="3"/>
	  </radius_list>
	  <path_mask_list>
	    <class1 path=""/>
	  </path_mask_list>
	  <path_target path="/path/to/target.mrc"/>
	</paramsGenerateTarget>
	
.. note::
   You can find classes with methods for automatically reading and writing these parameter files in utils/params.py
	
	

Training
--------
Usage::

	 train -p /path/to/parameters.xml
	 
Parameter file::

	<paramsTrain>
	  <path_out path="./"/>
	  <path_tomo>
	    <tomo0 path="/path/to/tomo0.mrc"/>
	    <tomo1 path="/path/to/tomo1.mrc"/>
	    <tomo2 path="/path/to/tomo2.mrc"/>
	  </path_tomo>
	  <path_target>
	    <target0 path="/path/to/target0.mrc"/>
	    <target1 path="/path/to/target1.mrc"/>
	    <target2 path="/path/to/target2.mrc"/>
	  </path_target>
	  <path_objl_train path="/path/to/objl_train.xml"/>
	  <path_objl_valid path="/path/to/objl_valid.xml"/>
	  <number_of_classes n="3"/>
	  <patch_size n="48"/>
	  <batch_size n="20"/>
	  <number_of_epochs n="100"/>
	  <steps_per_epoch n="100"/>
	  <steps_per_validation n="10"/>
	  <flag_direct_read flag="False"/>
	  <flag_bootstrap flag="True"/>
	  <random_shift shift="13"/>
	</paramsTrain>
	

Segmentation
------------
Usage::

	segment -t /path/tomogram.mrc 
	        -w /path/net_weights.h5 
		-c NCLASS 
		-p PSIZE 
		-o /path/output/segmentation.mrc
		
With NCLASS and PSIZE integer values. See :ref:`guide` for parameter description.

Clustering
----------
Usage::

	cluster -l /path/to/segmentation.mrc 
	        -r clusterRadius 
		-o /path/to/output/object_list.xml
