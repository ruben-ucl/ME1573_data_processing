*:･ﾟ✧ Scripts for working with greyscale radiography data in HDF5 format ✧･ﾟ:*


Repository structure
--------------------

✧ ME1573_data_processing (root - run scripts from here)
	|
	|---- ✧	dirs
	|	text files containing file directories referenced by other scripts in repo
	|
	|---- ✧	file
	|     	scripts for file management and managing formats
	|
	|---- ✧	prep
	|	scripts for pre-processing and preparing images for segmentation and visualisation
	|
	|---- ✧	segm
	|	scripts for binary segmentation using thresholding and contected component analysis
	|
	|---- ✧	meas
	|	scripts for measuring image features
	|
	|---- ✧	vis
	|	scripts for data visualisation
	|
	|---- ✧	template
	|	templates for HDF5 datset operations and connected component analysis and other examples 
	|
	|---- ✧	old
	|	deprecated scripts
	|
	|
	|
	|---- ✧	tools.py
		library of functions referenced by other scripts in the repo

