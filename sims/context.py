import os
import sys

# Add dmsrc to path
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
grandparent_directory = os.path.split(parent_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory))
import mosaic_paper_src
sys.path.insert(
	0, 
	os.path.join(os.path.abspath(grandparent_directory), "mosaicperm/")
)
import mosaicperm