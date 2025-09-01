import os
import sys

# Add src to path
file_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.split(file_directory)[0]
grandparent_directory = os.path.split(root_directory)[0]
sys.path.insert(
	0, 
	os.path.join(os.path.abspath(grandparent_directory), "mosaicperm/")
)
import mosaicperm
sys.path.insert(0, os.path.abspath(root_directory))
import mosaic_paper_src