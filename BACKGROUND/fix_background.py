import os
import numpy as np
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
from jwst.datamodels import ModelContainer
import copy



# update the background reference file

bkg_file = 'jwst_miri_bkg_0001.fits'
new_file = 'jwst_miri_bkg_20260311.fits'
model = datamodels.open(bkg_file)
new_model = model.copy()
new_model.meta.subarray.xsize = 1032
new_model.meta.subarray.ysize = 1024
new_model.meta.subarray.xstart = 1
new_model.meta.subarray.ystart = 1
new_model.save(new_file)

test_model = datamodels.WfssBkgModel(new_file)
test_model.validate()
