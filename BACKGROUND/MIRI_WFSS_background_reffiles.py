"""
This routine creates the MIRI WFSS background reference file.
The output reference file is in FITS format

"""

import datetime
import numpy as np
import asdf
from asdf.tags.core import Software, HistoryEntry
from astropy import units as u

from jwst.datamodels import WfssBkgModel
from astropy.io import fits
from stdatamodels.jwst.datamodels import dqflags

#-------------------------------------------------------------------------------

def read_data(bkg_file):
    hdu = fits.open(bkg_file)
    primary_hdu = hdu[0]
    data = primary_hdu.data
    hdu.close()
    return data

#-------------------------------------------------------------------------------
def create_dqdef():
    """Create the DQ definition data needed to describe the DQ plane.

    Returns
    -------
    definitions : list
        Bad pixel bit definitions
    """
    definitions = []
    standard_defs = dqflags.pixel
    for bitname in standard_defs:
        bitvalue = standard_defs[bitname]
        if bitvalue != 0:
            bitnumber = np.uint8(np.log(bitvalue)/np.log(2))
        else:
            bitnumber = 0
        newrow = (bitnumber, bitvalue, bitname, '')
        definitions.append(newrow)
    return definitions

def create_miri_wfss_bkg(datafile,
                         filter,
                         outname,
                         author="Andreea Petric",
                         history=None,
                         pedigree="INFLIGHT 2022-05-22 2022-05-22",
                         useafter="2022-01-01T00:00:00"):
    """
    Create the MIRI WFSS background reference file
    Parameters
    ----------
    datafile : str
        The text file containing the bkg data.
    filter : str
        P750L 
    outname : str
        Output name for the reference file.
    author : str
        The name of the author.
    history : str
        A comment about the refrence file to be saved with
        the meta information.
    pedigree : str
        GROUND or INFLIGHT with the dates of the data used to create the
        reference file.
    useafter : str
        Everything on this date or later will use this file (unless a file with
        an older useafter exists).


    Returns
    -------
    writes out the fits reference file
    """

    if not history:
        history = "Created from {0:s}".format(datafile)


    ref = WfssBkgModel()

    author="Andreea Petric"
    description="MIRI WFSS BKG Ref File"
    exp_type='MIR_WFSS'
    pedigree="INFLIGHT 2022-05-22 2024-08-04"
    reftype='BKG'
    title="MIRI Reference File"
    useafter="2022-01-01T00:00:00"

    ref.meta.author = author
    ref.meta.description = description
    ref.meta.telescope = "JWST"
    ref.meta.useafter = useafter
    ref.meta.title = title
    ref.meta.pedigree = pedigree
    ref.meta.reftype = reftype
    
    # 'meta' field:
    ref.meta.input_units = u.micron
    ref.meta.output_units = u.micron

    ref.meta.exposure.type = exp_type
    print(exp_type)
    
    ref.meta.instrument.name = "MIRI"
    ref.meta.instrument.detector="MIRIMAGE"
    ref.meta.instrument.filter = filter

    ref.meta.instrument.band = 'N/A'
    ref.meta.instrument.channel ='N/A'
    ref.meta.subarray.name = 'FULL' 
    ref.meta.subarray.xsize = 1032
    ref.meta.subarray.ysize = 1024
    ref.meta.subarray.xstart = 1
    ref.meta.subarray.ystart = 1
    data = read_data(datafile)
    error = np.zeros_like(data)
    dq = np.zeros_like(data, dtype=int)               
    dq_def = create_dqdef()


    # data fields:
    ref.data = data
    ref.err = error
    ref.dq = dq
    ref.dq_def = dq_def
    # history entries (also updates meta['history'] field)
    entry = HistoryEntry({'description': history,
                          'time': datetime.datetime.utcnow()})
    sdict = Software({'name': 'MIRI_WFSS_background_reffiles_v1.py',
                      'author': author,
                      'version': '0.1.0'})
    entry['software'] = sdict
    ref.meta.history = history
    ref.history.append(entry)
    # writing & validating ref. file
    ref.validate()
    print(ref.meta)
    print(outname) 
    ref.save(outname)


#-------------------------------------------------------------------------------


if __name__ == "__main__":

    datafile  = 'MIRI_wfss_bk_draftJuly23.fits'
    filter = 'P750L'
    outname="MIRI_WFSS_bkg_August2025.fits"
                        
    create_miri_wfss_bkg(datafile,filter, outname)
