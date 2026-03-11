#! /usr/bin/env python

"""Create photom files for the grisms based on Nor's V4c sensitivity files, which were
created using commissioning data

is below still correct? we still need the awkward data_list?

filter = np.array(['GR150C', 'GR150R'])
>>> pupil = np.array(['F140M', 'F200W'])
>>> order = np.array([1, 1], dtype=np.int16)
>>> photf = np.array([1.e-15, 3.e-15], dtype=np.float32)
>>> uncer = np.array([1.e-17, 3.e-17], dtype=np.float32)
>>> nrows = len(filter)
>>> nx = 437
>>> nelem = np.zeros(nrows, dtype=np.int16) + nx
>>> temp_wl = np.linspace(1.0, 5.0, nx, dtype=np.float32).reshape(1, nx)
>>> wave = np.zeros((nrows, nx), np.float32)
>>> wave[:] = temp_wl.copy()
>>> resp = np.ones((nrows, nx), dtype=np.float32)
>>> resp_unc = np.zeros((nrows, nx), dtype=np.float32)
>>> data_list = [(filter[i], pupil[i], order[i], photf[i], uncer[i], nelem[i],
...               wave[i], resp[i], resp_unc[i]) for i in range(nrows)]
>>> data = np.array(data_list,
...                 dtype=[('filter', 'S12'),
...                        ('pupil', 'S15'),
...                        ('order', '<i2'),
...                        ('photmjsr', '<f4'),
...                        ('uncertainty', '<f4'),
...                        ('nelem', '<i2'),
...                        ('wavelength', '<f4', (nx,)),
...                        ('relresponse', '<f4', (nx,)),
...                        ('reluncertainty', '<f4', (nx,))])
>>> output = datamodels.NrcWfssPhotomModel(phot_table=data)
>>> output.save('nircam_photom_0001.fits')

"""
from copy import deepcopy
from glob import glob
from math import atan2, hypot, sin
import os

import astropy.units as u
from synphot import SpectralElement, Observation, SourceSpectrum, units
from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d

from jwst import datamodels
import pysiaf
from stdatamodels import util

DESCRIPTION = 'Initial WFSS flux calibration values from in flight data, with high res curves'

HISTORY = ('Calibration using data from July 2022, obtained by tracing '
           'spectra of sources from PID 01076 obs105-124P330-E for both '
           'trace and absolute flux calibration. Data were processed by the '
           'Pipeline Stage 1 as well as part of Stage 2 (Flat fielding, gain,'
           'and WCS steps). Module A,B and grism R,C in filters F322W2 and '
           'F444W were fully calibrated. Flux calibration was performed using '
           'the STScI Calspec P330-E reference spectrum. Configurations are '
           'such that a range of 0<s<1 covers approximately the entire '
           'bandpass of a given filter/grism combination. For combinations '
           'not using either the F322W2 or F444W, filter thoughtputs of '
           'F322W2 or F444W were backed out and the appropriate filter '
           'throuhgput was then applied. Configuration files for F250M,F277W,'
           'F300M,F335M,F356W,F360M were derived using the configuration of '
           'F322W2. Configuration files for F410M,F430M,F444W,F460M,F480M '
           'were derived using the configuration of F444W. No filter wedge '
           'offsets were applied when recomputing the trace and wavelength for '
           'combinations not using the F322W2 or F444W filters. This reference '
           'file was created from the V6c data in Pirzkals grismconf repository. '
           'This file replaces the previous version, which had a lower resolution '
           'response curve. F250M and F277W entries are now derived from in flight '
           'data. There is insufficient data to derive sensitivity '
           'curves for the GRISMC 2nd order entries. These were copied from the '
           'previous, ground-based photom reference file.'
           )

def convert_sensitivity_to_MJy(dat):
    #convert sensitivity values (DN/sec per FLAM) to MJy

    # Make sure all the sensitivity=0 points are gone
    good = np.where(dat['SENSITIVITY'] != 0)[0]
    if len(good) != len(dat['SENSITIVITY']):
        print('still some zeros getting through')
        raise ValueError

    flam = 1. / dat['SENSITIVITY'] * units.FLAM
    gooderr = np.where(dat['ERROR'] != 0)[0]

    # Watch out for cases where sensitivity is zero but unc is not zero, or vice versa
    if not np.array_equal(good, gooderr):
        tofix = np.where((dat['SENSITIVITY'] != 0) & (dat['ERROR'] == 0))[0]
        for fix in tofix:
            dat['ERROR'][fix] = np.mean(dat['ERROR'][gooderr])

    # Check that everything now agrees
    gooderr = np.where((dat['ERROR'] != 0) & (dat['SENSITIVITY'] != 0))[0]
    if not np.array_equal(good, gooderr):
        for f,e in zip(dat['SENSITIVITY'], dat['ERROR']):
            print(f,e, e/f)
        raise ValueError('Errs and vals have difft zeros')

    #err = 1. / dat['ERROR'][good] * units.FLAM

    # Convert the sensitivity ranges in order to find the photmjsr uncertainty
    sense_max = 1. / (dat['SENSITIVITY'] + dat['ERROR'])

    # Deal with cases where the error is equal to the sensitivity
    # Just add 1 to the sensitivity
    delta = (dat['SENSITIVITY'] - dat['ERROR'])
    zeros = np.where(delta == 0)[0]
    #delta2 = (dat['SENSITIVITY'][good] - dat['ERROR'][good])
    #zeros2 = np.where(delta2 == 0)[0]
    if len(zeros) > 0:
    #    print('Some entries with sensitivity == error')
    #    #print(dat['SENSITIVITY'][good][zeros2])
        print(f'{len(zeros)} points have sensitivity == error. Fixing.')
        dat['ERROR'][zeros] = dat['ERROR'][zeros] - 1
    #    #dat['SENSITIVITY'][good][zeros2] = dat['SENSITIVITY'][good][zeros2] + 1
    #    #print(dat['SENSITIVITY'][good][zeros2])
    #    #full_tmp = deepcopy(dat['SENSITIVITY'])
    #    #tmp = dat['SENSITIVITY'][good][zeros]
    #    #print(tmp)
    #    #tmp += 1
    #    #print(tmp)
    #    #full_tmp[good][zeros] = tmp
    #    #print(full_tmp[good][zeros])
    #    #dat['SENSITIVITY'] = deepcopy(full_tmp)
    #    ##dat['SENSITIVITY'][good][zeros] = tmp
    #    ##dat['SENSITIVITY'][good][zeros] = dat['SENSITIVITY'][good][zeros] * 2.
    #    #print(dat['SENSITIVITY'][good][zeros])

    sense_min = 1. / (dat['SENSITIVITY'][good] - dat['ERROR'][good])

    delta = (dat['SENSITIVITY'][good] - dat['ERROR'][good])
    zeros = np.where(delta == 0)[0]
    #if len(zeros) > 0:
    #    print('Zeros: ')
    #    print(dat['SENSITIVITY'][good][zeros])
    #    print(dat['ERROR'][good][zeros])
    #    raise ValueError


    err = (sense_min - sense_max) / 2. * units.FLAM

    wavelengths = dat['WAVELENGTH'][good] * u.micron
    mjy = units.convert_flux(wavelengths, flam, u.MJy)
    errmjy = units.convert_flux(wavelengths, err, u.MJy)

    #OR MANUALLY:
    #jy = 3.33564095e4 * flam * wavelengths.to(u.angstrom)**2
    return mjy, wavelengths, errmjy


def get_pixel_area(siaf_inst):
    """Use SIAF to get nominal pixel area at reference location"""
    xscale = hypot(siaf_inst.Sci2IdlX10, siaf_inst.Sci2IdlY10)
    yscale = hypot(siaf_inst.Sci2IdlX11, siaf_inst.Sci2IdlY11)
    bx = atan2(siaf_inst.Sci2IdlX10, siaf_inst.Sci2IdlY10)
    return xscale * yscale * sin(bx) * u.arcsecond * u.arcsecond


def run():
    #a_mod_sensitivity_files = sorted(glob('/Volumes/wit/nircam/reference_files/specwcs/V6_from_Nor/V6/NIRCam.*.A.*.sensitivity.fits'))
    a_mod_sensitivity_files = sorted(glob('/Volumes/wit/nircam/reference_files/specwcs/V6c/NIRCam.*.A.*.sensitivity.fits'))
    current_reffile = 'jwst_nircam_photom_0094.fits'
    create_reffile(a_mod_sensitivity_files, current_reffile=current_reffile)

    #b_mod_sensitivity_files = sorted(glob('/Volumes/wit/nircam/reference_files/specwcs/V6_from_Nor/V6/NIRCam.*.B.*.sensitivity.fits'))
    b_mod_sensitivity_files = sorted(glob('/Volumes/wit/nircam/reference_files/specwcs/V6c/NIRCam.*.B.*.sensitivity.fits'))
    current_reffile = 'jwst_nircam_photom_0097.fits'
    create_reffile(b_mod_sensitivity_files, current_reffile=current_reffile)


def create_reffile(sensitivity_files, current_reffile=None):

    current_data = None
    if current_reffile is not None:
        current_data = fits.getdata(current_reffile)

    siaf = {'A': pysiaf.Siaf('nircam')['NRCA5_FULL'],
            'B': pysiaf.Siaf('nircam')['NRCB5_FULL']
            }

    pivot = {'F250M': 2.503,
             'F277W': 2.786,
             'F300M': 2.996,
             'F322W2': 3.247,
             'F323N': 3.237,
             'F335M': 3.365,
             'F356W': 3.563,
             'F360M': 3.621,
             'F405N': 4.055,
             'F410M': 4.092,
             'F430M': 4.280,
             'F444W': 4.421,
             'F460M': 4.624,
             'F466N': 4.654,
             'F470N': 4.707,
             'F480M': 4.834}

    nrows = len(sensitivity_files)
    nx = 3000

    filters = np.array([])
    pupils = np.array([])
    orders = np.zeros(nrows, dtype=np.int16)
    photmjsrs = np.zeros(nrows, dtype=np.float32)
    uncertainties = np.zeros(nrows, dtype=np.float32)
    nelems = np.zeros(nrows, dtype=np.int16)
    wavelengths = np.zeros((nrows, nx), np.float32)
    relresponses = np.ones((nrows, nx), dtype=np.float32)
    reluncertainties = np.zeros((nrows, nx), dtype=np.float32)

    # Loop over sensitivity files and build up arrays of needed data
    for i, sensitivity_file in enumerate(sensitivity_files):
        print(sensitivity_file)
        data = fits.getdata(sensitivity_file)

        if 'F322W2.R.A.1st' in sensitivity_file:
            import matplotlib.pyplot as plt
            f,a = plt.subplots()
            a.scatter(data['WAVELENGTH'], data["SENSITIVITY"], color='black')
            #firstbad = np.where((data['WAVELENGTH'] > 4.2532) & (data['WAVELENGTH'] < 4.2533))[0][0]
            #print(firstbad)
            #print(data['WAVELENGTH'][firstbad-6:firstbad+10], data['SENSITIVITY'][firstbad-6:firstbad+10])


        # Remove points where the sensitivity is zero
        good = np.where(data["SENSITIVITY"] > 0.0)[0]
        nozero = {}
        nozero['SENSITIVITY'] = data["SENSITIVITY"][good]
        nozero['WAVELENGTH'] = data['WAVELENGTH'][good]
        nozero['ERROR'] = data['ERROR'][good]
        data = deepcopy(nozero)

        if 'F322W2.R.A.1st' in sensitivity_file:
            a.scatter(data['WAVELENGTH'], data["SENSITIVITY"], color='blue')
            #print(f'After removing zeros, the max wavelength is: {np.max(data["WAVELENGTH"])}')
            #firstbad = np.where((data['WAVELENGTH'] > 4.2532) & (data['WAVELENGTH'] < 4.2533))[0][0]
            #print(data['WAVELENGTH'][firstbad-6:firstbad+10], data['SENSITIVITY'][firstbad-6:firstbad+10])



        print(f'File {sensitivity_file}, length {len(data["SENSITIVITY"])}')
        # If the input data have too many points, we'll need to downsize. The max the
        # reference file can handle is 3000 points.
        if len(data['WAVELENGTH']) > 3000:
            print("interpolating to bring down to 3000 points")
            minwave = np.min(data['WAVELENGTH'])
            maxwave = np.max(data['WAVELENGTH'])
            print(f'before interpolation, wave range is: {minwave}, {maxwave}')
            downsize_waves = np.linspace(minwave, maxwave, num=3000)
            sense_interp = interp1d(data['WAVELENGTH'], data['SENSITIVITY'], kind='cubic')
            err_interp = interp1d(data['WAVELENGTH'], data['ERROR'], kind='cubic')
            downsize_sensitivity = sense_interp(downsize_waves)
            downsize_err = err_interp(downsize_waves)

            #deltalambda = data['WAVELENGTH'][2000] - data['WAVELENGTH'][1999]
            #deltalambda3000 = (maxwave - minwave) / 3000
            #num_steps = deltalambda3000 / deltalambda
            #print(deltalambda, deltalambda3000, num_steps)
            #stop
            #newwaves = minwave + deltalambda * np.arange(3000)


            interp_data = {}
            interp_data['WAVELENGTH'] = downsize_waves
            interp_data['SENSITIVITY'] = downsize_sensitivity
            interp_data['ERROR'] = downsize_err
            data = interp_data

            if 'F322W2.R.A.1st' in sensitivity_file:
                print(f'After interpolating, the max wavelength is: {np.max(data["WAVELENGTH"])}')
                #firstbad = np.where((data['WAVELENGTH'] > 4.252) & (data['WAVELENGTH'] < 4.254))[0][0]
                #print(data['WAVELENGTH'][firstbad-6:firstbad+10], data['SENSITIVITY'][firstbad-6:firstbad+10])
                a.scatter(data['WAVELENGTH'], data["SENSITIVITY"], color='red')
                plt.show()

            minwave = np.min(data['WAVELENGTH'])
            maxwave = np.max(data['WAVELENGTH'])
            print(f'after interpolation, wave range is: {minwave}, {maxwave}')


        mjy, waves, err_mjy = convert_sensitivity_to_MJy(data)
        waves = waves.value
        mjy = mjy.value
        err_mjy = err_mjy.value


        if 'F322W2.R.A.1st' in sensitivity_file:
            import matplotlib.pyplot as plt
            f,a = plt.subplots()
            a.scatter(waves, mjy, color='red')
            a.scatter(waves, mjy+err_mjy, color='blue')
            a.scatter(waves, mjy-err_mjy, color='green')
            plt.show()



        basefile = os.path.basename(sensitivity_file)
        inst, filtername, grism_dir, module, orderval, _, _ = basefile.split('.')

        pix_area_arcsec = get_pixel_area(siaf[module])
        pix_area_sr = pix_area_arcsec.to(u.sr)

        mjysr = mjy / pix_area_sr.value
        err_mjysr = err_mjy / pix_area_sr.value

        # Now create the photmjsr and relresponse values for the file
        # The calibration is a simple multiplication: MJy/sr(lambda) = DN/s * photmjsr * relresponse(lambda)
        # So let's normalize at the element that is closest to the pivot wavelength
        pupil = f'GRISM{grism_dir}'
        order = orderval[0]

        pvwave = pivot[filtername]

        delta = np.abs(waves - pvwave)
        norm_ele = np.where(delta == np.min(delta))[0]

        photmjsr = mjysr[norm_ele]
        relresp = mjysr / photmjsr
        uncertainty = err_mjysr[norm_ele]
        relunc = err_mjysr / uncertainty
        nelem = len(mjysr)

        # In the relresp and relunc arrays, zero out entries
        # that are above a threshold where they are not contributing
        # to the total flux at all
        relresp, relunc, waves = filter_high_values(relresp, relunc, waves)


        # Sanity checks
        zerowave = np.where(waves == 0)[0]
        if len(zerowave) > 0:
            raise ValueError('zero wavelengths found')

        if len(waves) != len(relresp):
            raise ValueError('relresp and waves have different lengths')

        if len(waves) != len(relunc):
            raise ValueError('relunc and waves have different lengths')

        one_sens = np.where((waves == 0) & (relresp == 1))[0]
        if len(one_sens) > 0:
            raise ValueError('sensitivity 1 found before placing in arrays')


        # Add to arrays
        filters = np.append(filters, filtername)
        pupils = np.append(pupils, pupil)
        orders[i] = order
        photmjsrs[i] = photmjsr
        uncertainties[i] = uncertainty
        nelems[i] = nelem
        wavelengths[i, 0:len(waves)] = waves
        relresponses[i, 0:len(relresp)] = relresp
        reluncertainties[i, 0:len(relunc)] = relunc

        # All entries with a wavelength of 0 should have a sensitivity of 0
        zero_waves = np.where(wavelengths[i, :] == 0)[0]
        if len(zero_waves) > 0:
            print(f'zeroing out {len(zero_waves)} relresp and relunc points, where wavelength is zero')
        relresponses[i, zero_waves] = 0.
        reluncertainties[i, zero_waves] = 0.

    # Now we need to find all of the filter/pupil/order combinations that were not covered
    # by the sensitivity files, and copy the existing entry from the current reference file
    # into the new reference file. In this case, for Nor's version 6b, this is the 2nd order
    # entries for many of the filters
    print('After processing all sensitivity files, we have entries for:')
    for f, p, o in zip(filters, pupils, orders):
        print(f'{f} {p} order {o}')

    for row in current_data:
        #print('')
        #print('Attempting to match:')
        #print(row['filter'], len(filters))
        #print(row['pupil'], len(pupils))
        #print(row['order'], len(orders))
        match = np.where((filters == row['filter']) & (pupils == row['pupil']) & (orders == row['order']))[0]
        if len(match) == 0:
            print(f"No updated entry found for {row['filter']}, {row['pupil']}, order {row['order']}. Copying existing entry from current reference file.")
            filters = np.append(filters, row['filter'])
            pupils = np.append(pupils, row['pupil'])
            orders = np.append(orders, row['order'])
            photmjsrs = np.append(photmjsrs, row['photmjsr'])
            uncertainties = np.append(uncertainties, row['uncertainty'])
            nelems = np.append(nelems, row['nelem'])
            wavelengths = np.vstack([wavelengths, row['wavelength']])
            relresponses = np.vstack([relresponses, row['relresponse']])
            reluncertainties = np.vstack([reluncertainties, row['reluncertainty']])

    # Translate arrays of data into the proper format for the fits table
    data_list = [(filters[i], pupils[i], orders[i], photmjsrs[i], uncertainties[i], nelems[i],
                 wavelengths[i], relresponses[i], reluncertainties[i]) for i in range(len(filters))]
    data = np.array(data_list,
                    dtype=[('filter', 'S12'),
                           ('pupil', 'S15'),
                           ('order', '<i2'),
                           ('photmjsr', '<f4'),
                           ('uncertainty', '<f4'),
                           ('nelem', '<i2'),
                           ('wavelength', '<f4', (nx,)),
                           ('relresponse', '<f4', (nx,)),
                           ('reluncertainty', '<f4', (nx,))])

    # Create datamodel instance
    output = datamodels.NrcWfssPhotomModel(phot_table=data)

    # Populate metadata
    output.meta.filetype = 'PHOTOM'
    output.meta.telescope = 'JWST'
    output.meta.reftype = 'PHOTOM'
    output.meta.pedigree = 'INFLIGHT 2022-04-29 2022-04-30'
    output.meta.description = DESCRIPTION
    output.meta.author = 'N. Pirzkal and B. Hilbert'
    output.meta.useafter = '2014-01-01T00:00:01'
    output.meta.instrument.name = 'NIRCAM'
    output.meta.instrument.detector = f'NRC{module}LONG'
    output.meta.exposure.type = 'NRC_WFSS'
    output.meta.exposure.p_exptype = 'NRC_WFSS|NRC_TSGRISM|NRC_GRISM|'
    output.meta.photometry.pixelarea_steradians = pix_area_sr.value
    output.meta.photometry.pixelarea_arcsecsq = pix_area_arcsec.value

    history_entry = util.create_history_entry(HISTORY)
    output.history.append(history_entry)

    outfile = f'nircam_{module}mod_wfss_photom.fits'
    output.save(outfile, overwrite=True)


def filter_high_values(relresponse, reluncertainty, wavelength):
    """Remove points with excessively high relresponse, since they won't contribute to
    the flux at all and just take extra time
    """
    threshold = 100000
    ok = np.where(relresponse <= threshold)[0]
    nbad = len(relresponse) - len(ok)
    relresponse = relresponse[ok]
    reluncertainty = reluncertainty[ok]
    wavelength = wavelength[ok]
    if nbad > 0:
        print(f'Removing {nbad} entries whose relresp values are above threshold.')
    return relresponse, reluncertainty, wavelength


if __name__ == '__main__':
    run()