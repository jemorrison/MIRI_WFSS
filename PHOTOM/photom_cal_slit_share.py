import astropy.io.fits as fits
from glob import glob
import numpy as np
import pdb
from pylab import *
import matplotlib as mpl 
mpl.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal, interpolate
from stdatamodels.jwst import datamodels
import copy
from datetime import datetime
import pickle

## ==== Program and observation information ====
data_dir   = SELECT DIRECTORY	# Directory for extracted spectra (with final slash)
spec_dir   = SELECT DIRECTORY	# Directory with CALSPEC spectra (with final slash)
## ========== Target information ===============
prog_ids = ['01536', '04496', '04496']
obs_numbs = ['027', '015', '017']
target_names = ['bd60_1753', 'hd55677', 'j1757132']
## =============================================

# Get comparison calibration from latest PHOTOM ref file
file = 'jwst_miri_photom_0216.fits'
phot_file = datamodels.MirLrsPhotomModel(file)
hdulist_ref = fits.open(file)
data_ref = hdulist_ref['PHOTOM', 1].data
wave_comp = data_ref['wavelength'][0]
relres_comp = data_ref['relresponse'][0]
pixar_sr = 2.86063256542560e-13
pixar_a2 = 0.01217199
photmjsr = data_ref['photmjsr'][0]
photmjsr_err = data_ref['uncertainty'][0]
hdulist_ref.close()

## =============================================

for i, pid in enumerate(prog_ids[:3]):

	# Read x1d file
	x1d_file = f'{data_dir}jw{pid}-o{obs_numbs[i]}_t001_miri_p750l_x1d.fits'
	hdulist = fits.open(x1d_file)
	spec_data = hdulist['EXTRACT1D', 1].data
	wave, flux, fluxerr = spec_data['WAVELENGTH'], spec_data['FLUX'], spec_data['FLUX_ERROR']		# flux in DN/sec

	# Read in CALSPEC spectrum
	starfile = glob(f'{spec_dir}{target_names[i]}*slit.sp.tbl')[0]
	stardata = np.genfromtxt(starfile, skip_header=6)
	wavestar = stardata[:,0]		# in microns; this array is the same for all CALSPEC spectra and the default for the output
	fluxstar = stardata[:,1]		# in Jy
	fluxstar /= 1.e6 * pixar_sr		# in MJy/sr

	# Clean JWST spectrum
	wave_trim = wave[~np.isnan(flux)]	# Filter out NaNs
	smoothed_flux = signal.savgol_filter(flux[~np.isnan(flux)], 9, 3)	# Create smoothed spectrum
	fint = interpolate.interp1d(wave_trim, smoothed_flux, kind='cubic', fill_value='extrapolate')	# Interpolate to plug in gaps
	smoothed_flux = fint(wave)
	diff = flux - smoothed_flux			# Calculate difference
	mask = np.where(np.absolute(diff) < 0.5 * smoothed_flux)	# Filter outliers
	wave, flux, fluxerr = wave[mask], flux[mask], fluxerr[mask]

	# Resample JWST spectrum to stellar grid
	fint = interpolate.interp1d(wave[~np.isnan(flux)], flux[~np.isnan(flux)], kind='cubic', fill_value='extrapolate')
	flux_resampled = fint(wavestar)
	fint = interpolate.interp1d(wave[~np.isnan(flux)], fluxerr[~np.isnan(flux)], kind='cubic', fill_value='extrapolate')
	flux_resampled_err = fint(wavestar)

	# Calculate ratio
	ratio = fluxstar / flux_resampled
	ratio_err = ratio * flux_resampled_err / flux_resampled

	if i == 1:
		bad = np.where((wave > 7.2) & (wave < 7.6))
		ratio[bad] = np.nan

	# Compile relative response arrays
	if i == 0:
		relres_arrays = ratio
		relres_err_arrays = ratio_err
	else:
		relres_arrays = np.vstack([relres_arrays, ratio])
		relres_err_arrays = np.vstack([relres_err_arrays, ratio_err])

	# Plot
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111)
	ax.semilogy(wavestar, ratio, 'g.', zorder=2)
	ax.semilogy(wave_comp, photmjsr * relres_comp, 'r-', zorder=1)
	ax.xaxis.set_major_locator(MultipleLocator(1))
	ax.xaxis.set_minor_locator(MultipleLocator(0.2))
	ax.tick_params(labelsize=16)
	ax.tick_params(labelsize=16, top=True, bottom=True, left=True, right=True)
	ax.set_xlabel(r'Wavelength [$\mu$m]',fontsize=18)
	ax.set_ylabel('MJy/sr/(DN/sec)',fontsize=18)
	ax.set_ylim([5, 5e3])

	plt.title(f'PID {pid}, Obs {obs_numbs[i]} ({target_names[i]})', fontsize=20)
	plt.tight_layout()
	plt.savefig(f'photom_cal_{pid}_{target_names[i]}.png')
	plt.close()


# Average the relres arrays across all stars
relres = np.nanmean(relres_arrays, axis=0)
relres_err = np.sqrt(np.nansum(relres_err_arrays**2, axis=0))/ np.sum(~np.isnan(relres_arrays), axis=0)

# Normalize to 7 microns
fint = interpolate.interp1d(wavestar, relres, kind='linear')
new_photmjsr = fint(7.0)
fint = interpolate.interp1d(wavestar, relres_err, kind='linear')
new_photmjsr_err = fint(7.0)

# Invert to reflect native LRS wavelength solution
wave_out = wavestar[::-1]
relres_out = relres[::-1] / new_photmjsr
relres_err_out = relres_err[::-1] / new_photmjsr

# Extend arrays to lower wavelengths using previous ref file data
length_diff = len(wave_comp) - len(wave_out)
wave_out = np.concatenate([wave_out, wave_comp[-length_diff:]])
relres_out = np.concatenate([relres_out, relres_comp[-length_diff:]])
relres_err_out = np.concatenate([relres_err_out, np.array([np.nan]*length_diff)])

# Create new ref file
nelem = len(wave_out)
data_list = [('P750L',  'FULL', new_photmjsr, new_photmjsr_err, nelem, wave_out, relres_out, relres_err_out)]
phot_file.phot_table = data_list
out_file = 'jwst_miri_photom_slit_XXXXXX.fits'

phot_file.meta.filename = out_file
phot_file.meta.author = 'Ian Wong'
phot_file.meta.origin = 'STScI'
phot_file.meta.description = "LRS SLIT PHOTOM file generated during flight."
phot_file.meta.pedigree  = 'INFLIGHT 2022-07-08 2024-05-09'
phot_file.meta.useafter = '2022-04-01T00:00:00'
phot_file.meta.date = datetime.utcnow().isoformat()
del (phot_file.history[1:])
phot_file.history.append("This reference file was generated from A-star slit observations.")
phot_file.history.append("The following observations were used:")
phot_file.history.append("PID 1536, Obs 27 (BD+60 1753)")
phot_file.history.append("PID 4496, Obs 15 (HD 55677)")
phot_file.history.append("PID 4496, Obs 17 (J1757132)")
phot_file.history.append("The values were calculated by averaging the ratios between") 
phot_file.history.append("(1) the measured spectra extracted from the 2-nod difference")
phot_file.history.append("images, using the standard 8-column aperture, and")
phot_file.history.append("(2) the corresponding CALSPEC models convolved to the resolution of LRS.") 
phot_file.history.append("No smoothing has been applied to the relative pixel response array,")
phot_file.history.append("in order to correct for fixed-pattern noise apparent in the spectra.")

phot_file.save(out_file)     

hdulist = fits.open(out_file)
hdulist[0].header['PIXAR_SR'] = pixar_sr
hdulist[0].header['PIXAR_A2'] = pixar_a2
hdulist.writeto(out_file, overwrite=True)
hdulist.close()
