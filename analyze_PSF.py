import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from scipy.ndimage import map_coordinates
from scipy.signal import convolve2d

def azimuthal_slice_interp_pa(image, ra_vec, dec_vec,
                              r_arcsec, n_samples=720,
                              interp_order=1):
    """
    Interpolate image intensities along a circle of radius r_arcsec, ignoring NaNs.
    Angles follow astronomical PA convention:
        0° = North (+Dec), increasing toward East (+RA).

    Written by ChatGPT.

    image : 2D array (rows=Dec, cols=RA)
    ra_vec, dec_vec : 1D arrays of on-sky coords [arcsec] for unflipped cube
    r_arcsec : fixed projected separation (arcsec)
    interp_order : 1=bilinear, 3=bicubic
    """
    ny, nx = image.shape

    conv_img = image

    # Column coordinate vector for current image
    ra_cols =  ra_vec
    dec_rows = dec_vec

    # Define position angles (0 = North, CCW to East)
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    DEC = r_arcsec * np.cos(angles)  # cos -> Dec (North at angle=0)
    RA = r_arcsec * np.sin(angles)  # sin -> RA (East at angle=90°)

    # Convert sky coords -> pixel indices
    dra = ra_cols[1] - ra_cols[0]
    ddec = dec_rows[1] - dec_rows[0]
    cols = (RA - ra_cols[0]) / dra
    rows = (DEC - dec_rows[0]) / ddec

    in_bounds = (cols >= 0) & (cols <= nx - 1) & (rows >= 0) & (rows <= ny - 1)

    intensities = np.full(n_samples, np.nan, dtype=float)
    if np.any(in_bounds):
        coords = np.vstack([rows[in_bounds], cols[in_bounds]])

        # NaN-aware interpolation: values + weights
        img_filled = np.nan_to_num(conv_img, nan=0.0)
        wmap = np.isfinite(conv_img).astype(float)

        vals = map_coordinates(img_filled, coords, order=interp_order,
                               mode="constant", cval=0.0)
        wts = map_coordinates(wmap, coords, order=interp_order,
                              mode="constant", cval=0.0)

        good = wts > 0
        intensities[in_bounds] = np.where(good, vals, np.nan)

    # Convert radians to degrees [0,360)
    angles_deg = (np.degrees(angles)) % 360.0
    return angles_deg, intensities

fig_dir = "./figures/"
filename = "/fast/jruffio/data/exosims/HWO_PSFs/ifu_like_cube.fits"
hdul = fits.open(filename)
data = hdul[0].data  # Assuming the PSF data is in the primary HDU
header = hdul[0].header
hdul.close()

if 0:
    wavelength = np.array([0.925375, 0.926125, 0.926875, 0.927625, 0.928375, 0.929125,
           0.929875, 0.930625, 0.931375, 0.932125, 0.932875, 0.933625,
           0.934375, 0.935125, 0.935875, 0.936625, 0.937375, 0.938125,
           0.938875, 0.939625, 0.940375, 0.941125, 0.941875, 0.942625,
           0.943375, 0.944125, 0.944875, 0.945625, 0.946375, 0.947125,
           0.947875, 0.948625, 0.949375, 0.950125, 0.950875, 0.951625,
           0.952375, 0.953125, 0.953875, 0.954625, 0.955375, 0.956125,
           0.956875, 0.957625, 0.958375, 0.959125, 0.959875, 0.960625,
           0.961375, 0.962125, 0.962875, 0.963625, 0.964375, 0.965125,
           0.965875, 0.966625, 0.967375, 0.968125, 0.968875, 0.969625,
           0.970375, 0.971125, 0.971875, 0.972625, 0.973375, 0.974125,
           0.974875, 0.975625, 0.976375, 0.977125, 0.977875, 0.978625,
           0.979375, 0.980125, 0.980875, 0.981625, 0.982375, 0.983125,
           0.983875, 0.984625, 0.985375, 0.986125, 0.986875, 0.987625,
           0.988375, 0.989125, 0.989875, 0.990625, 0.991375, 0.992125,
           0.992875, 0.993625, 0.994375, 0.995125, 0.995875, 0.996625,
           0.997375, 0.998125, 0.998875, 0.999625, 1.000375, 1.001125,
           1.001875, 1.002625, 1.003375, 1.004125, 1.004875, 1.005625,
           1.006375, 1.007125, 1.007875, 1.008625, 1.009375, 1.010125,
           1.010875, 1.011625, 1.012375, 1.013125, 1.013875, 1.014625,
           1.015375, 1.016125, 1.016875, 1.017625, 1.018375, 1.019125,
           1.019875, 1.020625, 1.021375, 1.022125, 1.022875, 1.023625,
           1.024375, 1.025125, 1.025875, 1.026625, 1.027375, 1.028125,
           1.028875, 1.029625, 1.030375, 1.031125, 1.031875, 1.032625,
           1.033375, 1.034125, 1.034875, 1.035625, 1.036375, 1.037125,
           1.037875, 1.038625, 1.039375, 1.040125, 1.040875, 1.041625,
           1.042375, 1.043125, 1.043875, 1.044625, 1.045375, 1.046125,
           1.046875, 1.047625, 1.048375, 1.049125, 1.049875, 1.050625,
           1.051375, 1.052125, 1.052875, 1.053625, 1.054375, 1.055125,
           1.055875, 1.056625, 1.057375, 1.058125, 1.058875, 1.059625,
           1.060375, 1.061125, 1.061875, 1.062625, 1.063375, 1.064125,
           1.064875, 1.065625, 1.066375, 1.067125, 1.067875, 1.068625,
           1.069375, 1.070125, 1.070875, 1.071625, 1.072375, 1.073125,
           1.073875, 1.074625])
    print(np.where(wavelength==1.000375))

    plt.imshow(data[100,:,:], origin='lower', cmap='viridis')
    print(wavelength[100])
    plt.clim([0,5e-10])
    plt.colorbar()
    plt.show()

platescale = 24./98.  # lambda/D
conv_aper_rad = 1  # lambda/D
sep0 = 5 # lambda/D

nl,ny,nx = data.shape
ra_vec = np.arange(0,platescale*nx, platescale)  # arcsec
dec_vec = np.arange(0,platescale*nx, platescale)
ra_vec -= np.nanmean(ra_vec)
dec_vec -= np.nanmean(dec_vec)
dra = ra_vec[1] - ra_vec[0]
ddec = dec_vec[1] - dec_vec[0]
fontsize = 12

ra_grid, y_grid = np.meshgrid(ra_vec, dec_vec)
rad_grid = np.sqrt(ra_grid ** 2 + y_grid ** 2)

image = data[100, :, :]
kernel = np.zeros(image.shape)
kernel[np.where(rad_grid < conv_aper_rad)] = 1
kernel /= np.sum(kernel)
conv_img = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plt.sca(axes[0])
im = plt.imshow(conv_img, origin='lower', cmap='viridis',
                    extent=[ra_vec[0]-dra/2.,ra_vec[-1]+dra/2.,dec_vec[0]-ddec/2.,dec_vec[-1]+ddec/2.])
circle = plt.Circle((0,0), sep0, color="#ff9900", fill=False, linestyle='--', linewidth=3)
plt.gca().add_patch(circle)
txt = plt.text(0.0, sep0+0.005, "{0} $\lambda$/D".format(sep0), fontsize=fontsize, ha='center', va='bottom',color="#ff9900")
plt.xlim([-13,13])
plt.ylim([-13,13])
plt.clim([0,5e-10])
plt.gca().invert_xaxis()
plt.gca().set_aspect('equal')
plt.xlabel("$\Delta$x ($\lambda$/D)",fontsize=fontsize)
plt.ylabel("$\Delta$y ($\lambda$/D)",fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=fontsize)
plt.gca().tick_params(axis='y', labelsize=fontsize)
cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cb.set_label(r'Contrast', labelpad=5, fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=fontsize)
plt.gca().tick_params(axis='y', labelsize=fontsize)


plt.sca(axes[1])
# angles_deg_ori, intensities_ori = azimuthal_slice_interp_pa(image, ra_vec, dec_vec,sep0, n_samples=720)
angles_deg, intensities = azimuthal_slice_interp_pa(conv_img, ra_vec, dec_vec, sep0, n_samples=720)
plt.plot(angles_deg, intensities,color="#ff9900",linestyle="-",label="Starlight")
plt.plot(angles_deg, 1e-10+np.zeros(angles_deg.shape),color="#006699",linestyle="--",label="Planet")
txt = plt.text(0.03, 0.99, "{0} $\lambda$/D".format(sep0), fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
print(np.sqrt(np.mean(intensities)*np.mean(1/intensities)))
plt.xlabel('Position Angle (deg)',fontsize=fontsize)
plt.ylabel('Contrast',fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=fontsize)
plt.gca().tick_params(axis='y', labelsize=fontsize)
plt.legend(loc="upper right",frameon=True,fontsize=10)


plt.sca(axes[2])
sep_vec = np.linspace(0.1, 20, 200)
gain_vec = np.zeros_like(sep_vec)
gain_planet_vec = np.zeros_like(sep_vec)
for sep_id,sep in enumerate(sep_vec):
    angles_deg, intensities = azimuthal_slice_interp_pa(conv_img, ra_vec, dec_vec,sep, n_samples=720)
    gain_vec[sep_id] = np.sqrt(np.mean(intensities)*np.mean(1/intensities))
    gain_planet_vec[sep_id] = np.sqrt(np.mean((intensities+1e-10))*np.mean(1/(intensities+1e-10)))

plt.plot(sep_vec, gain_vec,color="#ff9900",linestyle="-",label="Starlight only")
plt.plot(sep_vec, gain_planet_vec,color="#006699",linestyle="--",label="Starlight and planet")
plt.xlabel('Separation ($\lambda$/D)',fontsize=fontsize)
plt.ylabel('Average Gain',fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=fontsize)
plt.gca().tick_params(axis='y', labelsize=fontsize)
plt.legend(loc="upper left",frameon=True,fontsize=10)


plt.tight_layout()
out_filename = os.path.join(fig_dir, "hwo_psf_avg_gain.png")
print("Saving " + out_filename)
plt.savefig(out_filename, dpi=300)
plt.savefig(out_filename.replace(".png", ".pdf"))
plt.show()