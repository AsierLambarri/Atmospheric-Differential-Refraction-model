# Atmospheric-Differential-Refraction-model
This is an 'interactive' implementation of Alexei V. Filippenkos model (1982) ([1982PASP...94..715F](https://ui.adsabs.harvard.edu/abs/1982PASP...94..715F/abstract)) for atmospheric differential refraction in python. The effects of atmospheric Seeing are also modeled. Both of these effects need to be accounted for when, for example, a fiber spectrograph, in order to collect all the light-beam comming from the object under study. The existence of this mini-project has been inspired by VNathir's owns implementation.

## The code

The code is implemented such that the user can input, either via the Terminal or through a .txt or .csv file, the Wavelengths at which the model is to be evaluated and the Atmospheric conditions (that may be different for each of the wavelengths) for each of them, the reference wavelength to compute the 'differential' part of the model and the Fried Parameters to model the Seeing. The `.py` file is self-contained and doesnt need anything else

