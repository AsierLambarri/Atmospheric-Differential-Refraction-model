import os
import sys
from datetime import date, time, datetime
import numpy as np
from astropy.table import Table, Column
import matplotlib.pyplot as plt

try:
	import smplotlib
except:
	pass
	
#####################################  FUNCTIONS  ##################################### 
                                    #(of the model)#
def H2Pw(H, T):
    """Computes the water vapour pressure for a given temperature and Humidity.
    
    Parameters
    ----------
    H : float
    	Humidity in %.
    T : float
    	Temperature in Celsius.
    	
    Returns
    -------
    Pwsat : float
    	Water vapour pressure in mmHg.
    """
    Pwsat = 6.11*10**((7.5*T)/(T+237.7))
    return Pwsat*(H/100)*1.0E+2*0.00750062

def n15_760(λ):
    """Computes the quantity n-1 for an arbitrary λ (in microns)
    for P=760 mmHg and T=15ºC. Taken from Alexei V. Filippenko (1982).
    
    Parameters
    ----------
    λ : float
    	Wavelength in microns.
    
    Returns
    -------
    n15_760 : float
    	Refraction index for radiation of wavelength λ for T=15ºC and P=760 mmHg. 
    	Unitless.
    """
    n15_760 = 1.0E-6*( 64.328+29498.1/(146-(1.0/λ)**2) + (255.4/(41-(1.0/λ)**2)))
    return n15_760

def nT_P(λ, T, P, f):
    """Computes the refraction index for an arbitrary λ (in microns)
    for arbitrary P and T, and is corrected for water vapour (f). Taken from
    Alexei V. Filippenko (1982).
    
    Parameters
    ----------
    λ : float
    	Wavelength in microns.
    T : float
    	Temperature in Celsius.
    P : float
    	Pressure in mmHg.
    f : float
    	Water vapour pressure in mmHg.
    	
    Returns
    -------
    nT_P : float
    	Refraction index for wavelength λ and atm. conditions of (T,P,f). 
    	Unitless.
    """
    nT_P = 1+n15_760(λ)*(P*(1+1.0E-6*P*(1.049-0.0157*T))/(720.883*(1+0.003661*T)))-1.0E-6*((0.0624-0.000680/λ**2)/(1+0.003661*T))*f
    return nT_P

def Deviation(z, λ, T, P, f, λ_ref):
    """Computes the deviation from the real and observed zenital angle in arcsec
    as a function of the various physical parameters, with respect to the
    reference wavelength.
    
    Parameters
    ----------
    z : float
    	Observed zenithal angle in degrees.
    λ : float
    	Wavelength in microns.
    T : float
    	Temperature in Celsius.
    P : float
    	Pressure in mmHg.
    f : float
    	Water vapour pressure in mmHg.
    λ_ref : float
    	Reference wavelength for differential refraction, in microns.
    	
    Returns
    -------
    dr : float
    	Difference between true and observed zenithal angles: z-z_0, in arcseconds
    """
    dr = 206265*( nT_P(λ, T, P, f) - nT_P(λ_ref, T, P, f) )*np.tan(np.deg2rad(z))
    return dr 

def AtmLenghtscale(z, λ, r_0, λ_0):
    """Computes the seeing disc in arcsec for given λ and zenital angle in deg
    for r_0 and λ_0, usually: r_0=15 cm and λ_0=550 nm.
    
    Parameters
    ----------
    z : float
    	Observed zenithal angle in degrees.
    λ : float
    	Wavelength in microns.
    r_0 : float
    	Reference Fried parameter for a given reference wavelength, in meters.
    λ_0 : float
    	Reference wavelength for Fried parameter, in microns.
    	
    Returns
    -------
    r_seeing : float
    	Fried parameter at a given wavelength and zenithal angle, in meters.
    """
    r_seeing = r_0*(λ/λ_0)**1.2*np.cos(np.deg2rad(z))**0.6
    return  r_seeing

def Seeing(z, λ, r_0, λ_0):
    """Returns the angular resolution due to the seeing.
    
    Parameters
    ----------
    z : float
    	Observed zenithal angle in degrees.
    λ : float
    	Wavelength in microns.
    r_0 : float
    	Reference Fried parameter for a given reference wavelength, in meters.
    λ_0 : float
    	Reference wavelength for Fried parameter, in microns.
    	
    Returns
    -------
    resolution : float
    	Seeing, in arcseconds.
    """
    resolution = 1.22*(λ*1.0E-6)/AtmLenghtscale(z, λ, r_0, λ_0)*206265
    return resolution

###################################  FUNCTIONS  ########################################
                                #(of the program)#
                                
def MakeIndividualPlot(z, deviations, size, λ, 
	conditions=(10,581.2978,30), 
	λ_ref=0.45, 
        directory=None, 
        seeing=[0.15, 0.55]
        ):
    """Plots dz = z-z_0 vs. z_0 and scales the circle sizes (with respect to y scale) to have the
    size given by the seeing for the wavelength which the data corresponds to.
    
    Parameters
    ----------
    z : float, array
    	Observed zenithal angles, in degrees.
    deviations : float, array
    	z-z_0 for each z value. Same size as z, units of arcseconds.
    size : float, array
    	Seeing discs at each z value. Same size as z, units of arcseconds.
    λ : float
    	Wavelength that corresponds to the data, in microns.
    conditions : float, tuple, optional
    	T,P,H atmosferic conditions used to evaluate the model.
    λ_ref : float, optional
    	Reference wavelength used to evaluate the model, in microns.
    directory : str, optional
    	Directory to which save the image.
    seeing :  float, array, optional
    	Parameters used to evaluate the seeing model.
    	
    Returns
    -------
    -An image saved to the specified directory.
    """                  
    T, P, H = conditions
    fig, ax = plt.subplots(figsize=(12,8))

    plt.title(f'Atmospheric differential refraction for {1000*λ} nm with respect to λ={1000*λ_ref} nm \nfor T={round(T+273,3)}'+
                f' K, P={round(P,1)} mmHg and H={H} %. Seeing parameters: r_0={seeing[0]} m, λ_0={seeing[1]} μm.')
    ax.set_xlabel('Apparent zenital angle (deg)')
    ax.set_ylabel('Δr (arcsec)')
    

    ax.set_xlim(-5,90)
    ax.set_ylim(-7,7)   
    y_to_pix_ratio = 633.5999999999998 #upright[1] - lowleft[1]
    

    ax.scatter(z, np.zeros_like(z) , color='none', label='1.8 arcsec reference disk',s=1.8**2*1.6*y_to_pix_ratio,edgecolors='black')

    ax.scatter(z, deviations, label=f'λ={1000*λ} nm',s=1.6*y_to_pix_ratio*size**2,alpha=0.3)
    ax.scatter(z, deviations, color='black', marker='+', s=19)
    
    ax.grid(True)

    legend = plt.legend(fontsize=11, loc='upper left')
    legend.legendHandles[0]._sizes = [100]
    legend.legendHandles[1]._sizes = [100]


    plt.savefig(f'{directory}/dispersion_ref{1000*λ_ref}_{1000*λ}nm.png')
    plt.close()

    return None

def MakeCombinedPlot(z, deviations_array, sizes_array, λ_array, 
	conditions=(10,581.2978,30), 
        λ_ref=0.45, 
        directory=None, 
        seeing=[0.15, 0.55]
        ):
    """Does thesame thing as MakeIndividualPlot but for multiple wavelengths.
    
    Parameters
    ----------
    z : float, array
    	Observed zenithal angles, in degrees.
    deviations_array : float, array
    	z-z_0 for each z value. One dimension has the same shape as z, the other the same units as λ_array.
    	Units of arcseconds.
    size_array : float, array
    	Seeing discs at each z value. One dimension has the same shape as z, the other the same units as λ_array.
    	Units of arcseconds.
    λ_array : float, array
    	Wavelength that corresponds to the data, in microns.
    conditions : float, tuple, optional
    	T,P,H atmosferic conditions used to evaluate the model.
    λ_ref : float, optional
    	Reference wavelength used to evaluate the model, in microns.
    directory : str, optional
    	Directory to which save the image.
    seeing :  float, array, optional
    	Parameters used to evaluate the seeing model.
    	
    Returns
    -------
    -An image saved to the specified directory.
    """                  
    Tarray, Parray, Harray = conditions
    
    fig, ax = plt.subplots(figsize=(12,8))

    plt.title(f'Atmospheric differential refraction with respect to λ={1000*λ_ref} nm.\nSeeing parameters: r_0={seeing[0]} m, λ_0={seeing[1]} μm.')
    ax.set_xlabel('Apparent zenital angle (deg)')
    ax.set_ylabel('Δr (arcsec)')
    
    ax.set_xlim(-5,90)
    ax.set_ylim(-7,7)   
    upright = ax.transData.transform((90,5.0))
    lowleft = ax.transData.transform((-7,-7.0))
    x_to_pix_ratio = upright[0] - lowleft[0]
    y_to_pix_ratio = upright[1] - lowleft[1]
    

    ax.scatter(z, np.zeros_like(z) , color='none', label='1.8 arcsec reference disk',s=1.8**2*1.6*y_to_pix_ratio,edgecolors='black')
    

    for deviations,size,T,P,H,λ in zip(deviations_array,sizes_array,Tarray,Parray,Harray,λ_array):
        ax.scatter(z, deviations, 
                   label=f'λ={1000*λ} nm with T={round(T+273,2)}K, P={round(P,2)} mmHg and H={H} %',
                   s=1.6*y_to_pix_ratio*size**2,alpha=0.3)
        ax.scatter(z, deviations, color='black', marker='+', s=19)

    ax.grid(True)

    legend = plt.legend(fontsize=11, loc='upper left')
    i=0
    for i in range(0,len(λ_array)+1):
        legend.legendHandles[i]._sizes = [100]


    plt.savefig(f'{directory}/dispersion_ref{1000*λ_ref}_combined.png')
    plt.close()

    return None

#def MakeAllPlot(z,deviations_array,sizes_array,λ_array, conditions=(10,581.2978,30), 
#                λ_ref=0.45, 
#                directory=None, seeing=[0.15, 0.55]):
#    Tarray, Parray, Harray = conditions
#    for deviations,size,T,P,H,λ in zip(deviations_array,sizes_array,Tarray,Parray,Harray,λ_array):
#        MakeIndividualPlot(z, deviations, size, λ, (T,P,H), λ_ref,directory)
#
#    MakeCombinedPlot(z,deviations_array,sizes_array,λ_array,conditions,λ_ref,directory)
#    
#    return None

def CreateAstropyTable(z_eval, λ_eval, R, s, IC, n, seeing):
    """Creates an Astropy Table with all the output of the model.
    
    Parameters
    ----------
    z_eval : float, array
    	Array of z values at which the model has been evaluated. Units of degrees.
    λ_eval : float, array
    	Array of wavelength values at which the model has been evaluated. Units of microns.
    	Length=n.
    R : float, array
    	Array containing z-z_0. Same length as z_eval. Units of arcsecs.
    s : float, array
    	Seeing discs. Same length as z_eval. Units of arcsecs.
    IC : float, array
    	Array of shape (3,n) containing (T,P,H) values (for each wavelength).
    n : float, array
    	Refraction indices predicted by the model.
    seeing : float, array
    	Seeing parameters r_0 and λ_0.
    	
    Returns
    -------
    tab : astropy table
    	Astropy table containing data generated by the models.
    """
    comment = 'This data file has been generated using an implementation of Alexei V. Filippenkos model (1982) by Asier Lambarri Martinez, while at UCMs MSc in Astrophysics. Please, acknowledge the author when using the code or data generated by it.\nThis file has been created using the Lastest version: 5 Feb. 2024 @12:00pm. Available in: https://github.com/AsierLambarri/Atmospheric-Differential-Refraction-model\n'

    tab = Table()

    tab.meta['Comment'] = comment.upper()
    tab.meta['Seeing parameters (r0 [m], λ_0 [μm])'] = tuple(seeing)
    tab.meta['Atm. cond. units'] = 'T in Celsius, P in mmHg and H in humidity percentage, refraction index is unitless.'
    tab['z'] = z_eval
    
    for i, λ in enumerate(λ_eval):
        tab.meta[f'Atm. cond. for {λ}μm ({i+1})'] = tuple(np.append(IC[:,i], n[i]))
        tab[f'z-z_0 {λ}μm ({i+1})'] = R[i,:]
        tab[f'seeing {λ}μm ({i+1})'] = s[i,:]
        
    return tab

def check_initial_conds(IC):
    """Checks that user provided data in valid.
    
    Parameters
    ----------
    IC : float, array
    	Array of shape (4,n) containing (λ,T,P,H) values (for each wavelength).
    	
    Returns
    -------
    -True if everything is okay.
    -False otherwise.
    """
    if IC.shape[0]!=4:
        return False
    if np.any(IC[0,:]<=0):
        return False
    if np.any(IC[1,:]<-273):
        return False
    if np.any(IC[2,:]<=0):
        return False
    if np.any(IC[3,:]<0) or np.any(IC[3,:]>100):
        return False
    else:
        return True
    
    
##########################################################################################################################################
########################################################################################################################################## 
########################################################################################################################################## 
########################################################################################################################################## 


print('======================================================================================================================')
print('======================================================================================================================')
print('                                                                                                                    ')   
print('                                   ATMOSPHERIC DIFFERENTIAL REFRACTION AND SEEING                                   '.upper())
print('                                                                                                                    ')   
print('  This program is an implementation of Alexei V. Filippenkos model (1982) for atmospheric differential refraction.  ')
print('  It also models the effects of seeing with the usual Fried Parameter implementation. The atmospheric conditions    ')
print('  and the Fried parameter and Fried wavelength are user specified.                                                  ')
print('                                                                                                                    ')   
print('  The inputs are: -A reference wavelength (in μm) to compute the differential refraction.                           ')
print('                  -Wavelengths (in μm) for which you want to evaluate the model.                                    ')
print('                  -Temperature (ºC), Pressure (mmHg) and Humidity (%) to model the behaviour of n(λ,T,P,H).         ')
print('                  -Friend parameter r_0 (in cm) and wavelength λ_0 (in μm), to model the seeing.                    ')
print('                                                                                                                    ')   
print('                                                                                                                    ')   
print('  There are three MODES of working, one of which produces and EXAMPLE and two that let you evaluate the model:      ')
print('                                                                                                                    ')   
print('                  -EXAMPLE: Produces an example for λ=0.35, 0.5, 0.92 μm, with reference wavelength of              ')
print('                            λ_ref=0.45 μm, seeing parameters of r_0=15 cm and λ_0=0.55 μm and for atm.              ')
print('                            conditions of T=10 ºC, P=581 mmHg and H=30%.                                            ')
print('                                                                                                                    ')   
print('                  -MANUAL MODE: You can input all the parameters through the terminal or IDE. The pro-              ')
print('                            checks for the validity of the inputed parameters. It is intended to be                 ')
print('                            used to evaluate just one wavelength under a set of atm. cond. and seeing               ')
print('                            parameters and so, only lets you input one set of "initial conditions".                 ')
print('                                                                                                                    ')   
print('                  -FILE MODE: Lets you load a file containing a number of different (or equal) wavelengths          ')
print('                            and atm. conditions (different or equal between λs) for which to evaluate               ')
print('                            the model. It is the most flexible mode of the three. The reference wave-               ')
print('                            length and seeing parameters are inputed via the terminal at the start and              ')
print('                            are kept constant throughout. The file MUST contain more than one evaluation            ')
print('                            i.e. more than one line. For a single evaluation please use the manual mode.            ')
print('                                                                                                                    ')   
print('                            The file must be structured as follows, and must be a csv file:                         ')
print('                                                                                                                    ')
print('                                                  λ1, T1, P1, H1                                                    ')
print('                                                  λ2, T2, P2, H2                                                    ')
print('                                                  λ3, T3, P3, H3                                                    ')
print('                                                  ...                                                               ')
print('                                                                                                                    ')
print('  In each of the MODES, there are two outputs: a png image showing the different seeing and refractions and a ecsv  ')
print('  file on which contains 20 evaluations of the models between 0º and 80º zenithal angles. The spacing between       ')
print('  is sufficiently small so that using it as interpolation data is possible. Both are saved in a folder with name    ')
print('  {date}_{hh:mm:ss}_MODE.                                                                                           ')
print('                                                                                                                    ')   
print('  The program is under GPL-3.0 License. Please acknowledge the author if you use it, or use data generated by it.   ')
print('                                                                                                                    ') 
print('  -AUTHOR: Asier Lambarri Martinez                                                                                  ')  
print('                                                                                                                    ')   
print('  -VERSION: 5 Feb. 2024 @12:00pm.                                                                                   ')
print('            Available in: https://github.com/AsierLambarri/Atmospheric-Differential-Refraction-model                ')
print('                                                                                                                    ')   
print('Dependencies: numpy, matplotlib, astropy and smplotlib (optional).                                                  ')
print('                                                                                                                    ')   
print('======================================================================================================================')
print('======================================================================================================================')




while True:
    z = np.array([0,20,35,50,65,80])
    z_data = np.linspace(0,80,20) 
    ndata = len(z_data)

    print('\nChoose MODE: enter MANUAL for MANUAL MODE, FILE for FILE MODE of type EXAMPLE to produce an example. If you want to stop type STOP:')
    
    KEY1 = input()

    if KEY1.upper() == 'STOP':
        print('STOPPING...\n')
        sys.exit()
            
    folder = date.today().strftime("%d%m%Y")+'_'+datetime.now().strftime('%H:%M:%S')+f'_{KEY1.upper()}'
    
    
    if KEY1.upper() == 'FILE':
        print('You are now in FILE LOADING MODE:')
    
    elif KEY1.upper() == 'MANUAL':
        print('You are now in MANUAL MODE:')
         
    elif KEY1.lower() == 'example':
        r_0, λ_0, λ_ref = 0.15, 0.55, 0.45
        λ_1, λ_2, λ_3 = 0.35, 0.5, 0.92
        T, P, H = 10, 581.2978041, 30
        f = H2Pw(H, T)
        
        print(f'\nCrunching some numbers to produce an example...')
    
        R1, s1, n1 = Deviation(z,λ_1,T,P,f,λ_ref), Seeing(z,λ_1,r_0,λ_0), nT_P(λ_1,T,P,f)
        R2, s2, n2 = Deviation(z,λ_2,T,P,f,λ_ref), Seeing(z,λ_2,r_0,λ_0), nT_P(λ_2,T,P,f)
        R3, s3, n3 = Deviation(z,λ_3,T,P,f,λ_ref), Seeing(z,λ_3,r_0,λ_0), nT_P(λ_3,T,P,f)
        
        λ_array = np.array([λ_1, λ_2, λ_3])
        conditions_array = np.vstack((np.array([T,P,H]),np.array([T,P,H]),np.array([T,P,H]))).T
        deviations_array = np.vstack((R1, R2, R3))
        sizes_array = np.vstack((s1,s2,s3))
        
        print('Saving the evaluations...')
    
        os.mkdir(folder)
        filename = f'./{folder}/model_evaluation.ecsv'
                   
        tab = CreateAstropyTable(z, λ_array, deviations_array, sizes_array, conditions_array, [n1, n2, n3], [r_0,λ_0])    
        tab.write(filename, format='ascii.ecsv', overwrite=True)
        
        print(f'Making the plot...')
    
        MakeCombinedPlot(z, deviations_array, sizes_array, λ_array, conditions_array, λ_ref, folder, [r_0,λ_0])
        
        print(f'Example saved to {folder}\n')
        continue
        
    else:
        print('You didnt provide a valid mode... Try again.')
        continue
        
        
        
    
    print('\n\n\nPlease enter a reference wavelength, in μm:')
    
    λ_ref = float(input().strip('\n'))
    
    print('\nPlease enter a Fried parameter (m) and Fried Wavelength (μm):')
    
    r_0, λ_0 = input().strip('\n').split(',')
    r_0, λ_0 =float(r_0), float(λ_0)
    print(r_0, λ_0)
    
    
    
    if KEY1.upper() == 'FILE':
        print('For the correct functioning of this mode the file must fullfill these requirements:\n\n '+
              'i)   Must be a .csv file.\n ii)  Wavelength must be in μm.\n iii) Temperature must be given in Celsius.\n'+
              ' iv)  Pressure must be provided in mmHg.\n v)   Humidity must be provided in %.\n \n')
        print('The csv file must be structured as follows: \n\n λ1, T1, P1, H1 \n λ2, T2, P2, H2 \n λ3, T3, P3, H3 \n ...\n')
        print('The model will be evaluated at each wavelength, for the given atm. conditions.\n')
        print('Please, provide a file path/name:')
        
        while True:
            path = input()
            try:
                initial_conditions = np.loadtxt(path, comments='#', delimiter=',').T
                break
            except:
                print('File could not be found. Please ensure that the provided filename/path is correct\nand that the file exists:')
                
        if check_initial_conds(initial_conditions):
            print(f'\nInitial conditions sucessfully loaded from {path}...')
        else:
            print('The provided file contain invalid wavelength or/and atm. cond. data. Aborting...\n')
            sys.exit()
            
        print('Crunching some numbers...')
        
        λ_array = initial_conditions[0,:]
        conditions_array = initial_conditions[1:,:]
        n, m = len(λ_array), len(z)
        
        R = np.zeros((n,m))
        s = np.zeros((n,m))
        R_data = np.zeros((n,ndata))
        s_data = np.zeros((n,ndata))
        n_index = np.zeros(n)
    
        for i in range(0,n):
            λ = λ_array[i]
            T, P, H = conditions_array[:,i] 
            f = H2Pw(H, T)
            R[i,:], s[i,:] = Deviation(z,λ,T,P,f,λ_ref), Seeing(z,λ,r_0,λ_0)
            
            R_data[i,:], s_data[i,:], n_index[i] = Deviation(z_data,λ,T,P,f,λ_ref), Seeing(z_data,λ,r_0,λ_0), nT_P(λ,T,P,f)
    
        print('Saving the evaluations...')
    
        os.mkdir(folder)
        filename = f'./{folder}/model_evaluation.ecsv'
                   
        tab = CreateAstropyTable(z_data, λ_array, R_data, s_data, conditions_array, n_index, [r_0,λ_0])    
        tab.write(filename, format='ascii.ecsv', overwrite=True)
    
        print(f'Making the plot...')
        
        MakeCombinedPlot(z, R, s, λ_array, conditions_array, λ_ref, folder, [r_0,λ_0])
        
        print(f'Sucessful run... Data saved to {folder}\n')
        continue
    
        
        
    elif KEY1.upper() == 'MANUAL':
        while True:
            print('\nPlease rovide a wavelength, in μm:')
            λ = float(input())
            if λ<=0:
                print('Please provide a valid wavelength.')
            else:
                break
        
        while True:
            print('\nPlease provide ambient temperature, in Celsius:')
            T = float(input())
            if T>=-273:
                break
            else:
                print('Temperature must be above absolute zero.')
            
        while True:
            print('\nPlease provide ambient pressure, in mmHg:')
            P = float(input())
            if P>=0:
                break
            else:
                print('Pressure must be positive.')
            
        while True:
            print('\nPlease provide ambient humidity, in %:')
            H = float(input())
            if 0<=H<=100:
                break
            else:
                print('Humidity must be between 0 and 100 %.')
            
        
        print(f'\nProvided data is valid. Proceeding to crunch some numbers...')
    
        f = H2Pw(H, T)
        R1, s1, n1 = Deviation(z,λ,T,P,f,λ_ref), Seeing(z,λ,r_0,λ_0), nT_P(λ,T,P,f)
        R_dat, s_dat = Deviation(z_data,λ,T,P,f,λ_ref), Seeing(z_data,λ,r_0,λ_0)
    
        
        print('Saving the evaluations...')
    
        os.mkdir(folder)
        filename = f'./{folder}/model_evaluation.ecsv'
                   
        tab = Table([Column(z_data, name='z',dtype='float'),
                     Column(R_dat, name=f'z-z_0 {λ}μm (1)',dtype='float'), 
                     Column(s_dat,name=f'seeing {λ}μm (1)', dtype='float')]
                    )
        
        tab.meta['Comment'] = f'This data file has been generated using an implementation of Alexei V. Filippenkos model (1982) by Asier Lambarri Martinez, while at UCMs MSc in Astrophysics. Please, acknowledge the author when using the code or data generated by it.\nThis file has been created using the Lastest version: 5 Feb. 2024 @12:00pm. Available in: https://github.com/AsierLambarri/Atmospheric-Differential-Refraction-model\n'.upper()
        tab.meta['Seeing parameters (r0 [m], λ_0 [μm])'] = r_0,λ_0
        tab.meta[f'Atm. cond. units'] = 'T in Celsius, P in mmHg and H in humidity percentage, refraction index is unitless.'
        tab.meta[f'Atm. cond. for {λ}μm (1)'] = T,P,H,n1
    
        tab.write(filename, format='ascii.ecsv', overwrite=True)
        
        print(f'Making the plot...')
    
        MakeIndividualPlot(z, R1, s1, λ, (T,P,H), λ_ref, folder, [r_0,λ_0])
        
        print(f'Sucessful run... Data saved to {folder}\n')
        continue
    

    

 
    







