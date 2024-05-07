# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:17:33 2024

@author: Andrea
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c,m_e,e,mu_0
import time
from radyno.utils.beamline_utils import beamline
from radyno.utils.beam_utils import beam
from radyno.utils.rad_utils import theor_und_rad,Hz_to_keV

"""
Ion channel beam dynamics and radiation emission test. 
Radiation is evaluated in some points at the end of undulator
and compared with theoretical ion channel radiation emission.
"""

if __name__=="__main__":
    
    t0 = time.time()
    np.random.seed(1)
    
    """switches to calulate spectra after beam dynamics"""
    calc_rad = True
    parallel_rad = True   
    nfreqs = 1000
    nangs = 2              
    
    """switch to see coherent/uncoherent or mixed emission cases:
        0 -> uncoherent emission (matched beam)
        1 -> coherent emission (matched beam)"""
    coherency = 1
    
    """beam params"""
    npart = 100                             # number of beam electrons
    ave_gamma = 50                          # desired Lorentz factor
    sigdg = 0                               # relative energy spread [0.01 = 1%]
    mu = np.array([0,0,0])                  # beam centroid position [m]
    if coherency==0:
        sig = np.array([1e-5,1e-5,1e-5])    # rms beam size [m]
        epsx = 1e-6                         # normalized x emittance [mm mrad]
        epsy = 1e-6                         # normalized y emittance [mm mrad]
    elif coherency==1:
        sig = np.array([1e-7,1e-7,1e-7])    # rms beam size [m]
        epsx = 1e-6                         # normalized x emittance [mm mrad]
        epsy = 1e-6                         # normalized y emittance [mm mrad]
    
    """integration parameters"""
    dt=1e-4/c                               # radiation calculation timestep [s]
    dtmax=1e-5/c                            # maximum dynamics integration timestep [s]
    print()
    print("max radiation energy =",'{:0.2e}'.format(Hz_to_keV(1/(dt/2/ave_gamma**2))),"keV")
    
    """init beamline"""
    bl = beamline()
    
    """add beamline elements"""
    K = 1                                   # undulator strength
    l_b = 0.03                              # betatronic wavelength [m]
    k_U = 2*np.pi/l_b                       # betatronic wavevector [m^-1]
    l = 0.1
    bl.add_segment("ion channel",
                    "ic1",
                    l_b=l_b,
                    K=K,
                    l=l,
                    gamma=ave_gamma)

    """init beam"""
    bm = beam(mu=mu,
              sig=sig,
              aveg=ave_gamma,
              sigdg=sigdg,
              epsx=epsx,
              epsy=epsy,
              npart=npart)
        
    """add beam to beamline"""
    bl.add_beam(beam=bm,        
                istart=0)                   # index of starting beamline element: 0 for the first
    
    """run beamline simulation"""
    bl.run(dt=dt,
           dtmax=dtmax)                     
    t1 = time.time()                        
    print()
    print("execution time:",t1-t0,"s")
    
    if calc_rad == True:
        """calculate variables for spectral evaluation"""
        if K<1:
            o = 4*np.pi*ave_gamma**2*c/l_b/(1 + K**2/2)
            lab = "first IC harmonic"
        else:
            o = 3*np.pi*K*ave_gamma**2*c/l_b              
            lab = "IC critical frequency"
        f1 = o/2/np.pi   
        freqs = np.linspace(0.1*f1, 4*f1, nfreqs)                    # frequencies [s^-1]
           
        """add radiation detectors: one on axis, one off"""
        det_dist = 10.
        KoG = K/ave_gamma
        thetas = np.linspace(0,2*KoG,nangs)
        det_dist = 10
        dets = [[det_dist*np.sin(theta),0,det_dist*np.cos(theta)] for theta in thetas] #detector points
        dets = np.array(dets)
        segdict = {"ic1":  [dets,
                           freqs]}
        bl.add_detectors(segdict)
        
        """calculate radiation: numerical and analytical"""
        bl.radiation(parall=parallel_rad)
        theor_spectrum = theor_und_rad(2*np.pi*freqs,
                                        th=0,
                                        ph=0,
                                        LU=l*1e2,
                                        KU=K,
                                        lambU=l_b*1e2,
                                        gam0=ave_gamma)
        
        """plot spectrum"""
        U = bl.segments["ic1"].U.transpose()
        plt.figure()
        plt.plot(freqs,U,label="ic1")
        plt.plot(freqs,theor_spectrum*npart,"--",label="theor und uncoherent")
        plt.plot(freqs,theor_spectrum*npart**2,"--",label="theor und coherent")
        plt.axvline(f1,color="black",linestyle="--",label="first harmonic")
        plt.legend()
        
    """plot beam trajectories"""    
    bl.plot_beam_bendplane()

    
    
    
    
    
    
    