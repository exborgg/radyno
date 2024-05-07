# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:05:50 2024

@author: Andrea
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c,m_e,e,mu_0
import time
from radyno.utils.beamline_utils import beamline
from radyno.utils.beam_utils import beam
from radyno.utils.rad_utils import theor_synch_rad,Hz_to_keV

"""
Sector dipole beam dynamics and radiation emission test. 
Beam spot is matched to dipole bending strength to avoid betatronic envelope 
oscillations. Beam particles trajectories are plotted in rectified coordinates.
Radiation is evaluated in some points along curved trajectory 
and compared with theoretical synchrotron radiation emission.
"""

if __name__=="__main__":
    
    t0 = time.time()
    np.random.seed(1)
    
    """switch to calulate spectra after beam dynamics"""
    calc_rad = True
    parallel_rad = True   
    nfreqs = 1000
    nangs = 2   
    
    """switch to see coherent/uncoherent or mixed emission cases:
        0 -> uncoherent emission (matched beam)
        1 -> coherent emission (matched beam)
        2 -> coherent to uncoherent emission (mismatched beam)"""
    coherency = 1
    
    """beam params"""
    npart = 100                             # number of beam electrons
    ave_gamma = 50                          # desired Lorentz factor
    sigdg = 0                               # relative energy spread [0.01 = 1%]
    mu = np.array([0,0,0])                  # beam centroid position [m]
    if coherency==0:
        sig = np.array([1e-5,1e-5,1e-5])    # rms beam size [m]
        epsx = 1                            # normalized x emittance [mm mrad]
        epsy = 1                            # normalized y emittance [mm mrad]
    elif coherency==1:
        sig = np.array([1e-8,1e-8,1e-8])    # rms beam size [m]
        epsx = 1e-6                         # normalized x emittance [mm mrad]
        epsy = 1e-6                         # normalized y emittance [mm mrad]
    elif coherency==2:
        sig = np.array([1e-8,1e-8,1e-8])    # rms beam size [m]
        epsx = 1e-4                         # normalized x emittance [mm mrad]
        epsy = 1e-4                         # normalized y emittance [mm mrad]
    
    """integration parameters"""
    dt=1e-4/c                               # radiation calculation timestep [s]
    dtmax=1e-5/c                            # maximum dynamics integration timestep [s]
    print()
    print("max radiation energy =",'{:0.2e}'.format(Hz_to_keV(1/(dt/2/ave_gamma**2))),"keV")
    
    """init beamline"""
    bl = beamline()
    
    """add beamline elements"""
    r_bend = 0.01                           # sector dipole bending radius [m]
    th_bend = np.pi/2                       # sector dipole bending angle [rad]
    bl.add_segment("CBM",
                    "cbm1",
                    r_bend=r_bend,
                    th_bend=th_bend,
                    lr="R",
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
        """set evaluation frequencies from theoretical calculations"""
        oc = 3*c*ave_gamma**3/2/0.5             # sychrotron critical frequency
        fc = oc/2/np.pi
        freqs = np.linspace(0.1*fc,4*fc,nfreqs)
        """
        add radiation detectors: 
        one at bending start, one at bending end, 1m from the beam 
        """
        dets = np.array([[np.pi*1/16,0,0,1.],
                         [np.pi*7/16,0,0,1.]])    # detector points
        segdict = {"cbm1":  [dets,
                              freqs]}
        bl.add_detectors(segdict)
        
        """calculate radiation: numerical and analytical"""
        bl.radiation(parall=parallel_rad)
        theor_spectrum = theor_synch_rad(2*np.pi*freqs,
                                          r_bend*100,
                                          ave_gamma,
                                          th=0)
        
        """plot spectrum"""
        U = bl.segments["cbm1"].U.transpose()
        plt.figure()
        plt.plot(Hz_to_keV(freqs),U,label="cbm1")
        plt.plot(Hz_to_keV(freqs),theor_spectrum*npart,"--",label="theor uncoherent")
        plt.plot(Hz_to_keV(freqs),theor_spectrum*npart**2,"--",label="theor coherent")
        plt.axvline(Hz_to_keV(fc),color="black",linestyle="--",label="critical energy")
        plt.xlabel("E [keV]")
        plt.ylabel(r"$d^2I/d\omega d\Omega$ [erg s]")
        plt.legend()
        plt.yscale("log")
    
    """plot beam trajectories"""    
    bl.plot_beam_bendplane()
    
    
    
    
    
    
    