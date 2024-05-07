# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:56:52 2024

@author: Andrea
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c,m_e,e,mu_0
import time
from radyno.utils.beamline_utils import beamline
from radyno.utils.beam_utils import beam

if __name__=="__main__":
    t0 = time.time()
    np.random.seed(2)
    ave_gamma = 100             # desired Lorentz factor
    
    """init beamline"""
    bl = beamline()
    
    """add beamline elements"""
    bl.add_segment("CBM",
                    "cbm1",
                    r_bend=0.5,
                    th_bend=np.pi/8,
                    lr="R",
                    gamma=ave_gamma)
    bl.add_segment("drift",
                    "d2",
                    l=0.1)
    bl.add_segment("CBM",
                    "cbm2",
                    r_bend=0.5,
                    th_bend=np.pi/8,
                    lr="L",
                    gamma=ave_gamma)
    bl.add_segment("undulator",
                    "u1",
                    l_U=0.03,
                    K=50,
                    l=0.1,
                    gamma=ave_gamma)
    
    """init beam"""
    bm = beam(mu=np.array([0.0,
                           0,
                           0]),
              sig=np.array([1e-6,
                            1e-6,
                            1e-5]),
              aveg=ave_gamma,
              sigdg=0.00,
              epsx=2e-8,
              epsy=2e-8,
              npart=1)
        
    """add beam to beamline"""
    bl.add_beam(beam=bm,        
                istart=0)       # index of starting beamline element: 0 for the first
    
    """run beamline simulation"""
    bl.run(dt=1e-5/c,
           dtmax=1e-5/c)
    t1 = time.time()
    print()
    print("execution time:",t1-t0,"s")
    
    """add radiation detectors"""
    oc = 3*c*ave_gamma**3/2/0.5
    fc = oc/2/np.pi
    
    l1 = bl.segments["u1"].kwargs["l_1"]                        # first harmonic wavelength [m]
    o1 = c*2*np.pi/l1                                           # first harmonic angular frequency [s^-1]
    f1 = o1/2/np.pi   
    
    segdict = {"cbm1":  [np.array([[np.pi/16,0,0,1.]]),
                         np.linspace(0.1*fc,4*fc,100)],
               
               "u1":    [np.array([[0,0,1.]]),
                         np.linspace(0.1*f1,200000*f1,1000)]}
    bl.add_detectors(segdict)
    
    """calculate radiation"""
    bl.radiation(parall=False)
    
    """plot"""    
    bl.plot_beam_bendplane()
    
    
    
    
    
    
    