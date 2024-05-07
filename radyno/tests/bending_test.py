# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:13:02 2024

@author: Andrea
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c,m_e,e,mu_0
import time
from radyno.utils.beamline_utils import beamline
from radyno.utils.beam_utils import beam
from radyno.utils.rad_utils import theor_synch_rad

"""
Sector dipoles beam dynamics test.
Note: 
If the beam has a very low aspect ratio (long and narrow), 
the imperfect energy conservation at the interface between 
one segment and the next is more noticeable, while the effect 
is negligible for aspect ratios approaching 1 or greater.
"""

if __name__=="__main__":
    
    t0 = time.time()
    np.random.seed(1)
    
    """beam params"""
    ave_gamma = 100                         # desired Lorentz factor
    sigdg = 0                               # relative energy spread [0.01 = 1%]
    mu = np.array([0,0,0])                  # beam centroid position [m]
    sig = np.array([1e-8,1e-8,1e-4])    # rms beam size [m]
    epsx = 1e-7                         # normalized x emittance [mm mrad]
    epsy = 1e-7                         # normalized y emittance [mm mrad]
    npart = 100                             # number of beam electrons
    
    """init beamline"""
    bl = beamline()
    
    """add beamline elements"""
    r_bend = 0.1                           # sector dipole bending radius [m]
    th_bend = np.pi/4                       # sector dipole bending angle [rad]
    
    
    bl.add_segment("CBM",
                    "cbm1",
                    r_bend=r_bend,
                    th_bend=th_bend,
                    lr="R",
                    gamma=ave_gamma)
    # bl.add_segment("drift",
    #                 "d2",
    #                 l=0.1)
    # bl.add_segment("CBM",
    #                 "cbm2",
    #                 r_bend=r_bend,
    #                 th_bend=th_bend,
    #                 lr="L",
    #                 gamma=ave_gamma)

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
    bl.run(dt=1e-5/c,
           dtmax=1e-5/c)                    # maximum integration timestep [m/c]: should be comparable with beam duration, 
    t1 = time.time()                        # i.e. length in [m] should be <= beam length
    print()
    print("execution time:",t1-t0,"s")
    
    """plot"""    
    bl.plot_beam_bendplane()
    
    
    
    
    
    
    