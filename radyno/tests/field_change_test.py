# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:43:58 2024

@author: Andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c,m_e,e,mu_0
import time
from radyno.utils.beamline_utils import beamline
from radyno.utils.beam_utils import beam

if __name__=="__main__":
    np.random.seed(2)

    """init beamline"""
    bl2 = beamline()
    
    """init beam"""
    ave_gamma = 100
    bm = beam(mu=np.array([0.0,
                           0,
                           0]),
              sig=np.array([1e-4,
                            1e-4,
                            1e-3]),
              aveg=ave_gamma,
              sigdg=0.00,
              epsx=1,
              epsy=1,
              npart=1000)
    
    """add beam to beamline"""
    bl2.add_beam(beam=bm, istart=0)
    
    """add elements"""
    bl2.add_segment("drift",
                    "d1",
                    l=0.1)
    bl2.add_segment("CBM",
                    "cbm1",
                    r_bend=1,
                    th_bend=np.pi/4,
                    lr="r",
                    gamma=ave_gamma)
    bl2.add_segment("undulator",
                   "u1",
                   l_U=0.01,
                   K=1,
                   l=1,
                   gamma=ave_gamma)
    bl2.add_segment("drift",
                    "d2",
                    l=0.1)
    
    """check field spatial change at device interface"""
    fl = bl2.link_fields(bl2.segments["cbm1"],bl2.segments["u1"],0.01)
    n = 1000
    x = np.linspace(0,0,n)
    y = np.linspace(0,0,n)
    z = np.linspace(-0.1,0.1,n)
    f = fl(x,y,z)[4,:]

    plt.figure()
    plt.plot(z,f)
    
    
    