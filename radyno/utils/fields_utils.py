# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:54:18 2024

@author: Andrea
"""
import numpy as np
from scipy.constants import c,m_e,e,mu_0,epsilon_0

"""fields for several beamline elements and spatial ramp"""

def defdict():
    field_dict = {"drift":         no_field,
                  "undulator":     undulator_field,
                  "ion channel":   plasma_field,
                  "CBM":           CBM_field,
                  "ABP":           ABP_field}
    return field_dict

def no_field(x,y,z,**kwargs):
    zero = np.zeros_like(x)
    return zero,zero,zero,zero,zero,zero

def undulator_field(x,y,z,**kwargs):
    Ex = np.zeros_like(x)
    Ey = np.zeros_like(x)
    Ez = np.zeros_like(x)
    Bx = np.zeros_like(x)
    By = kwargs["B0"]*np.cos(kwargs["k_U"]*z)
    # By = kwargs["B0"]*np.cos(kwargs["k_U"]*(z))
    Bz = np.zeros_like(x)
    return Ex,Ey,Ez,Bx,By,Bz

def plasma_field(x,y,z,**kwargs):
    r = np.sqrt((x - kwargs["r_off"])**2 + y**2)
    theta = np.arctan2(y,(x - kwargs["r_off"]))
    Er = kwargs["n_p"]*e*r/2/epsilon_0
    Ex = Er*np.cos(theta)
    Ey = Er*np.sin(theta)
    Ez = np.zeros_like(x)
    Bx = np.zeros_like(x)
    By = np.zeros_like(x)
    Bz = np.zeros_like(x)
    return Ex,Ey,Ez,Bx,By,Bz    

def CBM_field(x,y,z,**kwargs):
    Ex = np.zeros_like(x)
    Ey = np.zeros_like(x)
    Ez = np.zeros_like(x)
    Bx = np.zeros_like(x)
    By = -kwargs["B0"]*np.ones_like(x)#*kwargs["cff"]
    Bz = np.zeros_like(x)
    return Ex,Ey,Ez,Bx,By,Bz

def ABP_field(x,y,z,**kwargs):
    r = np.sqrt(x**2 + z**2)
    dr = r-kwargs["rho_c"]
    drax = np.sqrt(dr**2 + y**2)            # distance from design trajectory [m]
    the = np.arctan2(y, dr)                 # azimuthal angle around capillary axis[rad]
    phi = np.arctan2(z, x)                  # angle around capillary bending axis[rad]
    Bmod = kwargs["B1"]*drax
    By = Bmod*np.cos(the)*kwargs["cff"] 
    Br = Bmod*np.sin(the)*(-kwargs["cff"])
    Bx = Br*np.cos(phi)
    Bz = Br*np.sin(phi)
    Ex = np.zeros_like(x)
    Ey = np.zeros_like(x)
    Ez = np.zeros_like(x)   
    return Ex,Ey,Ez,Bx,By,Bz

def heaviside_close(x1, x2):
    closeCheck = np.isclose(x1, np.zeros_like(x1))
    heavisideBare = np.heaviside(x1, 0.0)
    zeroVal = np.where(closeCheck, x2, 0.0)-np.where(closeCheck, heavisideBare, np.zeros_like(heavisideBare))
    result = heavisideBare+zeroVal
    return result

def spatial_field_ramp(z,z0,updown):
    """
    Fermi-Dirac function for spatial field linking.
    
    z: longitudinal coordinate [m]
    z0: field linking point [m]
    updown: 1 for upramp, -1 for downramp
    """
    
    """linking stiffnes"""
    # a = 1e-6
    # return 1/(1 + np.exp(-updown*(z - z0)/a))
    return np.heaviside(updown*(z-z0), (1+updown)/2)
    # return heaviside_close(updown*z,z0)

"""leave as last row"""
field_dict = defdict()
