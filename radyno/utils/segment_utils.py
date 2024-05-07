# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:17:35 2024

@author: Andrea
"""
import numpy as np
from scipy.constants import c,m_e,e,mu_0,epsilon_0
from .beam_utils import beam
import radyno.utils.fields_utils as fu

class segment():
    def __init__(self,device,name,**kwargs):
        self.field = fu.field_dict[device]
        self.kind = device
        self.name = name
        self.kwargs = self.check_kwargs(device,kwargs)
        
        """simulation times and evaluation points"""
        self.t_end = self.kwargs["l"]/c
        
        """utils"""
    
    def add_timesteps(self,dt):
        """beam dynamics simulation timestep value and number"""
        self.dt = dt
        self.npts = int(self.t_end/dt)
        
    def init_arrays(self,npart):
        """
        init empty arrays for segment beam variables storage: note that segment
        should first have a self.npts attribute, so self.add_timesteps should 
        first be run
        """
        # assert hasattr(self.__class__, 'self.npts')
        self.pos = np.zeros((self.npts,npart,3))
        self.sigpos = np.zeros((self.npts,3))
        self.mupos = np.zeros((self.npts,3))
        self.bet = np.zeros((self.npts,npart,3))
        self.gam = np.zeros((self.npts,npart))
        self.E_field = np.zeros((self.npts,npart,3))
        self.B_field = np.zeros((self.npts,npart,3))
    
    def check_kwargs(self,device,kwargs):
        d = {"cff":1}
        kwargs.update(d)
        
        if device=="drift":
            assert "l" in kwargs
            r_bend = 0
            d = {"r_bend":r_bend}
            kwargs.update(d)
        
        elif device=="undulator":
            assert "K" in kwargs and "l_U" in kwargs and "l" in kwargs and "gamma" in kwargs
            B0 = kwargs['K']*2*np.pi*m_e*c/e/kwargs['l_U']              # magnetic field max amplitude [T]
            k_U = 2*np.pi/kwargs['l_U']                                 # undulator wavevector [m^-1]
            l_1 = kwargs['l_U']/2/kwargs["gamma"]**2*(1 + kwargs["K"]**2/2)        # first harmonic wavelength [m^-1]
            r_bend = 0
            d = {"B0":B0,
                 "k_U":k_U,
                 "l_1":l_1,
                 "r_bend":r_bend}
            kwargs.update(d)
        
        elif device=="ion channel":
            assert "l_b" in kwargs and "K" in kwargs and "l" in kwargs and "gamma" in kwargs
            n_p = 8*np.pi**2*epsilon_0*m_e*c**2*kwargs["gamma"]/e**2/kwargs["l_b"]**2
            omega_p = np.sqrt(n_p*e**2/m_e/epsilon_0)
            k_b = 2*np.pi/kwargs["l_b"]
            r_off = kwargs["K"]/kwargs["gamma"]/k_b
            r_bend = 0
            d = {"omega_p":omega_p,
                 "k_b":k_b,
                 "r_off":r_off,
                 "r_bend":r_bend,
                 "n_p":n_p}
            kwargs.update(d)
        
        elif device=="CBM":
            assert "B0" in kwargs or "r_bend" in kwargs
            assert "lr" in kwargs and "th_bend" in kwargs and "gamma" in kwargs
            aveb = np.sqrt(1-1/kwargs["gamma"]**2)
            if kwargs["lr"] == "L":
                kwargs["cff"] = -1
            if "B0" in kwargs:
                r_bend = aveb*kwargs["gamma"]*m_e*c/kwargs["B0"]/e
                l = r_bend*kwargs["th_bend"]
                d = {"l":l,
                     "r_bend":r_bend}
            elif "r_bend" in kwargs:
                B0 = aveb*kwargs["gamma"]*m_e*c/kwargs["r_bend"]/e
                l = kwargs["r_bend"]*kwargs["th_bend"]
                d = {"l":l,
                     "B0":B0}
            kwargs.update(d)

        elif device=="ABP":
            """change J with I"""
            assert "r_c" in kwargs and "rho_c" in kwargs and "J" in kwargs and "lr" in kwargs and "th_bend" in kwargs and "gamma" in kwargs
            B1 = mu_0*kwargs["J"]/2                                # field slope [T/m]
            I = kwargs["J"]*np.pi*kwargs["r_c"]**2                           # discharge current [A]
            if kwargs["lr"] == "L":
                kwargs["cff"] = -1
            aveb = np.sqrt(1-1/kwargs["gamma"]**2)
            r_bend = kwargs["rho_c"]/2*(1+np.sqrt(1 + aveb*kwargs["gamma"]*m_e*c*8*np.pi*kwargs["r_c"]**2/(mu_0*e*kwargs["rho_c"]**2*I)))
            l = self.r_bend*self.th_bend
            d = {"B1":B1,
                 "I":I,
                 "l":l,
                 "r_bend":r_bend}
            kwargs.update(d)
        
        return kwargs
    
    def add_detectors(self,specs):
        """
        Init evaluation frequencies array and empty array for differential 
        intensity from each detector
        """
        dets,freqs = specs
        self.freqs = freqs
        self.U = np.zeros((len(dets),len(freqs)))
    
    def match(self):
        """overwrites original segment kwargs and matches parameters to given beam"""
