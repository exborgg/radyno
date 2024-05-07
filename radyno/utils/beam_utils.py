# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:27:04 2024

@author: Andrea
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c,m_e,e,mu_0,epsilon_0
from scipy.integrate import solve_ivp
from .rad_utils import rad_parall,rad_no_parall

class beam():
    """default: gaussian beam"""
    def __init__(self,sig,mu,aveg,sigdg,epsx,epsy,npart):
        """default: gaussian beam"""
        self.mux = mu[0]        # x rms size [m]
        self.muy = mu[1]        # y rms size [m]
        self.muz = mu[2]        # z rms size [m]
        self.sigx = sig[0]      # x rms size [m]
        self.sigy = sig[1]      # y rms size [m]
        self.sigz = sig[2]      # z rms size [m]
        self.aveg = aveg        # average Lorentz factor
        self.sigdg = sigdg      # rms relative energy spread dgamma/gamma
        self.epsx = epsx        # x normalized emittance [mm mrad]
        self.epsy = epsy        # y normalized emittance [mm mrad] 
        self.npart = npart      # number of beam particles

        """calculated variables"""
        self.sigg = aveg*sigdg                                              # rms energy spread dgamma
        self.epsxr = np.sqrt(epsx**2/aveg**2/(1 + self.sigg**2/aveg**2))    # x rms emittance [mm mrad]
        self.epsyr = np.sqrt(epsy**2/aveg**2/(1 + self.sigg**2/aveg**2))    # y rms emittance [mm mrad]
    
        """gammas and positions"""
        self.gamma = np.random.normal(aveg,self.sigg,(npart))
        self.pos0 = np.random.normal(mu, sig, (npart, 3))          # particle initial positions [m]
        
        """angles: calculated throug normalized emittance expression with gamma spread WITHOUT trace space correlation"""
        try:
            self.sigthx = epsx*1e-6/self.sigx/aveg/np.sqrt(self.sigg**2/aveg**2+1)
        except:
            self.sigthx = 0
        try:
            self.sigthy = epsy*1e-6/self.sigy/aveg/np.sqrt(self.sigg**2/aveg**2+1)
        except:
            self.sigthy = 0
        self.thx = np.random.normal(0, self.sigthx, (npart))
        self.thy = np.random.normal(0, self.sigthy, (npart))
        self.th = np.sqrt(self.thx**2 + self.thy**2)
        self.ph = np.arctan2(self.thy, self.thx)
        
        """normalized velocities"""
        beta = np.sqrt(1-1/self.gamma**2)
        bx0 = beta*np.sin(self.th)*np.cos(self.ph)
        by0 = beta*np.sin(self.th)*np.sin(self.ph)
        bz0 = beta*np.cos(self.th)
        self.bet0 = np.zeros_like(self.pos0)
        self.bet0[:,0] = bx0
        self.bet0[:,1] = by0
        self.bet0[:,2] = bz0
        
    def plot(self,var,c0,c1):
        plt.figure()
        if var=="pos":
            xp = self.pos[:,:,c0]
            yp = self.pos[:,:,c1]
        if var=="sigpos":
            xp = self.mupos[:,c0]
            yp = self.sigpos[:,c1]
        if var=="bet":
            xp = self.bet[:,:,c0]
            yp = self.bet[:,:,c1]
        plt.plot(xp,yp,alpha=0.5)
        plt.scatter(xp[0],yp[0])