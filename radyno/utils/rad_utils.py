# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:26:09 2024

@author: Andrea
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
import sys
from multiprocessing import Array
import os
import h5py
from scipy.stats import median_abs_deviation as MAD
import multiprocessing as mp
import ctypes as ctp
import tqdm
from scipy.special import kv

c       = 2.99792458e8*1e2                  #cm/s
me      = 9.109e-31*1e3                     #g
e       = 4.8032e-10                        #statC
h       = 6.626176e-27                      #erg*s

npn = np.newaxis

def Hz_to_keV(Hz):
    return Hz/2.41799050402417e17

def rad_no_parall(xint,bint,gint,Eint,Bint,wt,xO,fr,t,ft_array):
    F            = -wt*e*(Eint + np.cross(bint,Bint))
    betdot       = (F - np.sum(bint*F,axis=1)[:,npn]*bint)/(c*me*wt*gint[:,npn])
    betdot       = betdot
    x            = xint
    bet          = bint
    gam          = gint
    #gamave      = np.nanmean(gam)
    """calcolo posizioni rispetto al detector e tempi ritardati"""
    Rvec         = xO - x                                                         
    R            = np.linalg.norm(Rvec,axis=1)
    n            = Rvec/R[:,npn]
    tend         = t + R/c                                #lab time when signal started at R gets to O
    """calcolo campi: A ∝ RE (jackson IIedition pg657,pg668)"""
    betdotn      = np.sum(bet*n,axis=1)
    A1           = np.sqrt(wt)*e*(n-bet)                                        /     (gam**2 * (1-betdotn)**3 * R)[:,npn]
    A2           = np.sqrt(wt)*(e/c)*(np.cross(n,np.cross((n-bet),betdot)))     /     ((1-betdotn)**3)[:,npn]
    A            = (c/4/np.pi)**0.5*(A1 + A2) 
    try:
        Aslo = (A[1:] - A[:-1])/(tend[1:,npn] - tend[:-1,npn])  
        Aslo = np.r_[Aslo , [np.array([0,0,0])]]  #field slope between consecutive points in linear interpolation
        Aint = A - Aslo*tend[:,npn]                                      #field intercept for linear interpolation
        Aint[-1] = np.array([0,0,0])
        A = np.c_[Aslo,Aint]
        lf = len(fr)
        ft = np.zeros((lf,3), dtype=complex)
        frN = fr
        lfN = len(frN)
        for i in range(lfN):
            om = 2*np.pi*frN[i]
            t1 = tend[1:][:,npn]
            t0 = tend[:-1][:,npn]
            f1 = np.exp(1j*om*t1)*(A[:-1,0:3]*(1-1j*om*t1) - 1j*om*A[:-1,3:6])
            f0 = np.exp(1j*om*t0)*(A[:-1,0:3]*(1-1j*om*t0) - 1j*om*A[:-1,3:6])
            df = (f1-f0)
            fttot = (1/np.sqrt(2*np.pi))*np.nansum(df,axis=0)/om**2
            ft[i] = fttot
        ft_array[:,0:3] += ft.real
        ft_array[:,3:6] += ft.imag
    except Exception as exc:
        print("noooo")
        pass
    return ft_array

def rad_parall(xint,bint,gint,Eint,Bint,wt,xO,fr,t,ft_array):
    pint = bint*gint[:,:,npn]
    wt = np.ones(gint.shape)*wt
    """share arrays for parallel computing"""
    rad_make_shared_array(xint, name='my_shared_array_6')           # create shared memory array from numpy array
    XfinS = rad_get_shared_array('my_shared_array_6')               # get shared memory array as numpy array
    rad_make_shared_array(pint, name='my_shared_array_6')  
    PfinS = rad_get_shared_array('my_shared_array_6')  
    rad_make_shared_array(Eint, name='my_shared_array_6')  
    EfinS = rad_get_shared_array('my_shared_array_6') 
    rad_make_shared_array(Bint, name='my_shared_array_6')  
    BfinS = rad_get_shared_array('my_shared_array_6') 
    rad_make_shared_array(wt, name='my_shared_array_6')  
    wtfinS = rad_get_shared_array('my_shared_array_6')  
    rad_make_shared_array(t, name='my_shared_array_6')
    tS = rad_get_shared_array('my_shared_array_6') 
    rad_make_shared_array(fr, name='my_shared_array_6')  
    freqsS = rad_get_shared_array('my_shared_array_6')
    rad_make_shared_array(xO, name='my_shared_array_6')  # create shared memory array from numpy array
    xOS = rad_get_shared_array('my_shared_array_6')  # get shared memory array as numpy array
    ftM = mp.Array(ctp.c_double, len(fr)*6) #all frequencies, 3 real parts, 3 imag parts

    items               = [i for i in range(xint.shape[1])]

    """parallel ft"""               
    with mp.Pool(initializer=rad_init, initargs=(ftM,
                                                 XfinS,
                                                 PfinS,
                                                 EfinS,
                                                 BfinS,
                                                 wtfinS,
                                                 xOS,
                                                 freqsS,
                                                 tS,)) as pool:
        results         = list(tqdm.tqdm(pool.imap(rad_task, items), total=(xint.shape[1])))
    
    fts                 = np.frombuffer(ftM.get_obj())
    fts                 = fts.reshape(len(fr),6)
    ft                  = fts[:,0:3] + 1j*fts[:,3:6]       
    return ft

def theor_und_rad(om,th,ph,LU,KU,lambU,gam0):
    """analytical undulator radiation spectrum [erg s]"""

    """support functions"""
    def term(m,xs,zt,K,om,N,th,fr):
        """single term of analytical undulator radiation summation"""
        Fm = (4*m*xs/K)**2*(jv((m-1)/2,m*xs) - jv((m+1)/2,m*xs))**2
        return Fm * sinc(nu(m,om,N,fr)/2)**2

    def sinc(x):
        return np.sin(x)/x

    def nu(m,om,N,fr):
        return 2*np.pi*N*(m - om/(2*np.pi*fr))

    """analytical undulator radiation"""
    order = 30    
    l1 = lambU/2/gam0**2*(1 + KU**2/2)
    o1 = c*2*np.pi/l1
    f1 = o1/2/np.pi    
    N = LU/lambU #3#100
    zt = -KU*om*th*np.cos(ph)/(gam0*o1)
    xs = KU**2/4/(1+KU**2/2)
    cf = N**2*e**2*gam0**2/c
    tr = 0
    for i in range(order):
        io = 2*i+1
        trp = term(io,xs,zt,KU,om,N,th,f1)   
        tr +=  trp      
    return cf*tr

def theor_synch_rad(om,rho,gam,th):
    xi = om*rho*(1/gam**2 + th**2)**(3/2)/3/c
    c1 = e**2/3/np.pi**2/c
    c2 = (om*rho/c)**2
    c3 = (1/gam**2 + th**2)**2
    ks = kv(2/3,xi)**2 + th**2*kv(1/3,xi)**2/(1/gam**2 + th**2)
    return c1*c2*c3*ks

def wtstd(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def rad_init(s1,s2,s3,s4,s5,s6,s7,s8,s9):
    global ftM
    global XfinS
    global PfinS
    global EfinS
    global BfinS
    global wtfinS
    global xOS
    global freqsS
    global tS
    
    ftM = s1
    XfinS = s2
    PfinS = s3
    EfinS = s4
    BfinS = s5
    wtfinS = s6
    xOS = s7
    freqsS = s8
    tS = s9
    
def rad_task(item):
    """CGS units"""

    it = item

    xint = XfinS[:,it,:]
    pint = PfinS[:,it,:]
    eint = EfinS[:,it,:]
    bint = BfinS[:,it,:]
    wt = np.nanmax(wtfinS[:,it])
    xO = xOS
    fr = freqsS
    t = tS

    pmod        = np.linalg.norm(pint,axis=1)
    gamint      = np.sqrt(1+pmod**2)
    betint      = pint/gamint[:,npn]
 
    F           = -wt*e*(eint + np.cross(betint,bint))
    betdot      = (F - np.sum(betint*F,axis=1)[:,npn]*betint)/(c*me*wt*gamint[:,npn])
    betdot      = betdot
    x           = xint
    bet         = betint
    gam         = gamint
    
    """position respect to detector and retarded times calculation"""
    Rvec         = xO - x                                                         
    R             = np.linalg.norm(Rvec,axis=1)
    n             = Rvec/R[:,npn]
    tend         = t + R/c                                #lab time when signal started at R gets to O
    
    """fields calculation: A ∝ RE (jackson IIedition pg657,pg668)"""
    betdotn        = np.sum(bet*n,axis=1)
    A1             = np.sqrt(wt)*e*(n-bet)                                        /     (gam**2 * (1-betdotn)**3 * R)[:,npn]
    A2             = np.sqrt(wt)*(e/c)*(np.cross(n,np.cross((n-bet),betdot)))     /     ((1-betdotn)**3)[:,npn]
    A               = (c/4/np.pi)**0.5*(A1 + A2) 
    try:
        Aslo = (A[1:] - A[:-1])/(tend[1:,npn] - tend[:-1,npn])  
        Aslo = np.r_[Aslo , [np.array([0,0,0])]]                        #field slope between consecutive points in linear interpolation
        Aint = A - Aslo*tend[:,npn]                                     #field intercept for linear interpolation
        Aint[-1] = np.array([0,0,0])
        addcol = np.c_[Aslo,Aint]
        tim = tend
        A = addcol
        lf = len(fr)
        ft = np.zeros((lf,3), dtype=complex)
        frN = fr
        lfN = len(frN)
        for i in range(lfN):
            om = 2*np.pi*frN[i]
            t1 = tim[1:][:,npn]
            t0 = tim[:-1][:,npn]
            f1 = np.exp(1j*om*t1)*(A[:-1,0:3]*(1-1j*om*t1) - 1j*om*A[:-1,3:6])
            f0 = np.exp(1j*om*t0)*(A[:-1,0:3]*(1-1j*om*t0) - 1j*om*A[:-1,3:6])
            df = (f1-f0)
            fttot = (1/np.sqrt(2*np.pi))*np.nansum(df,axis=0)/om**2
            ft[i] = fttot
            # ft[i] = (1/np.sqrt(2*np.pi))*np.trapz(A*np.exp(1j*om*tim[:,npn]),tim,axis=0)
        with ftM.get_lock():
            final_array = np.frombuffer(ftM.get_obj())
            final_array = final_array.reshape((lf,6))
            final_array[:,0:3] += ft.real
            final_array[:,3:6] += ft.imag
    except Exception as exc:
        print(exc)
        print(xint)
        print(pint)
        print(eint)
        print(bint)
        print(wt)
        sys.quit()

def rad_get_shared_array(name: str, shape=None):
    mp_array = globals()[name]
    np_array = np.frombuffer(mp_array.get_obj(), dtype=np.dtype(mp_array.get_obj()._type_))
    if (shape is None) and (name + '_shape' in globals().keys()):
        shape = globals()[name + '_shape']
        shape = np.frombuffer(shape.get_obj(), dtype=int)
    if shape is not None:
        np_array = np_array.reshape(shape)
    return np_array

def rad_make_shared_array(np_array: np.ndarray, name: str):
    mp_dtype = np.ctypeslib.as_ctypes(np_array.dtype.type())._type_
    mp_array = Array(typecode_or_type=mp_dtype, size_or_initializer=int(np.prod(np_array.shape)))
    globals()[name] = mp_array
    shared_np_array = rad_get_shared_array(name, shape=np_array.shape)
    shared_np_array[:] = np_array
    mp_array_shape = Array(typecode_or_type='l', size_or_initializer=len(np_array.shape))
    globals()[name + '_shape'] = mp_array_shape
    shared_np_array = rad_get_shared_array(name + '_shape')
    shared_np_array[:] = np_array.shape
    
class H5_rad():
    """CGS unit system"""
    def __init__(self,pat,dt,maxpart,Nangsx,Nangsy,Nfreqs,fmax,bandwidth,detdist,nep,thxlim=None,thylim=None):
        self.pat = pat                  # h5 files directory path
        self.dt = dt                    # timestep between consecutive h5 files [s]
        self.maxpart = maxpart          # maximum desired number of analized particles
        self.Nangsx = Nangsx
        self.Nangsy = Nangsy
        self.Nfreqs = Nfreqs
        self.fmax = fmax                #maximum frequency for FT [Hz]      (sets number of interpolated timesteps)
        self.bandwidth = bandwidth
        self.detdist = detdist
        self.nep = nep                  #plasma density [cm^-3]
        self.thxlim = thxlim
        self.thylim = thylim
        
        """extract files, order data and perform radiation analysis"""
        self.extract_files(pat,dt,maxpart)
        self.sort_data()
        self.share_arrays()
        self.div()
        self.rad()
        
    def extract_files(self,pat,dt,maxpart):    
        files           = []
        for file in os.listdir(pat):
            if file.endswith('.h5'):
                files.append(file)
        files           = sorted(files)    
        zpl             = []
        xpl             = []
        IDl             = []
        wtl             = []
        Xl              = []
        Pl              = []
        El              = []
        Bl              = []
        self.supparr = np.zeros(12)                                                       
    
        """preliminary ID check"""
        for num,filenam in enumerate(files):
            f           = h5py.File(pat+'/'+filenam, "r")
            fdatagroup  = f['data'][list(f['data'].keys())[0]]                       
            ele         = fdatagroup['particles/electrons']                                 
            ID          = ele['id'][()]
            IDl.append(ID)
    
            print(num)
        """final beam total charge and energy"""
        wt0             = ele['weighting'][()]                                              #weights
        PX0             = ele['momentum']['x'][()]                                          #moments [beta*gamma] of each particle inside macroparticle
        PY0             = ele['momentum']['y'][()]                                  
        PZ0             = ele['momentum']['z'][()]    
        P0              = np.sqrt(PX0**2 + PY0**2 + PZ0**2)
        charge          = np.nansum(wt0)*1.602e-19*1e12                                     #[pC]
        Eave            = np.nansum(P0*wt0*me*c**2)/np.nansum(wt0)                          #[erg]
        Espread         = np.sqrt(np.nansum((P0*me*c**2 - Eave)**2*wt0)/np.nansum(wt0))     #[erg]
        Eave            = Eave*6.24*1e11/1e6                                                #[MeV]
        Espread         = Espread*6.24*1e11/1e6                                             #[MeV] 
        lIDl            = len(IDl)
        t               = np.arange(lIDl)*dt
        self.t          = t
        IDrange         = np.sort(np.unique(np.concatenate(IDl)))                           # total extension of ID values 
        self.IDrange    = IDrange
        IDsize          = len(IDrange)
        self.supparr[0] = IDsize
        nmacro          = max(1,IDsize/maxpart)
        self.nmacro     = nmacro
        if nmacro != 1:    
            IDrange     = np.sort(np.random.choice(IDrange, size=maxpart,replace=False))
        IDsize          = len(IDrange)      
        self.supparr[1] = IDsize
        self.supparr[2] = nmacro
        self.supparr[3] = charge
        self.supparr[4] = Eave
        self.supparr[5] = Espread
    
        """temporary arrays to be filled and ordered"""
        num             += 1
        self.Xtemp      = np.zeros((num,IDsize,3))                  #[timestep,ID,coord]
        self.Ptemp      = np.zeros((num,IDsize,3))                  #[timestep,ID,coord]
        self.Etemp      = np.zeros((num,IDsize,3))                  #[timestep,ID,coord]
        self.Btemp      = np.zeros((num,IDsize,3))                  #[timestep,ID,coord]
        self.wttemp     = np.zeros((num,IDsize))                    #[timestep,ID,coord]
        self.IDtemp     = np.zeros((num,IDsize))    
        self.maxs,self.mins = 0,np.inf                              #max and min transverse rms size
        for numm,filename in enumerate(files):    
            f           = h5py.File(pat+'/'+filename, "r")
            fdatagroup  = f['data'][list(f['data'].keys())[0]]                       
            ele         = fdatagroup['particles/electrons']                          
            
            ID          = ele['id'][()]
            whID        = np.in1d(ID,IDrange)
            ID          = ID[whID]
            X0          = ele['position']['x'][()][whID]*1e-4#                               #positions     [cm]
            Y0          = ele['position']['y'][()][whID]*1e-4#                               #positions     [cm]
            Z0          = ele['position']['z'][()][whID]*1e-4#                               #positions     [cm]
            PX0         = ele['momentum']['x'][()][whID]                                     #moments       [beta*gamma]
            PY0         = ele['momentum']['y'][()][whID]                                     #moments       [beta*gamma]
            PZ0         = ele['momentum']['z'][()][whID]                                     #moments       [beta*gamma]
            EX0         = ele['fields']['E']['x'][()][whID]*1e-4/2.9979                      #Efield        [statVolt/cm]
            EY0         = ele['fields']['E']['y'][()][whID]*1e-4/2.9979
            EZ0         = ele['fields']['E']['z'][()][whID]*1e-4/2.9979
            BX0         = ele['fields']['B']['x'][()][whID]*1e4                              #Bfield        [Gauss]
            BY0         = ele['fields']['B']['y'][()][whID]*1e4
            BZ0         = ele['fields']['B']['z'][()][whID]*1e4
            wt          = ele['weighting'][()][whID]                                         #weights
            sortarr     = np.c_[ID,X0,Y0,Z0,PX0,PY0,PZ0,EX0,EY0,EZ0,BX0,BY0,BZ0,wt]
            
            sortarr     = sortarr[sortarr[:,0].argsort()]                                    #sort by ID
            
            ID          = sortarr[:,0]
            X0          = sortarr[:,1]
            Y0          = sortarr[:,2]
            Z0          = sortarr[:,3]
            PX0         = sortarr[:,4]
            PY0         = sortarr[:,5]
            PZ0         = sortarr[:,6]
            EX0         = sortarr[:,7]
            EY0         = sortarr[:,8]
            EZ0         = sortarr[:,9]
            BX0         = sortarr[:,10]
            BY0         = sortarr[:,11]
            BZ0         = sortarr[:,12]
            wt          = sortarr[:,13]
    
            lIDi        = len(ID)
    
            X           = np.stack([X0,Y0,Z0]).transpose()
            P           = np.stack([PX0,PY0,PZ0]).transpose()
            E           = np.stack([EX0,EY0,EZ0]).transpose()
            B           = np.stack([BX0,BY0,BZ0]).transpose()
                
            self.Xtemp[numm,:lIDi,:] = X
            self.Ptemp[numm,:lIDi,:] = P
            self.Etemp[numm,:lIDi,:] = E
            self.Btemp[numm,:lIDi,:] = B
            self.wttemp[numm,:lIDi] = wt
            self.IDtemp[numm,:lIDi] = ID
    
            wtl.append(list(wt))
            Xl.append(X)
            Pl.append(P)
            El.append(E)
            Bl.append(B)
            pmodu       = np.linalg.norm(P,axis=1)
            minp        = np.nanmin(pmodu)
            maxp        = np.nanmax(pmodu)
            
            """transverse rms sizes"""
            avex,sigx   = wtstd(X0, wt)
            avey,sigy   = wtstd(Y0, wt)
            sig         = (sigx + sigy)/2
            if sig > self.maxs:
                self.maxs   = sig
                self.maxsx  = sigx*10    #[mm]
                self.maxsy  = sigy*10    #[mm]
            if sig < self.mins:
                self.mins   = sig
                self.minsx  = sigx*10    #[mm]
                self.minsy  = sigy*10    #[mm]        
            print(pmodu.shape)
            try:
                    print(numm, ID.shape, "min p = ", minp, "max p = ", maxp)
            except:
                    print(numm, ID.shape)
            self.lIDl   = lIDl
            self.IDsize = IDsize
                    
    def sort_data(self):    
        lIDl            = self.lIDl
        IDsize          = self.IDsize
        self.Xfin       = np.zeros((lIDl,IDsize,3))*np.nan                  #[timestep,ID,coord]
        self.Pfin       = np.zeros((lIDl,IDsize,3))*np.nan                  #[timestep,ID,coord]
        self.Efin       = np.zeros((lIDl,IDsize,3))*np.nan                  #[timestep,ID,coord]
        self.Bfin       = np.zeros((lIDl,IDsize,3))*np.nan                  #[timestep,ID,coord]
        self.wtfin      = np.zeros((lIDl,IDsize))*np.nan                    #[timestep,ID,coord]
        
        checkarr        = np.zeros(lIDl).astype(int)                        #0th ID 
        indarr          = np.arange(lIDl)                                   #time indexes
        
        for i in range(IDsize):
            IDc         = self.IDtemp[indarr,checkarr]
            mask        = IDc - self.IDrange[i]
            mask1       = abs(np.sign(mask)).astype(int)
            mask2       = 1 - mask1                                         # 1 only where local ID equals ith ID
            indxrow     = (mask2*(indarr+1))-1                              # +1 so that this array is zero only where IDs don't match, -1 to then set missing IDs to -1
            dove        = (indxrow != -1)
            indxrow     = indxrow[dove]
            indxcol     = np.ones_like(indxrow)*i
        
            temprow     = indxrow
            tempcol     = checkarr[dove]
        
            checkarr    += mask2
        
            self.wtfin[indxrow,indxcol] = self.wttemp[temprow,tempcol]
            self.Xfin[indxrow,indxcol]  = self.Xtemp[temprow,tempcol]
            self.Pfin[indxrow,indxcol]  = self.Ptemp[temprow,tempcol]
            self.Efin[indxrow,indxcol]  = self.Etemp[temprow,tempcol]
            self.Bfin[indxrow,indxcol]  = self.Btemp[temprow,tempcol]
            print(i)
            
    def share_arrays(self):
        rad_make_shared_array(self.Xfin[:,:,:], name='my_shared_array_6')   # create shared memory array from numpy array
        self.XfinS = rad_get_shared_array('my_shared_array_6')              # get shared memory array as numpy array
        rad_make_shared_array(self.Pfin[:,:,:], name='my_shared_array_6')  
        self.PfinS = rad_get_shared_array('my_shared_array_6')  
        rad_make_shared_array(self.Efin[:,:,:], name='my_shared_array_6')  
        self.EfinS = rad_get_shared_array('my_shared_array_6') 
        rad_make_shared_array(self.Bfin[:,:,:], name='my_shared_array_6')  
        self.BfinS = rad_get_shared_array('my_shared_array_6') 
        rad_make_shared_array(self.wtfin[:,:], name='my_shared_array_6')  
        self.wtfinS = rad_get_shared_array('my_shared_array_6')  
        rad_make_shared_array(self.t, name='my_shared_array_6')
        self.tS = rad_get_shared_array('my_shared_array_6')
        self.freqs = np.logspace(15, 19, self.Nfreqs)  
        rad_make_shared_array(self.freqs, name='my_shared_array_6')  
        self.freqsS = rad_get_shared_array('my_shared_array_6')
        
    def div(self):
        """radiation divergence extimation"""
        wts         = np.nanmax(self.wtfin,axis=0) #[npart]
        xs          = self.Xfin[:,:,0]
        wnn         = ~np.isnan(xs)

        pmod        = np.linalg.norm(self.Pfin,axis=2)
        gamint      = np.sqrt(1+pmod**2)
        betint      = self.Pfin/gamint[:,:,npn]
        kbs         = np.sqrt(2*np.pi*self.nep*e**2/gamint/me/c**2)

        kbs         = kbs[wnn]    
        gamint      = gamint[wnn]
        xs          = xs[wnn]
        ys          = self.Xfin[:,:,1]
        ys          = ys[wnn]
        rs          = np.sqrt(xs**2 + ys**2)
        bxs         = betint[:,:,0]
        bxs         = bxs[wnn]
        bys         = betint[:,:,1]
        bys         = bys[wnn]
        bs          = np.sqrt(bxs**2 + bys**2)
        r0s         = np.sqrt(rs**2 + bs**2/kbs**2)
        x0s         = np.sqrt(xs**2 + bxs**2/kbs**2)
        y0s         = np.sqrt(ys**2 + bys**2/kbs**2)
        r0MAD       = MAD(r0s)
        x0MAD       = MAD(x0s)
        y0MAD       = MAD(y0s)
        self.x0MAD  = x0MAD
        self.y0MAD  = y0MAD
        print()
        print("std r=",r0MAD)
        print("std x=",x0MAD)
        print("std y=",y0MAD)
        ths         = kbs
        wtg         = gamint**4*kbs**4*r0s**2
        this        = np.nansum(ths*wtg,axis=0)/np.nansum(wtg,axis=0)
        gcoeff      = np.nansum(this*wts)/np.nansum(wts)
        thxrad      = x0MAD*gcoeff
        thyrad      = y0MAD*gcoeff
        
        if self.thxlim==None:
            self.thxlim = thxrad*1.5
        if self.thylim==None:            
            self.thylim = thyrad*1.5
        print()
        print("x rad angle =",thxrad)
        print("y rad angle =",thyrad)
        print()
        
    def rad(self):               
        
        self.angx               = np.linspace(-self.thxlim,self.thxlim,self.Nangsx)
        self.angy               = np.linspace(-self.thylim,self.thylim,self.Nangsy)
        Nangs                   = len(self.angx)*len(self.angy)
        AX,AY                   = np.meshgrid(self.angx,self.angy)
        angs                    = np.stack([AX.flatten(),AY.flatten()]).transpose() 
        Ax,Ay                   = AX.flatten(),AY.flatten()
        A                       = np.sqrt(Ax**2 + Ay**2)
        detZ                    = self.detdist*np.cos(A)
        detX                    = self.detdist*np.sin(Ax)
        detY                    = self.detdist*np.sin(Ay)
        self.FLUXangs           = np.empty((Nangs,self.Nfreqs))*np.nan
        self.dWdO               = np.empty(Nangs)*np.nan
        xO                      = np.stack([detX,detY,detZ]).transpose()
        
        self.supparr[6]         = self.Nangsx
        self.supparr[7]         = self.Nangsy
        self.supparr[8]         = self.minsx
        self.supparr[9]         = self.minsy
        self.supparr[10]        = self.maxsx
        self.supparr[11]        = self.maxsy
        
        """## PARTICLE RANGE ##"""
        a = 0
        b = self.IDsize
        for an in range(Nangs):
            times               = np.zeros(((self.lIDl),(b-a)))             #(times,ID)
            Acoeffs             = np.zeros(((self.lIDl),(b-a),6))           #(times,ID,fieldcomponents)
            Atot                = np.zeros(((self.lIDl),(b-a),3))           #(times,ID,fieldcomponents)
            print()
            print()
            print("######################")
            print("angle {}/{}".format(an+1,Nangs))
            print("######################")
            print()
            print()
            """analisi traiettorie"""
            w                   = np.nanmax(self.wtfin,axis=0)              #all the ordered weigths
        
            """parameters"""
            rad_make_shared_array(xO[an], name='my_shared_array_6')         # create shared memory array from numpy array
            self.xOS            = rad_get_shared_array('my_shared_array_6') # get shared memory array as numpy array
            items               = [i for i in range(self.IDsize)]
            self.ftM            = mp.Array(ctp.c_double, self.Nfreqs*6)     #all frequencies, 3 real parts, 3 imag parts
            
            """parallel ft"""               
            with mp.Pool(initializer=rad_init, initargs=(self.ftM,
                                                         self.XfinS,
                                                         self.PfinS,
                                                         self.EfinS,
                                                         self.BfinS,
                                                         self.wtfinS,
                                                         self.xOS,
                                                         self.freqsS,
                                                         self.tS,)) as pool:
                results         = list(tqdm.tqdm(pool.imap(rad_task, items), total=(b-a)))
            
            fts                 = np.frombuffer(self.ftM.get_obj())
            fts                 = fts.reshape(self.Nfreqs,6)
            ft                  = fts[:,0:3] + 1j*fts[:,3:6]        
            Aampl               = abs(np.linalg.norm(ft,axis=1))
        
            """photon number per solid angle"""
            self.Udens           = 2*Aampl**2                                   #d^I/domedOme (jackson IIedition pg669)
            self.dWdOmefreq      = np.trapz(self.Udens,2*np.pi*self.freqs)      #total energy per unit solid angle, integrated from frequency spectrum
            self.FLUXangs[an]    = self.Udens*self.nmacro
            self.dWdO[an]        = self.dWdOmefreq*self.nmacro
            
    """multiprocessing functions"""
        
    def wtstd(self,values, weights):
        average = np.average(values, weights=weights)
        variance = np.average((values-average)**2, weights=weights)
        return (average, np.sqrt(variance))
    