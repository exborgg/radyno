# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:11:24 2024

@author: Andrea
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c,m_e,e,mu_0,epsilon_0
from scipy.integrate import solve_ivp
import copy
from .rad_utils import rad_parall,rad_no_parall
from .segment_utils import segment
from .fields_utils import field_dict,spatial_field_ramp

npn = np.newaxis

class beamline():
    def __init__(self):
        self.segments = {}              # empty dictionary to add beamline devices and drifts; devices will be saverd by names
        self.add_segment("drift",       # first not executed drift to obtain field interface with first real segment
                         "start_drift",
                         l=1)
        self.color_dict = {"drift":         "gray",
                           "undulator":     "yellow",
                           "ion channel":   "green",
                           "CBM":           "blue",
                           "ABP":           "red"}
    
    def add_beam(self,beam,istart=0):
        """
        Function add a beam to the beamline in a given segment position. 
        
        """
        self.istart = istart+1  # index of the starting beamline item
        self.beam = beam
        
    def add_segment(self,device,name,**kwargs):
        """
        Function to extend beamline adding segments. 
        Kwargs should include segment specs and length, 
        and will be updated with additional and derived vsariables.
        Then they will be stored in a beamline dict, together with segment name and type.
        
        """
        
        """create segment object and update segments dictionary"""
        seg = segment(device,name,**kwargs)
        d = {name:seg}
        self.segments.update(d)
        
        """update segments name list"""
        self.slist = list(self.segments.keys())
    
    def link_fields(self,seg1,seg2,z0):
        """
        Gives spatial connection between two field types.
        
        dname1,dname2: strings, names of the two devices to link
        """
        if seg1 is not None:
            """left and right bend use the same magnetic field (ref sys swapping): 
                going from left to right bend should simulate the real field change"""
            swapCBMfield = 1
            if seg1.kind=="CBM":
                if seg2.kind=="CBM" and seg1.kwargs["lr"]!=seg2.kwargs["lr"]:
                    swapCBMfield = -1
                if seg2.kind!="CBM" and seg1.kwargs["lr"]=="L":
                    swapCBMfield = -1
            
            """add offset for correct undulator phase injection"""
            z01 = 0
            z02 = 0
            if seg1.kind=="undulator":
                z01 = z0
            if seg2.kind=="undulator":
                z02 = z0
                
            """retrieve separate fields"""
            field1 = field_dict[seg1.kind]
            field2 = field_dict[seg2.kind]
            
            """define combined fields function"""
            def linked_field(x,y,z):
                """fading field 1"""
                f1 = field1(x,y,z - z01,**seg1.kwargs)*spatial_field_ramp(z,z0,updown=-1)*swapCBMfield
                """growing field 2"""
                f2 = field2(x,y,z - z02,**seg2.kwargs)*spatial_field_ramp(z,z0,updown=1)
                return f1 + f2
        
        else:
            z02 = 0
            if seg2.kind=="undulator":
                z02 = z0
            field2 = field_dict[seg2.kind]
            def linked_field(x,y,z):
                f2 = field2(x,y,z - z02,**seg2.kwargs)
                return f2
        
        return linked_field
    
    def match(self):
        """overwrites original transverse rms sizes and matches the beam over given emittances"""

    def IVP(self,t,y,npart):   
        
        """reshape variables array"""
        U = y.reshape(npart, 6)    
        x = U[:, 0]
        y = U[:, 1]
        z = U[:, 2]
        bx = U[:, 3]
        by = U[:, 4]
        bz = U[:, 5]
        
        # if np.amin(z)<0:
        #     print(np.amin(z))
        #     print(np.amax(z))
        #     print()

        """calculate fields"""
        Ex,Ey,Ez,Bx,By,Bz = self.field(x,y,z)
        
        """tempoal smoothing"""
        #smooth = self.temp_field_ramp(t,self.ti,self.te)
        
        """calculate force"""
        g = 1/np.sqrt(1-(bx**2 + by**2 + bz**2))
        Fx = -e*(Ex + c*(by*Bz - bz*By))#*smooth 
        Fy = -e*(Ey + c*(bz*Bx - bx*Bz))#*smooth
        Fz = -e*(Ez + c*(bx*By - by*Bx))#*smooth
        bdF = bx*Fx + by*Fy + bz*Fz
        
        """calculate acceleration"""
        ax = 1/m_e/g * (Fx - bdF*bx)
        ay = 1/m_e/g * (Fy - bdF*by)
        az = 1/m_e/g * (Fz - bdF*bz)
        
        """derivatives"""
        dxdt = bx*c
        dydt = by*c
        dzdt = bz*c
        dbxdt = ax/c
        dbydt = ay/c
        dbzdt = az/c
        
        """reshape and flatten variables array"""
        out = np.stack([dxdt, dydt, dzdt, dbxdt, dbydt, dbzdt]).transpose().flatten()
        return out
    
    def dyn(self,u0,te,teval,dtmax,seg_curr,event):
        """
        Solves inital value problem for beam dynamics
        """
        def threshold(t,y,npart):
            """
            Event to stop fine Runge-Kutta beam dynamics integration once the beam
            has fully passed from previous to current segment
            """
            U = y.reshape(npart, 6)   
            z = U[:, 2]
            return np.amin(z)
        threshold.terminal = True
        threshold.direction = 1
        u0 = np.stack(u0).transpose()
        u0 = u0.flatten()
        if event==False:
            u = solve_ivp(self.IVP, 
                          [0, te], 
                          u0, 
                          method='RK45',
                          t_eval=teval, 
                          max_step=dtmax, 
                          args=(self.beam.npart,))
        else:
            u = solve_ivp(self.IVP, 
                          [0, te], 
                          u0, 
                          method='RK45',
                          t_eval=teval, 
                          max_step=dtmax, 
                          args=(self.beam.npart,),
                          events=threshold)
        x = u.y[slice(0, 0+6*self.beam.npart, 6)].transpose()
        y = u.y[slice(1, 0+6*self.beam.npart, 6)].transpose()
        z = u.y[slice(2, 0+6*self.beam.npart, 6)].transpose()
        bx = u.y[slice(3, 0+6*self.beam.npart, 6)].transpose()
        by = u.y[slice(4, 0+6*self.beam.npart, 6)].transpose()
        bz = u.y[slice(5, 0+6*self.beam.npart, 6)].transpose()
        return x,y,z,bx,by,bz
    
    def run(self,dt,dtmax):
        """ 
        dt: output sampling time [s]
        dtmax: maximum allowed timestep for Runge-Kutta integration
            
        self.cff is used to effectively change the direction of beam bending, 
        making it easy to rectify trajectories in order to prepare beam for 
        next segment: for a left bend, first x particles coordinates and speeds 
        are flipped, then the beam is propagated "backwards" and finally x 
        coordinates and speeds are flipped back
        """
        
        """check existence of at least one segment"""
        assert len(self.segments)>0
        
        """preliminary loop over segments for empty arrays initialization"""
        self.rangelist = []                             # timestep ranges list for varisables postprocessing
        for sname in self.slist[self.istart:]:
            """extract segment"""
            seg = self.segments[sname]
            """set segment timesteps number and their value"""
            seg.add_timesteps(dt)
            """once the segment has a simulation timestep, empty variables arrays are initialized"""
            seg.init_arrays(self.beam.npart)
        
        """init integration variables"""
        x0 = copy.deepcopy(self.beam.pos0[:,0])
        y0 = copy.deepcopy(self.beam.pos0[:,1])
        z0 = copy.deepcopy(self.beam.pos0[:,2])
        bx0 = copy.deepcopy(self.beam.bet0[:,0])
        by0 = copy.deepcopy(self.beam.bet0[:,1])
        bz0 = copy.deepcopy(self.beam.bet0[:,2])
        
        """counter for timesteps"""
        nptcount = 0
        for i,sname in enumerate(self.slist[self.istart:]):
            """previous and current segment specs"""
            if i == 0:
                """always choose 'start drift' as first beamline element"""
                seg_prev = self.segments[self.slist[i]]
                i += self.istart-1
            else:
                i += self.istart-1
                seg_prev = self.segments[self.slist[i]]
            seg_curr = self.segments[self.slist[i+1]]

            """extract current segment specs"""
            npt = seg_curr.npts
            
            """extract segments names"""
            s0 = self.slist[i]
            s1 = self.slist[i+1]
            
            """push beam below z=0 line and set longitudinal position where to 
            join fields: slightly forward first beam particle"""
            z0 -= np.amax(z0)
            z_join = 0
            
            """change reference system in case of bending direction change
            and add r_bend offset (that is automatically zero for straigth elements)"""
            if seg_curr.kwargs["cff"]!=seg_prev.kwargs["cff"]:
                x0 *= -1
                bx0 *= -1
            x0 += seg_curr.kwargs["r_bend"]
            
            if self.beam.npart>1:
                """set segment field joining previous and current segment fields:
                   junction is made at first beam particle position"""
                self.field = self.link_fields(seg_prev,seg_curr,z_join)
                print()
                print(s0,"->",s1)
                
                """solve IVP along previous and current segment junction with small
                enough timestep"""
                stdz = np.std(z0)
                te_j = 10*stdz/c
                teval_j = np.linspace(0, te_j, npt, endpoint=True)
                dtmax_j = stdz/c/100
                print()
                print("dt_junc:",dtmax_j)
                print("dt_seg:",dtmax)
                print()
                u0 = (x0, y0, z0, bx0, by0, bz0)
                x,y,z,bx,by,bz = self.dyn(u0,te_j,teval_j,dtmax_j,seg_curr,event=True)
                x0 = x[-1]
                y0 = y[-1]
                z0 = z[-1]
                bx0 = bx[-1]
                by0 = by[-1]
                bz0 = bz[-1]
                
                b = np.sqrt(bx[0]**2 + by[0]**2 + bz[0]**2)
                g = 1/np.sqrt(1-b**2)
                b0 = np.sqrt(bx0**2 + by0**2 + bz0**2)
                g0 = 1/np.sqrt(1-b0**2)
            
            """set current segment field"""
            self.field = self.link_fields(None,seg_curr,z_join)
            print()
            print(s1)
            
            """solve IVP in current segment"""
            te = seg_curr.t_end
            teval = np.linspace(0, te, npt, endpoint=True)
            u0 = (x0, y0, z0, bx0, by0, bz0)
            x,y,z,bx,by,bz = self.dyn(u0,te,teval,dtmax,seg_curr,event=False)
            
            """define segment index range for array update"""
            rg = range(nptcount,nptcount+npt)
            self.rangelist.append(rg)
            
            """extend dynamic variables arrays: not yet rectified, so that full information is available for radiation evaluation"""
            seg_curr.pos[:,:,0] = x
            seg_curr.pos[:,:,1] = y
            seg_curr.pos[:,:,2] = z
            seg_curr.bet[:,:,0] = bx
            seg_curr.bet[:,:,1] = by
            seg_curr.bet[:,:,2] = bz
            
            """extend fields seen from particles during evolution: not rectified"""
            fields = self.field(x,y,z)
            seg_curr.E_field[:,:,0] = fields[0]
            seg_curr.E_field[:,:,1] = fields[1]
            seg_curr.E_field[:,:,2] = fields[2]
            seg_curr.B_field[:,:,0] = fields[3]
            seg_curr.B_field[:,:,1] = fields[4]
            seg_curr.B_field[:,:,2] = fields[5]
            
            """rectify variables for bending elements"""
            if seg_curr.kind == "CBM" or seg_curr.kind == "ABP":
                xr,yr,zr,bxr,byr,bzr = self.rectify(x,y,z,bx,by,bz,seg_curr.kwargs["r_bend"],0)
                """rms variables: after rectifying, quantities relative to reference trajectory"""
                sx = np.std(xr,axis=1)
                sy = np.std(yr,axis=1)
                sz = np.std(zr,axis=1)
                mx = np.average(xr,axis=1)
                my = np.average(yr,axis=1)
                mz = np.average(zr,axis=1)
            else:
                sx = np.std(x,axis=1)
                sy = np.std(y,axis=1)
                sz = np.std(z,axis=1)
                mx = np.average(x,axis=1)
                my = np.average(y,axis=1)
                mz = np.average(z,axis=1)
            
            """rotate last particle positions/velocity in case of bending elements"""
            if seg_curr.kind == "CBM" or seg_curr.kind == "ABP":
                r = np.sqrt(x**2 + z**2)
                be = np.sqrt(bx**2 + bz**2)
                thp = np.arctan2(z, x)
                thbe = np.arctan2(bx, bz)
                """particle head position"""
                thmax = np.amax(thp)
                """rotate angles"""
                thp -= thmax
                thbe += thmax
                """update positions and velocities with rotated angles"""
                x = r*np.cos(thp) - seg_curr.kwargs["r_bend"]
                z = r*np.sin(thp)
                bx = be*np.sin(thbe)
                bz = be*np.cos(thbe)
                print("amax z",np.amax(z[-1]))

            """extend rms variables arrays"""
            seg_curr.sigpos[:,0] = sx
            seg_curr.sigpos[:,1] = sy
            seg_curr.sigpos[:,2] = sz
            seg_curr.mupos[:,0] = mx
            seg_curr.mupos[:,1] = my
            seg_curr.mupos[:,2] = mz
            seg_curr.gam[:,:] = 1/np.sqrt(1-np.linalg.norm(seg_curr.bet,axis=2)**2)
            
            """update initial integration variables: they need to be the 
            rectified ones since each segment has its reference system. beam
            longitudinal position is initialized before next segment's z=0 line"""
            x0 = x[-1]
            y0 = y[-1]
            z0 = z[-1]-np.amax(z[-1])
            bx0 = bx[-1]
            by0 = by[-1]
            bz0 = bz[-1]
            
            """update counter"""
            nptcount += npt
    
    def rectify(self,x,y,z,bx,by,bz,r_bend,z0):
        """Rectifies coordinates for bending elements. Axis origin referred to bending radius"""
        r = np.sqrt(x**2 + z**2)
        be = np.sqrt(bx**2 + bz**2)
        thp = np.arctan2(z, x)
        thet = np.average(thp,axis=1)
        thbe = np.arctan2(bx, bz)
        xth = r - r_bend
        zth = r_bend*thp
        bexth = be*np.sin(thbe+thp)
        bezth = be*np.cos(thbe+thp)
        """cut"""
        # maxrth = np.amax(rth, axis=1)
        # where = np.where(maxrth < r_cap*1000)[0]
        # npart = len(where)
        x1 = xth
        y1 = y
        z1 = zth
        bx1 = bexth
        by1 = by
        bz1 = bezth
        return x1,y1,z1,bx1,by1,bz1    
    
    def SI_CGS(self,seg):
        pos_CGS = copy.deepcopy(seg.pos)*1e2
        E_CGS = copy.deepcopy(seg.E_field)*1e-4/2.9979
        B_CGS = copy.deepcopy(seg.B_field)*1e4
        return pos_CGS,E_CGS,B_CGS
    
    def add_detectors(self,segdict):
        """
        Adds radiation detectors to desired beamline segments. 
        segdict:    dictionary with segment names as keys and lists as values. 
                    First list item should be an array of shape (N,3) with N 
                    number of detector points and their 3 spatial coordinates.
                    Second list item should be an array of shape (M) with the 
                    desired spectrum frequencies.
        """
        self.radsegdict = segdict
        for sname in self.radsegdict.keys():
            self.segments[sname].add_detectors(self.radsegdict[sname])
    
    def radiation(self,parall):
        """
        Returns d^2I / dome dOme (jackson IIedition pg669) as a function of 
        radiation frequency for each segment detector point listed in self.radsegdict
        
        """
        for i,sname in enumerate(self.slist[self.istart:]):
            if sname in self.radsegdict.keys():
                print()
                print(sname+" radiation:")
                """current segment specs"""
                i += self.istart-1
                seg_curr = self.segments[self.slist[i+1]]
                te = seg_curr.t_end
                npt = seg_curr.npts
                teval = np.linspace(0, te, npt, endpoint=True)
                """extract detector points and evaluation frequencies"""
                dets,freqs = copy.deepcopy(self.radsegdict[sname])
                """turn variables in CGS"""
                pos_CGS,E_CGS,B_CGS = self.SI_CGS(seg_curr)
                if seg_curr.kind == "CBM" or seg_curr.kind == "ABP":
                    """detectors angles respect to segment bending axis"""
                    thetas = dets[:,0]
                    """detector cartesian position in thetas-rotated reference system (CGS)"""
                    dets = dets[:,1:]*100 
                    dets[:,0] += copy.deepcopy(seg_curr.kwargs["r_bend"])*100 
                    """detectors angles and radia respect to their non-rotated reference system"""
                    thetd = np.arctan2(dets[:,2],dets[:,0])
                    rd = np.sqrt(dets[:,0]**2 + dets[:,2]**2)
                    """detectors angles and position in segment's reference system"""
                    thetd += thetas
                    dets[:,0] = rd*np.cos(thetd)
                    dets[:,2] = rd*np.sin(thetd)
                else:
                    dets *= 100                 
                """calculate radiation"""
                wt = 1
                for d,detpos in enumerate(dets):
                    ft_array = np.zeros((len(freqs),6))
                    print("detector {}".format(d))
                    if parall==False:
                        for ii in range(self.beam.npart):
                            xint = pos_CGS[:,ii,:]
                            bint = copy.deepcopy(seg_curr.bet[:,ii,:])
                            gint = copy.deepcopy(seg_curr.gam[:,ii])
                            Eint = E_CGS[:,ii,:]
                            Bint = B_CGS[:,ii,:]
                            ft_array = rad_no_parall(xint,bint,gint,Eint,Bint,wt,detpos,freqs,teval,ft_array)
                        ft_out = ft_array[:,0:3] + 1j*ft_array[:,3:6]        
                        Aampl = abs(np.linalg.norm(ft_out,axis=1))
                        U = 2*Aampl**2               

                    elif parall==True:
                        xint = pos_CGS
                        bint = copy.deepcopy(seg_curr.bet)
                        gint = copy.deepcopy(seg_curr.gam)
                        Eint = E_CGS
                        Bint = B_CGS
                        ft_out = rad_parall(xint,bint,gint,Eint,Bint,wt,detpos,freqs,teval,ft_array)
                        Aampl = abs(np.linalg.norm(ft_out,axis=1))
                        U = 2*Aampl**2    
                    """update current segmets' radiation array"""
                    seg_curr.U[d] = U
                
    def plot_beam_bendplane(self):
        plt.figure()
        l = [0]
        miny = np.inf
        maxy = -np.inf
        for i,sname in enumerate(self.slist[self.istart:]):
            seg = self.segments[sname]
            rg = self.rangelist[i]
            x,y,z,bx,by,bz = seg.pos[:,:,0],seg.pos[:,:,1],seg.pos[:,:,2],seg.bet[:,:,0],seg.bet[:,:,1],seg.bet[:,:,2]
            """rectify variables for bending elements"""
            if seg.kind == "CBM" or seg.kind == "ABP":
                x,y,z,bx,by,bz = self.rectify(x,y,z,bx,by,bz,seg.kwargs["r_bend"],0)
                x *= seg.kwargs["cff"]
                bx *= seg.kwargs["cff"]
            else:
                pass
            if np.amax(x)>maxy:
                maxy = np.amax(x)
            if np.amin(x)<miny:
                miny = np.amin(x)
            plt.plot(z+sum(l),x,alpha=0.2)
            plt.scatter(z[0]+sum(l),x[0])
            l.append(seg.kwargs["l"])
        
        """plot segment labels with colors from dict in constructor"""
        npt = 10
        h = 0.1*(maxy - miny)
        upline = np.ones(npt)*(miny - h)
        downline = np.ones(npt)*(miny - 2*h)
        for i,sname in enumerate(self.slist[self.istart:]):
            """previous current and following segment specs"""
            seg = self.segments[sname]
            xax = np.linspace(sum(l[:i+1]),sum(l[:i+2]),npt)
            col = self.color_dict[seg.kind]
            if seg.kind == "CBM" or seg.kind == "ABP":
                sname = seg.kwargs["lr"]+" "+sname
            box = plt.fill_between(xax,downline,upline,color=col,alpha=0.3)
            """place segment specs inside box"""
            (x0, y0), (x1, y1) = box.get_paths()[0].get_extents().get_points()
            plt.text((x0 + x1) / 2, (y0 + y1) / 2, sname, ha='center', va='center')
        plt.legend()
            
            