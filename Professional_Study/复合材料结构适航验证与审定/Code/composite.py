import numpy as np 

class layer:
    def __init__(self,name='boron_epoxy'):
        """
        four kinds of material already in this package:
        'glass_epoxy', 'boron_epoxy', 
        'graphite_epoxy' and 'aramid_epoxy' 
        use layer('glass_epoxy') to create a layer made by glass_epoxy
        """
        if name=='glass_epoxy':
            EL,ET,vLT,GLT=54.8e9,18.3e9,0.25,9.1e9 
            S11=1/EL
            S22=1/ET
            S12=-vLT/EL
            S66=1/GLT
            self.S=np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
            self.Q=np.linalg.inv(self.S)
            self.S_=self.S
            self.Q_=self.Q
            pass
        elif name=='boron_epoxy':
            EL,ET,vLT,GLT=211e9,21.1e9,0.30,7.0e9
            S11=1/EL
            S22=1/ET
            S12=-vLT/EL
            S66=1/GLT
            self.S=np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
            self.Q=np.linalg.inv(self.S)
            self.S_=self.S
            self.Q_=self.Q
            pass
        elif name == 'graphite_epoxy':
            EL,ET,vLT,GLT=211e9,5.3e9,0.25,2.6e9
            S11=1/EL
            S22=1/ET
            S12=-vLT/EL
            S66=1/GLT
            self.S=np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
            self.Q=np.linalg.inv(self.S)
            self.S_=self.S
            self.Q_=self.Q
            pass
        elif name == 'aramid_epoxy':
            EL,ET,vLT,GLT=77e9,5.6e9,0.34,2.1e9
            S11=1/EL
            S22=1/ET
            S12=-vLT/EL
            S66=1/GLT
            self.S=np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
            self.Q=np.linalg.inv(self.S)
            self.S_=self.S
            self.Q_=self.Q
            pass
        else:
            pass
        pass
        self.name=name
        self.angle=0  # default angle = 0
        self.t=0.125e-3  # default t = 0.125mm
        pass

    def set_by_E(self,EL,ET,vLT,GLT):
        """
        input: EL,ET,vLT,GLT  GPa
        """
        EL,ET,vLT,GLT=EL*1e9,ET*1e9,vLT,GLT*1e9
        S11=1/EL
        S22=1/ET
        S12=-vLT/EL
        S66=1/GLT
        self.S=np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
        self.Q=np.linalg.inv(self.S)
        self.S_=self.S
        self.Q_=self.Q
        pass

    def set_by_Q(self,Q11,Q22,Q12,Q66):
        """
        input: Q11,Q22,Q12,Q66  GPa
        """
        Q11,Q22,Q12,Q66=Q11*1e9,Q22*1e9,Q12*1e9,Q66*1e9
        self.Q=np.array([[Q11,Q12,0],[Q12,Q22,0],[0,0,Q66]])
        self.S=np.linalg.inv(self.Q)
        self.Q_=self.Q
        self.S_=self.S
        pass

    def set_by_S(self,S11,S22,S12,S66):
        """
        input: Q11,Q22,Q12,Q66  GPa
        """
        S11,S22,S12,S66=S11*1e-9,S22*1e-9,S12*1e-9,S66*1e-9
        self.S=np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
        self.Q=np.linalg.inv(self.S)
        self.S_=self.S
        self.Q_=self.Q 
        pass

    def set_angle_degree(self,angle):
        self.angle=angle/180*np.pi 
        self.__Q2Q_()
        self.__S2S_()
        pass

    def set_angle_rad(self,angle):
        self.angle=angle
        self.__Q2Q_()
        self.__S2S_()
        pass

    def __Q2Q_(self):
        angle=self.angle
        m,n=np.cos(angle),np.sin(angle)
        T=np.array([[m*m,n*n,2*m*n],[n*n,m*m,-2*m*n],[-m*n,m*n,m*m-n*n]])
        T_inv=np.linalg.inv(T)
        T_inv_trans = np.transpose(T_inv)
        Q=self.Q
        Q_ = np.dot(np.dot(T_inv,Q),T_inv_trans)
        self.Q_= Q_
        pass

    def __S2S_(self):
        angle=self.angle
        m,n=np.cos(angle),np.sin(angle)
        T=np.array([[m*m,n*n,2*m*n],[n*n,m*m,-2*m*n],[-m*n,m*n,m*m-n*n]])
        T_inv=np.linalg.inv(T)
        T_trans=np.transpose(T)
        S=self.S
        S_ = np.dot(np.dot(T_trans,S),T)
        self.S_=S_
        pass

    def set_t(self,t):
        self.t=t
        pass

    def set_name(self,name):
        self.name=name
        pass

    def get_status(self):
        print('name:\t'+self.name)
        print('t:\t',self.t)
        print('angle:\t',self.angle)
        print('Q:\t',self.Q)
        print('S:\t',self.S)
        print('Q_:\t',self.Q_)
        print('S_:\t',self.S_)
        pass
    
class compound:
    def __init__(self,layer_list,status='S'):
        if status=='S':
            layout=np.array(layer_list)
            reverse=layout[::-1]
            self.layout=np.append(layout,reverse)
            pass
        else:
            self.layout=np.array(layer_list)
            pass
        self.__get_t()
        self.__get_angle()
        self.h=np.sum(self.t)
        self.__ABDcal()
        self.__abdcal()
        pass

    def __get_t(self):
        t=np.array([])
        for item in self.layout:
            t=np.append(t,item.t)
            self.t=t
            pass
        pass

    def __get_angle(self):
        angle=np.array([])
        for item in self.layout:
            angle=np.append(angle,item.angle)
            self.angle=angle
            pass
        pass

    def __ABDcal(self):
        h_temp=0
        A,B,D=0,0,0
        for item in self.layout:
            if h_temp<self.h:
                bottom=h_temp-self.h/2
                head=bottom+item.t
                A=A+item.Q_*np.abs(head-bottom)
                B=B+item.Q_*(head**2-bottom**2)/2
                D=D+item.Q_*np.abs(head**3-bottom**3)/3
                h_temp=h_temp+item.t 
                pass
            else:
                break
            pass
        self.A=A
        self.B=B
        self.D=D 
        pass

    def __abdcal(self):
        A,B,D=self.A,self.B,self.D
        B_=-np.dot(np.linalg.inv(A),B)
        C_=np.dot(B,np.linalg.inv(A))
        D_=D-np.dot(np.dot(B,np.linalg.inv(A)),B)
        self.a=np.linalg.inv(A)-np.dot(np.dot(B_,np.linalg.inv(D_)),C_)
        self.b=np.dot(B_,np.linalg.inv(D_))
        self.d=np.linalg.inv(D_)
        pass

    def strain_cal(self,Nx=0,Ny=0,Nxy=0,Mx=0,My=0,Mxy=0,outermost=False):
        f=np.array([[Nx,Ny,Nxy,Mx,My,Mxy]]).T
        a,b,d=self.a,self.b,self.d
        abd=np.r_[np.c_[a,b],np.c_[b,d]]
        eps_kappa_list=np.dot(abd,f)
        self.eps_kappa_list=eps_kappa_list
        self.eps0_x=eps_kappa_list[0]
        self.eps0_y=eps_kappa_list[1]
        self.eps0_xy=eps_kappa_list[2]
        self.kappa_x=eps_kappa_list[3]
        self.kappa_y=eps_kappa_list[4]
        self.kappa_xy=eps_kappa_list[5]
        if outermost==True:
            eps_x=self.eps0_x+self.h*self.kappa_x/2
            eps_y=self.eps0_y+self.h*self.kappa_y/2
            eps_xy=self.eps0_xy+self.h*self.kappa_xy/2
            return np.array([[eps_x,eps_y,eps_xy]]).T
        else:
            return eps_kappa_list
        pass
    pass
