import numpy as np 

class layer:
    def __init__(self,name='HT3/5224'):
        """
        two kinds of material 'HT/5224' and 'HT3/QY8911' already in this package
        use layer('HT3/5224') to create a layer made by HT3/5224
        """
        if name=='HT3/5224':
            EL,ET,vLT,GLT=140e9,8.6e9,0.35,5.0e9 
            S11=1/EL
            S22=1/ET
            S12=-vLT/EL
            S66=1/GLT
            self.S=np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
            self.Q=np.linalg.inv(self.S)
            self.S_=self.S
            self.Q_=self.Q
            pass
        elif name=='HT3/QY8911':
            EL,ET,vLT,GLT=135e9,8.8e9,0.33,4.47e9
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

    '''
    def set_angle_degree(self,theta):
        self.angle=theta/180*np.pi 
        m,n=np.cos(theta),np.sin(theta)
        T=np.array([[m*m,n*n,2*m*n],[n*n,m*m,-2*m*n],[-m*n,m*n,m*m-n*n]])
        T_inv=np.linalg.inv(T)
        Q_=np.dot(np.dot(T_inv,self.Q),T)
        self.Q_=Q_
        pass

    def set_angle_rad(self,theta):
        self.angle=theta
        m,n=np.cos(theta),np.sin(theta)
        T=np.array([[m*m,n*n,2*m*n],[n*n,m*m,-2*m*n],[-m*n,m*n,m*m-n*n]])
        T_inv=np.linalg.inv(T)
        Q_=np.dot(np.dot(T_inv,self.Q),T)
        self.Q_=Q_
        pass
    '''
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
        Q=self.Q
        Q[:,2]=Q[:,2]*2
        Q_=np.dot(np.dot(T_inv,Q),T)
        Q_[:,2]=Q_[:,2]/2
        Q[:,2]=Q[:,2]/2
        self.Q_=Q_
        pass

    def __S2S_(self):
        angle=self.angle
        m,n=np.cos(angle),np.sin(angle)
        T=np.array([[m*m,n*n,2*m*n],[n*n,m*m,-2*m*n],[-m*n,m*n,m*m-n*n]])
        T_inv=np.linalg.inv(T)
        S=self.S
        S[2]=S[2]/2
        S_=np.dot(np.dot(T_inv,S),T)
        S_[2]=S_[2]*2
        S[2]=S[2]*2
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
    