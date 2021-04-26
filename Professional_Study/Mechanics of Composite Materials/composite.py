import numpy as np 

class layer:
    def __init__(self,name='a kind of material'):
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
            pass
        elif name=='HT3/QY8911':
            EL,ET,vLT,GLT=135e9,8.8e9,0.33,4.47e9
            S11=1/EL
            S22=1/ET
            S12=-vLT/EL
            S66=1/GLT
            self.S=np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
            self.Q=np.linalg.inv(self.S)
            pass
        else:
            pass
        pass
        self.name=name
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
        pass

    def set_by_Q(self,Q11,Q22,Q12,Q66):
        """
        input: Q11,Q22,Q12,Q66  GPa
        """
        Q11,Q22,Q12,Q66=Q11*1e9,Q22*1e9,Q12*1e9,Q66*1e9
        self.Q=np.array([[Q11,Q12,0],[Q12,Q22,0],[0,0,Q66]])
        self.S=np.linalg.inv(self.Q)
        pass

    def set_by_S(self,S11,S22,S12,S66):
        """
        input: Q11,Q22,Q12,Q66  GPa
        """
        S11,S22,S12,S66=S11*1e-9,S22*1e-9,S12*1e-9,S66*1e-9
        self.S=np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
        self.Q=np.linalg.inv(self.S)
        pass

