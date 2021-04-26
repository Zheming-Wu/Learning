import numpy as np


class layer:    #定义层类
    #定义初始化方法
    def __init__(self,name='A kind of material'):
        if name=='HT3/5224':
            self.E1=140e9
            self.E2=8.6e9
            self.v12=0.35
            self.G23=5.0e9
            self.v21=self.E2*self.v12/self.E1
            self.set_by_basic(self.E1,self.E2,self.v12,self.G23)
            pass
        elif name=='HT3/QY8911':
            self.E1=135e9
            self.E2=8.8e9
            self.v12=0.33
            self.G23=4.47e9
            self.v21=self.E2*self.v12/self.E1
            self.set_by_basic(self.E1,self.E2,self.v12,self.G23)
            pass
        else:
            pass
        self.name=name
        self.t=0
        self.angle=0
        self.hk=0
        pass

    '''给定基本弹性常数的初始化定义的方法'''
    def set_by_basic(self,E1,E2,v12,G23):
        self.E1=E1
        self.E2=E2
        self.v12=v12
        self.G23=G23
        self.v21=self.E2*self.v12/self.E1
        S11=1/self.E1
        S12=-self.v12/self.E1
        S22=1/self.E2
        S66=1/self.G23
        tmp_S=np.array([
            [S11,S12,0],
            [S12,S22,0],
            [0,0,S66]
        ])
        tmp_Q=np.linalg.inv(tmp_S)
        self.Q=tmp_Q
        self.S=tmp_S
        self.set_angle(0)
        pass
    '''给定S或者Q时候的初始化定义方法'''
    def __get_basic(self):
        self.E1=1/self.S[0][0]
        self.E2=1/self.S[1][1]
        self.G23=1/self.S[2][2]
        self.v12=-self.S[1][0]*self.E1
        self.v21=self.E2*self.v12/self.E1
        pass
    def set_by_Q(self,Q11,Q12,Q22,Q66):
        self.Q=np.array([
            [Q11,Q12,0],
            [Q12,Q22,0],
            [0,0,Q66]
        ])
        self.S=np.linalg.inv(self.Q)
        self.__get_basic()
        self.set_angle(0)
        pass
    def set_by_S(self,S11,S12,S22,S66):
        self.S=np.array([
            [S11,S12,0],
            [S12,S22,0],
            [0,0,S66]
        ])
        self.Q=np.linalg.inv(self.S)
        self.__get_basic()
        self.set_angle(0)
        pass

    '''定义设置角度下的情况'''
    def __set_SQ_theta(self):
        Q_=np.array(list(self.Q))
        Q_[2]=Q_[2]*2
        Q_=self.T_inv.dot(Q_.dot(self.T))
        Q_=Q_.T
        Q_[2]=Q_[2]/2
        self.Q_=Q_.T
        self.S_=np.linalg.inv(self.Q_)
        self.Ex=1/self.S_[0][0]
        self.Ey=1/self.S_[1][1]
        self.Gxy=1/self.S_[2][2]
        self.vxy=-self.S_[1][0]*self.Ex
        self.vyx=-self.S_[0][1]*self.Ey
        self.nxy_x=self.S_[2][0]*self.Ex
        self.nxy_y=self.S_[2][1]*self.Ey
        self.nx_xy=self.S_[0][2]*self.Gxy
        self.ny_xy=self.S_[1][2]*self.Gxy
        pass
    
    def set_angle(self,theta):#设置角度deg
        self.angle=theta
        m=np.cos(theta/180*np.pi)
        n=np.sin(theta/180*np.pi)
        T=np.array([
            [m**2,n**2,2*m*n],
            [n**2,m**2,-2*m*n],
            [-m*n,m*n,m**2-n**2]
        ])
        self.T=T
        self.T_inv=np.linalg.inv(T)
        self.__set_SQ_theta()
        pass
    pass

    '''定义层合板状态下的情况'''
    def set_t(self,t):
        self.t=t
        pass
    
    def set_hk(self,hk):
        self.hk=hk
        pass
    
    '''定义打印出来的基本信息'''
    def info(self):
        [print('-',end='-') for i in range(30)]
        print('\n')
        print('name of the compound material is:  '+self.name+'\n')
        for item in ['E1','E2','G23','S','Q']:
            if item=='S':
                print(item+':\n',eval('self.'+item)*1e9,' GPa^-1\n')
                pass
            else:
                print(item+':\n',eval('self.'+item)/1e9,' GPa\n')
                pass
            pass
        print('v12=',self.v12,'\n')
        print('v21=',self.v21,'\n')
        print('place angle=',str(self.angle*180/np.pi),' degree\n')
        print('thickness t=',str(self.t),'\n')
        print('placement hk=',str(self.hk),'\n')
        print('S_:\n',self.S_*1e9,' GPa^-1\n')
        print('Q_:\n',self.Q_/1e9,' GPa\n')
        [print('-',end='-') for i in range(30)]
        print('\n')
        pass



class plate:  #定义板类
    def __init__(self,list_layer,name='this is a laminate'):
        self.layout=list_layer
        self.__get_t()
        self.__get_angle()
        self.H=np.sum(self.t)
        self.Hm=self.H/2
        self.set()
        self.name=name
        pass
    
    def __get_t(self):
        t=list()
        for item in self.layout:
            t.append(item.t)
            pass
        self.t=t
        pass
    def __get_angle(self):
        angle=list()
        for item in self.layout:
            angle.append(item.angle)
            pass
        self.angle=angle
        pass
    
    def set(self):
        A=np.zeros([3,3])
        B=np.zeros([3,3])
        D=np.zeros([3,3])    
        for i in range(len(self.layout)):
            self.layout[i].set_angle(self.angle[i])
            self.layout[i].set_t(self.t[i])
            tmp_t=0
            for j in range(i):
                tmp_t+=self.t[j]
                pass
            tmp_t-=self.t[i]
            tmp_t=self.Hm-tmp_t
            self.layout[i].set_hk(tmp_t)
            tmp=self.layout[i]
            A+=tmp.Q_*(tmp.t)
            B+=1/2*tmp.Q_*(tmp.hk**2-(tmp.hk-tmp.t)**2)
            D+=1/3*tmp.Q_*(tmp.hk**3-(tmp.hk-tmp.t)**3)
            pass
        AB=np.c_[A,B]
        BD=np.c_[B,D]
        NM=np.array([AB,BD])
        self.A=A
        self.B=B
        self.D=D
        self.NM=NM
        pass
    def reset(self):
        self.set()
        pass
    
    def set_angle(self,angle):
        self.angle=angle
        self.reset()
        pass
    def set_t(self,t):
        self.t=t
        self.reset()
        pass
    
    def info(self):
        [print('-',end='-') for i in range(30)]
        print('\n')
        print("the laminate's name is: "+self.name)
        print('the number of layers:',len(self.layout),'\n')
        print("each layer's t is:",self.t,'\n')
        print('total height is:',self.H,'\n')
        print('angles of the layers are:\n',self.angle,'\n')
        print('thickness of the laryers are:\n',self.t,'\n')
        print('the matrix of this laminate is :\n',self.NM/1e9,' GPa/mm^1,2,3\n')
        for item in ['A','B','D']:
            print(item+'=\n',eval('self.'+item)/1e9,'GPa/mm^1,2,3\n')
            pass
        [print('-',end='-') for i in range(30)]
        print('\n')
        pass
    
    def info_detailed(self):
        for i in range(len(self.layout)):
            print('the '+str(i+1)+' layer')
            self.layout[i].info()
            pass
        pass