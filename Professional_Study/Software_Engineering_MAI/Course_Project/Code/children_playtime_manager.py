import unittest


class Whitelist:

    def __init__(self,Table = None):
        self.Table=Table

    def Add(self, AppName):
        pass

    def Delete(self, AppName):
        pass

    pass


class TimeLimit:

    def __init__(self,Time=0):
        self.T=Time

    def SetLimit(self, Time):
            self.T=Time



class Monitor:

    def __init__(self, TL=0, WL=[]):
        self.TL=TL
        self.WL=WL

    def BackgroundMonitor(self):
        pass

    def TimeStatistics(self):
        pass

    def GameTimeCalculation(self):
        return 0

    def OvertimeCheck(self,GameTime=0):
        return 0

    def LockPC(self):
        pass

    def UnlockPC(self):
        pass

    pass


class Application:

    def __init__(self):
        self.WL=Whitelist
        self.TL=TimeLimit

    def SetTime(self,Time):
        self.TL.SetLimit(Time)

    def SetWhitelist(self,Appname):
        self.WL.Add(Appname)

    def Monitor(self):
        M=Monitor(self.TL,self.WL)
        M.BackgroundMonitor()

        #keep running these
        M.TimeStatistics()
        GameTime=M.GameTimeCalculation()
        Ov=M.OvertimeCheck(GameTime)
        if Ov==1:
            M.LockPC()

        pass

    def RemoteLock(self):
        pass

    def RemoteUnlock(self):
        pass

    def Report(self):
        pass

    pass




class System:

    def __init__(self):
        self.App=Application

    def MonitorSwitch(self):
        self.App.Monitor()

    def SetLimit(self, TimePerDay):
        self.App.SetTime(TimePerDay)

    def SetWhitelist(self,Appname):
        self.App.SetWhitelist(Appname)

    def RemoteControl_Lock(self):
        self.App.RemoteLock()

    def RemoteControl_Unlock(self):
        self.App.RemoteUnlock()

    def ShowReport(self):
        self.App.Report()

    pass


class TestMonitor(unittest.TestCase):

    def testOvertimeCheck(self):
        monitor = Monitor()

        self.assertEquals(monitor.OvertimeCheck(), 0)

    def testGameTimeCalculation(self):
        monitor = Monitor()

        self.assertEquals(monitor.GameTimeCalculation(), 0)


if __name__=='__main__':

    unittest.main()
