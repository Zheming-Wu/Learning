# Author: Wu Zheming


import math


class RobotSystem:

    def __init__(self, initial_x, initial_y, initial_angle):
        """
        :param initial_x: The initial X position
        :param initial_y: The initial Y position
        :param initial_angle: The initial Angle between the robot and the X direction. The counterclockwise direction is positive
        """
        self.x = initial_x
        self.y = initial_y
        self.angle = initial_angle

        self.diesel_engine_status = False
        self.power_switch_status = False
        self.time = 0

        print('The position is ({}, {}), the angle is {} degree, the time is {}'.format(self.x, self.y, self.angle, self.time))
        pass

    def diesel_engine(self, status=False):
        """
        :param status: 'True' for ON device and 'False' for OFF device
        :return: void
        """
        self.diesel_engine_status = status
        if status:
            print('The diesel engine is ON')
        elif not status:
            print('The diesel engine is OFF')
        pass

    def power_switch(self, status=False, check_status=False):
        """
        :param status: 'True' for ON device and 'False' for OFF device
        :return: void
        """
        self.power_switch_status = status
        if status:
            print('The power switch is ON')
        elif not status:
            print('The power switch is OFF')

        if check_status:
            self.diesel_engine(self.diesel_engine_status)
            self.power_switch(self.power_switch_status)
        pass

    def robot(self, forward_velocity, forward_time, rotate_velocity, rotate_time, stop_time):
        """
        :param forward_velocity: The velocity at which the robot moves forward
        :param forward_time: The time at which the robot moves forward
        :param rotate_velocity: The speed at which the robot rotates counterclockwise
        :param rotate_time: The time at which the robot rotates counterclockwise
        :param stop_time: The time at which the robot stop
        :return:
        """
        if not self.diesel_engine_status:
            print('The diesel engine is OFF')
            print('The position is ({}, {}), the angle is {} degree, the time is {}'.format(self.x, self.y, self.angle, self.time))
        elif not self.power_switch_status:
            print('The power switch is OFF')
            print('The position is ({}, {}), the angle is {} degree, the time is {}'.format(self.x, self.y, self.angle, self.time))
        else:
            print('The diesel engine & power switch is ON')
            angle_plus = rotate_velocity * rotate_time
            self.angle = self.angle + angle_plus
            x_plus = forward_velocity * forward_time * math.cos(self.angle / 180 * math.pi)
            y_plus = forward_velocity * forward_time * math.sin(self.angle / 180 * math.pi)
            self.x = self.x + x_plus
            self.y = self.y + y_plus
            self.time = self.time + forward_time + rotate_time + stop_time
            print('The position is ({}, {}), the angle is {} degree, the time is {}'.format(self.x, self.y, self.angle, self.time))
        pass


if __name__ == '__main__':

    robot1 = RobotSystem(0, 0, 0)
    robot1.robot(0, 0, 5, 9, 0)
    robot1.power_switch(False, check_status=True)
    robot1.diesel_engine(True)
    robot1.power_switch(True, check_status=False)
    robot1.robot(0, 0, 5, 9, 0)
    robot1.robot(2, 1, 0, 0, 1)
