# Author: Wu Zheming
# Data: 2024/04/13


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


gravity = 9.8


class Movement_Simulation:

    def __init__(self, vx0, vy0, t_sim, interval=1.0):
        """
        :param vx0: Initial velocity in the x direction
        :param vy0: Initial velocity in the y direction
        :param t_sim: Simulation duration
        :param interval: Simulation interval (default:1)
        """

        self.t = np.arange(0, t_sim+interval, interval)
        self.vx = vx0 - gravity * self.t
        self.vy = vy0
        self.x = vx0 * self.t - 0.5 * gravity * self.t * self.t
        self.y = vy0 * self.t
        result_dic = {'t': self.t,
                      'x': self.x,
                      'y': self.y,
                      'vx': self.vx,
                      'vy': self.vy}
        self.result = pd.DataFrame(result_dic)
        pass

    def print_result(self):
        print(self.result)
        pass

    def save_result(self, filepath='./result'):
        self.result.to_csv('{}.csv'.format(filepath))
        pass

    def plot_result(self, figure_path=None):

        plt.style.use('default')

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 7), dpi=100)
        lns1 = ax1.plot(self.result["y"], self.result["x"], color='tab:red', label='Simulation_data')

        ax1.set_title('Movement trajectory simulation', fontdict={'fontsize': 24, 'family': 'Times New Roman'})

        ax1.tick_params(labelsize=24)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontname('Times New Roman')

        ax1.set_xlabel('y', fontproperties='Times New Roman', fontsize=24)
        ax1.set_ylabel('x', fontproperties='Times New Roman', fontsize=24)

        plt.show()

        if figure_path:
            fig.savefig('{}.png'.format(figure_path), dpi=300, format='png')


if __name__ == '__main__':

    move1 = Movement_Simulation(vx0=100, vy0=15, t_sim=15, interval=1.0)  # Initialize parameters

    move1.print_result()  # Print result
    move1.save_result(filepath='./result')  # Save result
    move1.plot_result(figure_path='./result')  # Plot result
