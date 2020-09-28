import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from LorenzSystem import LorenzSystem


class AnimatedLorenz(LorenzSystem):
    def __init__(self, x0, t_max, dt=0.005, anim_speed=1):
        LorenzSystem.__init__(self, x0, dt)
        self.propagate(t_max)
        self.anim_speed = anim_speed
        self.fig = plt.figure(tight_layout=True, figsize=(15, 10))
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.axs = [self.fig.add_subplot(3, 2, 2*i+2) for i in range(3)]
        # Animation setup
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1,
                                           init_func=self.setup_plot, blit=True,
                                           frames=int(t_max/self.dt/self.anim_speed))
        plt.show()
        # self.writer = animation.writers['ffmpeg']
        # self.writer = self.writer(fps=20, bitrate=1000)
        # self.ani.save('LorenzSystem2.mp4',fps=20,dpi=600)

    def setup_plot(self):
        # Initialize plot
        data = self.sim_data[0, :]
        plot_lims = [[np.min(self.sim_data[:,i+1])-2, np.max(self.sim_data[:,i+1])+2] for i in range(3)]
        self.point = self.ax.scatter(data[1], data[2], data[3],
                                     edgecolor='k', vmin=0, vmax=1)
        self.trajectory, = self.ax.plot3D(data[1], data[2], data[3])
        self.plots = [self.axs[i].plot(data[0], data[i+1]) for i in range(3)]
        self.ax.set_xlim3d(-30, 30)
        self.ax.set_ylim3d(-30, 30)
        self.ax.set_zlim3d(0, 40)
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        self.ax.set_zlabel("X3")
        [self.axs[i].set_xlim(0, self.sim_data[-1:, 0]) for i in range(3)]
        [self.axs[i].set_ylim(plot_lims[i][0], plot_lims[i][1]) for i in range(3)]
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.point, self.trajectory, *[plot[0] for plot in self.plots]

    def update(self, i):
        data = self.sim_data[:i*self.anim_speed, :]
        # Update the scatter-plot data
        # self.point._offsets3d = (data[-1:, 1], data[-1:, 2], data[-1:, 3])
        # Update trajectory
        self.trajectory.set_data(data[:, 1:3].T)
        self.trajectory.set_3d_properties(data[:, 3].T)
        # Update plots
        [self.plots[ii][0].set_data(data[:, 0].T, data[:, ii+1].T) for ii in range(3)]
        # Pan the camera
        # t = data[-1:, 0]
        # self.ax.view_init(azim=-45 + 30 * np.sin(t / 2), elev=20 + 15 * np.sin(t * 4))
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.trajectory, self.point, *[plot[0] for plot in self.plots]

anim = AnimatedLorenz([-10,-15,50], 15, anim_speed=5)
anim = AnimatedLorenz([-10,-15,50.1], 15, anim_speed=5)
anim = AnimatedLorenz([-10,-15,49.9], 15, anim_speed=5)

