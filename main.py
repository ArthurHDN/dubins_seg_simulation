from math import cos, sin
import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from PIL import Image

from pydubinsseg import state
from pydubinsseg.segregationcontrol import SegregationControl


class Simulator():
    
    def __init__(self,groups,params,initial_poses):
        number = 0
        self._dubins_control = []
        self._states_array = [ [] for i in range(len(groups)) ]
        for i in range(len(groups)):
            robot = SegregationControl(number, groups[i],state['in circle'],params)
            robot.set_pose2D(initial_poses[i])
            robot.calculate_initial_conditions()
            self._dubins_control.append(robot)
            number = number + 1
        self._n_robots = number - 1
        self._n_groups = max(groups)

    def run(self):
        t_start = 0.0
        t_stop = 100.0
        dt = 0.1
        self._time_array = np.linspace(t_start, t_stop, int((t_stop - t_start) / dt + 1))
        for t in self._time_array:
            for i in range(len(self._dubins_control)):
                robot = self._dubins_control[i]
                robot.update_memory_about_itself()
                robot_state = robot.get_state()
                robot_pose = np.array(robot.get_pose2D())
                # Store states
                self._states_array[i].append(robot_pose)
                # Control itetration
                if robot_state == state['in circle']:
                    robot.calculate_lap()
                    robot.calculate_wills()
                    # robot.prevent_collision()
                    robot.evaluate_wills()
                elif robot_state == state['transition']:
                    robot.check_arrival()
                [v,w] = robot.calculate_input_signals()
                # Integrate control signals
                dx = dt*v*cos(robot_pose[2])
                dy = dt*v*sin(robot_pose[2])
                dtheta = dt*w
                robot_pose = robot_pose + np.array([dx,dy,dtheta])
                robot.set_pose2D(robot_pose)

    def communicate(self):
        pass

    def animate(self, skip = 10, fixed_axes = True):
        frame_rate = range(0,len(self._time_array),skip)
        random_colors = np.random.randint(0, 255, [self._n_groups+1, 3])
        for n in frame_rate:
            fig = plt.figure(figsize=(20, 20))
            for robot_id in range(len(self._states_array)):
                pose = self._states_array[robot_id][n]
                x = pose[0]; y = pose[1]; theta = pose[2]

                traj_x = [pose[0] for pose in self._states_array[robot_id][0:n]]; traj_y = [pose[1] for pose in self._states_array[robot_id][0:n]]; traj_theta = [pose[2] for pose in self._states_array[robot_id][0:n]]
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(x,y,theta,s=40,marker='o', color = '#888888')
                # ax.plot3D(traj_x, traj_y, traj_theta, color = '#222222')
                # ax.grid(True)
                group = self._dubins_control[robot_id].get_group()
                color = random_colors[group,:]/255
                plt.scatter(x,y,s=100,marker='o', color = color)
                plt.plot(traj_x, traj_y, color = color,linewidth=3)
                plt.grid(True)
                # ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                if fixed_axes:
                #     ax.set_xlim(-10,10); ax.set_ylim(-10,10); ax.set_zlim(-10,10)
                    plt.xlim([40, -40]);  plt.ylim([40, -40])
                # ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('theta')
            plt.xlabel('x'); plt.ylabel('y')
            plt.savefig(f"animation/{n}.jpg")
            plt.close()

        images = [Image.open(f"animation/{n}.jpg") for n in frame_rate]
        images[0].save('animation.gif', save_all=True, append_images=images[1:], duration=skip, loop=0)



if __name__ == '__main__':
    groups = [0,0,1,2]
    params = {
        'Rb': 3.5,
        'd': 10.0,
        'c': 36.0,
        'ref_vel': 1.0
    }
    initial_poses = [
        [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        [0.0, 30.0, 0.0],
        [20.0, 0.0, 0,0]
    ]
    sim = Simulator(groups, params, initial_poses)
    sim.run()
    sim.animate()