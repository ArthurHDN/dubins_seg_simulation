import os
import shutil
from math import cos, sin
import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from PIL import Image
import logging
import logging.handlers
from tqdm import tqdm
# from alive_progress import alive_bar

from pydubinsseg import state
from pydubinsseg.segregationcontrol import SegregationControl


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s | %(funcName)s | %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
file_handler = logging.handlers.WatchedFileHandler(filename='simulation.log', mode='a', encoding='utf-8', delay=False)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


class Simulator():

    _TMP_ANIMATION_DIR = 'animation'
    _ANIMATION_OUTPUT_FILE = 'animation.gif'
    
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
        self._params = params
        logger.info('Succesfully created simulator object ' + str(self))

    def run(self, dt = 0.1, t_start = 0.0, t_stop = 100.0):
        logger.info('Running simulator')
        self._time_array = np.linspace(t_start, t_stop, int((t_stop - t_start) / dt + 1))
        # with alive_bar(1000) as bar:
        #     for i in self.run_loop(dt=dt):
        #         bar()
        for i in tqdm(self.run_loop(dt=dt)):
            pass
        logger.info('Simulation finished')

    def run_loop(self, dt = 0.1):
        for t in self._time_array:
            # All send/recieve memory info
            self.communicate()
            # For each robot
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
            yield

    def communicate(self):
        for i in range(self._n_robots):
            for j in range(i+1,self._n_robots):
                robot_i = self._dubins_control[i]
                robot_j = self._dubins_control[j]
                i_position = np.array( robot_i.get_pose2D()[:2] )
                j_position = np.array( robot_j.get_pose2D()[:2] )
                if np.linalg.norm(i_position  - j_position) <= self._params['c']:
                    i_memory = robot_i.send_memory()
                    j_memory = robot_j.send_memory()
                    robot_i.recieve_memory(j_memory)
                    robot_j.recieve_memory(i_memory)

    def animate(self, skip = 10, fixed_axes = True):
        logger.info('Creating animation')
        # Create animation tmp dir
        if os.path.isfile(self._ANIMATION_OUTPUT_FILE):
            os.remove(self._ANIMATION_OUTPUT_FILE)
        if not os.path.isdir(self._TMP_ANIMATION_DIR):
            os.mkdir(self._TMP_ANIMATION_DIR)
        # Animation loop
        frame_rate = range(0,len(self._time_array),skip)
        random_colors = np.random.randint(0, 255, [self._n_groups+1, 3])
        # with alive_bar(1000) as bar:
        #     for i in self.animate_loop(frame_rate,random_colors,fixed_axes=fixed_axes):
        #         bar()
        for i in tqdm(self.animate_loop(frame_rate,random_colors,fixed_axes=fixed_axes)):
            pass
        # Export result
        images = [Image.open(f"{self._TMP_ANIMATION_DIR}/{n}.jpg") for n in frame_rate]
        images[0].save(self._ANIMATION_OUTPUT_FILE, save_all=True, append_images=images[1:], duration=skip, loop=0)
        # Cleanup animation tmp
        shutil.rmtree(self._TMP_ANIMATION_DIR)
        logger.info('Animation finished')

    def animate_loop(self, frame_rate, random_colors, fixed_axes = True):
        for n in frame_rate:
            fig = plt.figure(figsize=(20, 20))
            for robot_id in range(len(self._states_array)):
                pose = self._states_array[robot_id][n]
                x = pose[0]; y = pose[1]; theta = pose[2]
                traj_x = [pose[0] for pose in self._states_array[robot_id][0:n]]; traj_y = [pose[1] for pose in self._states_array[robot_id][0:n]]; traj_theta = [pose[2] for pose in self._states_array[robot_id][0:n]]
                group = self._dubins_control[robot_id].get_group()
                color = random_colors[group,:]/255
                plt.scatter(x,y,s=100,marker='o', color = color)
                plt.plot(traj_x, traj_y, color = color,linewidth=3)
                plt.grid(True)
                if fixed_axes:
                    plt.xlim([40, -40]);  plt.ylim([40, -40])
            plt.xlabel('x'); plt.ylabel('y')
            plt.savefig(f"{self._TMP_ANIMATION_DIR}/{n}.jpg")
            plt.close()
            yield


if __name__ == '__main__':
    params = {
        'Rb': 3.5,
        'd': 10.0,
        'c': 36.0,
        'ref_vel': 1.0
    }
    groups = [0,0,2,1]
    initial_poses = [
        [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        [0.0, 30.0, 0.0],
        [20.0, 0.0, 0,0]
    ]
    sim = Simulator(groups, params, initial_poses)
    sim.run(dt = 0.5, t_start = 0.0, t_stop = 6000.0)
    sim.animate(skip = 20)
