import os
from math import cos, sin, pi
import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
# from PIL import Image
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
    _ANIMATION_OUTPUT_FILE = 'animation.mp4'
    
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
        self._n_robots = number
        self._m_groups = max(groups)+1
        self._params = params
        logger.info('Succesfully created simulator object ' + str(self))
        logger.info(f'Simulation parameters: N robots = {self._n_robots}, M groups = {self._m_groups}, params = {params}')

    def run(self, dt = 0.1, t_start = 0.0, t_stop = 100.0):
        logger.info('Running simulator')
        self._time_array = np.linspace(t_start, t_stop, int((t_stop - t_start) / dt + 1))
        self._segragation_indexes = []
        self._random_colors = np.random.randint(0, 255, [self._m_groups+1, 3])
        # with alive_bar(1000) as bar:
        #     for i in self._run_loop(dt=dt):
        #         bar()
        for i in tqdm(self._run_loop(dt=dt)):
            pass
        logger.info('Simulation finished')

    def _run_loop(self, dt = 0.1):
        for t in self._time_array:
            # All send/recieve memory info
            self._communicate_robots()
            self._segragation_indexes.append(self._compute_segregation_index())
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
                    robot.prevent_collision()
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

    def _communicate_robots(self):
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

    def _compute_segregation_index(self):
        home_zones = {}
        for robot_i in self._dubins_control:
            i_group = robot_i.get_group()
            i_position = np.array( robot_i.get_pose2D()[:2] )
            i_curve = int(round(np.linalg.norm(i_position)/self._params['d']))
            if i_group in home_zones:
                if i_curve < home_zones[i_group]['inner']:
                    home_zones[i_group]['inner'] = i_curve
                if i_curve > home_zones[i_group]['outer']:
                    home_zones[i_group]['outer'] = i_curve
            else:
                home_zones[i_group] = {'inner': i_curve, 'outer': i_curve}
        index = 0.0
        for robot_i in self._dubins_control:
            i_group = robot_i.get_group()
            i_position = np.array( robot_i.get_pose2D()[:2] )
            i_curve = int(round(np.linalg.norm(i_position)/self._params['d']))
            for group, zone in home_zones.items():
                if group == i_group:
                    continue
                if i_curve >= zone['inner'] and i_curve <= zone['outer']:
                    index = index + 1.0
        return 1 - (index/(self._n_robots*self._m_groups))

    def animate(self, interval = 50, fps = 30):
        logger.info('Creating animation')
        if os.path.isfile(self._ANIMATION_OUTPUT_FILE):
            os.remove(self._ANIMATION_OUTPUT_FILE)
        # Animation loop
        fig = plt.figure(figsize=(20, 20))
        ax = plt.axes(xlim=(-40, 40), ylim=(-40, 40))
        self._robots_plot = []
        n = 0
        for robot_id in range(len(self._states_array)):
            pose = self._states_array[robot_id][n]
            x = pose[0]; y = pose[1]; theta = pose[2]
            traj_x = [pose[0] for pose in self._states_array[robot_id][0:n]]; traj_y = [pose[1] for pose in self._states_array[robot_id][0:n]]; traj_theta = [pose[2] for pose in self._states_array[robot_id][0:n]]
            group = self._dubins_control[robot_id].get_group()
            color = self._random_colors[group,:]/255
            pose_plot, = ax.plot(x,y,'o',markersize=self._params['Rb'], color = color)
            traj_plot, = ax.plot(traj_x, traj_y, color = color,linewidth=3)
            self._robots_plot.append({
                'position': pose_plot,
                'trajectory': traj_plot
            })
        plt.xlabel('x'); plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        anim = ani.FuncAnimation(fig, self._animate_loop, frames = len(self._time_array), interval = interval, blit = False)
        plt.tight_layout()
        # plt.show()
        logger.info('Exporting animation file')
        anim.save(self._ANIMATION_OUTPUT_FILE, fps = fps, writer = 'ffmpeg')
        logger.info('Exported animation file to ' + str(self._ANIMATION_OUTPUT_FILE))
     
    def _animate_loop(self,n):
        for robot_id in range(len(self._states_array)):
            pose = self._states_array[robot_id][n]
            x = pose[0]; y = pose[1]; theta = pose[2]
            traj_x = [pose[0] for pose in self._states_array[robot_id][0:n]]; traj_y = [pose[1] for pose in self._states_array[robot_id][0:n]]; traj_theta = [pose[2] for pose in self._states_array[robot_id][0:n]]
            self._robots_plot[robot_id]['position'].set_data(x,y)
            self._robots_plot[robot_id]['trajectory'].set_data(traj_x,traj_y)
        
    def plot_results(self):
        logger.info('Plotting results')
        fig = plt.figure()
        plt.plot(self._time_array, self._segragation_indexes)
        plt.grid(True)
        plt.xlabel('t'); plt.ylabel('Segregation Index')
        plt.title('Segregation Index')
        # plt.show(block=False)
        plt.savefig('segregation_index.jpg')
        logger.info('Saved file segregation_index.jpg')

        fig = plt.figure()
        for robot_id in range(len(self._states_array)):
            pose = self._states_array[robot_id][-1]
            x = pose[0]; y = pose[1]; theta = pose[2]
            traj_x = [pose[0] for pose in self._states_array[robot_id][:]]; traj_y = [pose[1] for pose in self._states_array[robot_id][:]]; traj_theta = [pose[2] for pose in self._states_array[robot_id][:]]
            group = self._dubins_control[robot_id].get_group()
            color = self._random_colors[group,:]/255
            plt.scatter(x,y,s=pi*self._params['Rb']**2,marker='o', color = color)
            plt.plot(traj_x, traj_y, color = color,linewidth=3)
            plt.grid(True)
            plt.gca().set_aspect('equal', adjustable='box')
        # plt.show(block=False)
        # plt.show()
        plt.savefig('trajectory.jpg')
        logger.info('Saved file trajectory.jpg')


if __name__ == '__main__':
    params = {
        'Rb': 3.5,
        'd': 10.0,
        'c': 36.0,
        'ref_vel': 1.0
    }
    groups = [0,0,0]
    initial_poses = [
        [10.0, 0.0, -pi/2],
        # [0.0, 20.0, -pi],
        # [-20.0, 0.0, -pi/2],
        # [20.0, 0.0, pi/2],
        [20.0, 0.0, pi/2],
        [-30.0, 0.0, pi/2]
    ]
    sim = Simulator(groups, params, initial_poses)
    sim.run(dt = 0.1, t_start = 0.0, t_stop = 60.0)
    sim.plot_results()
    # sim.animate()
