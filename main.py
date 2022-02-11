from pydubinsseg import state
from pydubinsseg.segregationcontrol import SegregationControl
import numpy as np


class Simulator():
    def __init__(self,groups,params,initial_poses):
        number = 0
        self._dubins_control = []
        for i in range(len(groups)):
            robot = SegregationControl(number, groups[i],state['in circle'],params)
            robot.set_pose2D(initial_poses[i])
            robot.calculate_initial_conditions()
            self._dubins_control.append(robot)
            number = number + 1

    def run(self):
        t_start = 0.0
        t_stop = 10.0
        dt = 0.1
        time_array = np.linspace(t_start, t_stop, int((t_stop - t_start) / dt + 1))
        for t in time_array:
            for robot in self._dubins_control:
                robot.update_memory_about_itself()
                robot_state = robot.get_state()
                if robot_state == state['in circle']:
                    robot.calculate_lap()
                    robot.calculate_wills()
                    # robot.prevent_collision()
                    robot.evaluate_wills()
                elif robot_state == state['transition']:
                    robot.check_arrival()
                [v,w] = robot.calculate_input_signals()

                robot_pose = robot.get_pose2D()

        

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
        [0.0, 10.0, 0.0],
        [0.0, 20.0, 0.0],
        [20.0, 0.0, 0,0]
    ]
    sim = Simulator(groups, params, initial_poses)
    sim.run()