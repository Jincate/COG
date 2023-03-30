import numpy as np
import os

from utils import get_theta, get_dist, get_state
import algorithm
import torch
import random


class Agent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.navigation_model_path = os.path.join(self.model_path, "navigation/eva_reward_max")
        self.shoot_model_path = os.path.join(self.model_path, "shoot_policy/shoot")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.state_dim = 9
        self.action_dim = 3
        self.max_pose = 1

        self.policy = algorithm.DPG(self.state_dim, self.action_dim, 1)
        self.policy.load(self.navigation_model_path)

        self.shoot_policy = algorithm.DQN(3, 2)
        self.shoot_policy.load(self.shoot_model_path)
        self.max_action = 1.0

        self.flag_stuck = False
        self.bool_colli = False
        self.len_stuck = 10
        self.i_stuck = 20
        self.angle_list = np.linspace(0.75 * np.pi, -0.75 * np.pi, 61)

        self.turn = random.choice([-1, 1])
        self.length = random.choice([2, 3, 4, 5, 6])
        self.shoot_time = False
        self.queue = np.zeros((self.len_stuck, 2))
        self.i_eva = 0
        self.goal = 0
        self.stuck_times = 0
        self.before_info = None


        self.count = 0
        # you can customize the necessary attributes here

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def func(self, x):
        if x > 0:
            return np.pi * np.log2(1 + x) / np.log2(1 + np.pi)
        else:
            return -np.pi * np.log2(1 - x) / np.log2(1 + np.pi)

    def agent_control(self, obs, done, info):
        # The formats of obs, done, info obey the  CogEnvDecoder api
        # realize your agent here
        if info is not None:
            if self.before_info is not None:
                # 新的一局
                if info[1][1] < self.before_info[1][1]:
                    self.stuck_times = 0
                    self.goal = 0
            if info[1][3] < 5:
                self.shoot_time = False

            self.before_info = info


        self.count += 1

        vector_data = obs["vector"]

        state = get_state(obs, self.goal)

        if not self.shoot_time:
            # print(self.flag_stuck, self.bool_colli, self.shoot_time)
            if self.count > 1:
                before_vector_data = self.before_obs["vector"]
                # print(self.before_obs)
                if abs(vector_data[10][1] - before_vector_data[10][1]) > 0.1 or (np.sum(np.var(self.queue, axis=0)) <= 0.005) and (not self.bool_colli) and (not self.flag_stuck):
                    self.stuck_times += 2
                if abs(vector_data[10][1] - before_vector_data[10][1]) > 0.1:
                    self.bool_colli = True

                # FIFO队列记录 之前 i_stuck 步 的 移动距离
                self.queue[0:self.len_stuck - 1, :] = self.queue[1:self.len_stuck, :]
                self.queue[self.len_stuck - 1, :] = vector_data[0][0:2]
                # 之前第20步 与 当前步的位姿差
                if (np.sum(np.var(self.queue, axis=0)) <= 0.005):
                    self.flag_stuck = True

            if not self.bool_colli and not self.flag_stuck:
                self.stuck_times = max(self.stuck_times - 1, 0)
                action = self.policy.select_action(state)
                shoot = 0
            elif self.flag_stuck or self.bool_colli:

                laser_data = obs["laser"]
                max_index = np.argmax(laser_data)
                if(self.i_stuck > 15):
                    # print(self.i_stuck, self.flag_stuck, self.bool_colli, self.shoot_time, "1")
                    self.i_eva = self.i_eva + 1
                    self.i_stuck -= 1
                    max_vec = max(abs(np.cos(self.angle_list[max_index])), abs(np.sin(self.angle_list[max_index])))
                    action = [
                        2 * (self.sigmoid(2 * get_dist(obs, self.goal)) - 0.5) * np.cos(self.angle_list[max_index]) / max_vec,
                        2 * (self.sigmoid(2 * get_dist(obs, self.goal)) - 0.5) * np.sin(self.angle_list[max_index]) / max_vec,
                        self.angle_list[max_index] / np.pi, 0]
                    shoot = 0

                elif(self.i_stuck > 0):
                    # print(self.i_stuck, self.flag_stuck, self.bool_colli, self.shoot_time, "2")
                    self.i_eva = self.i_eva + 1
                    self.i_stuck -= 1
                    laser_data = obs["laser"]
                    theta = get_theta(obs, self.goal)
                    delta_list = self.angle_list - theta
                    max_index = np.argmax(np.multiply(laser_data, np.cos(delta_list)))
                    max_vec = max(abs(np.cos(self.angle_list[max_index])), abs(np.sin(self.angle_list[max_index])))

                    action = [
                        2 * (self.sigmoid(2 * get_dist(obs, self.goal)) - 0.5) * np.cos(self.angle_list[max_index]) / max_vec,
                        2 * (self.sigmoid(2 * get_dist(obs, self.goal)) - 0.5) * np.sin(self.angle_list[max_index]) / max_vec,
                        self.angle_list[max_index] / (0.75 * np.pi), 0]
                    shoot = 0

                if(self.i_stuck == 0):
                    self.i_stuck = 20 + self.stuck_times
                    self.flag_stuck = False
                    self.bool_colli = False

        if self.shoot_time:
            # print(self.flag_stuck, self.bool_colli, self.shoot_time)
            shoot_state = [state[3], state[7], state[8]]
            modified_state = [state[0], state[1], state[2], state[3], state[4], state[5], state[6],
                              state[7] * self.sigmoid(state[7] - 5), state[8]]

            if state[7] > 5 or abs(state[8]) > 0.1:
                action = self.policy.select_action(modified_state)
            else:
                if self.length > 0:
                    self.length -= 1
                else:
                    self.length = random.choice([2, 3, 4, 5, 6])
                    self.turn = random.choice([-1, 1])
                action = [0, self.turn, -self.turn / state[7]]

            shoot = self.shoot_policy.select_action(shoot_state)

        mx, my, dtheta = action[0], action[1], action[2]
        action_take = [mx, my, dtheta, shoot]

        self.before_obs = obs
        self.before_state = state


        if vector_data[5 + self.goal][2] and not self.shoot_time:
            self.goal = self.goal + 1
        if self.goal > 4:
            self.shoot_time = True
            self.goal = -2



        # here is a simple demo
        # action = self.simple_demo_control(obs, done, info)
        # return action:[vel_x, vel_y, vel_w, shoud_shoot]
        return action_take

    # -------------------- the following codes are for simple demo, can be deleted -----------------------
    def simple_demo_control(self, obs, done, info=None):
        action = [0.0, 0.0, 0.0, 0.0]
        vector_data = obs["vector"]
        num_activated_goals = 0
        if info is not None:
            num_activated_goals = info[1][3]
        self_pose = vector_data[0]
        enemy_activated = vector_data[2]
        enemy_pose = vector_data[3]
        goals_list = [vector_data[i] for i in range(5, 10)]

        if not enemy_activated:
            for goal in goals_list:
                is_activated = goal[-1]
                if is_activated:
                    continue
                else:
                    move_control = self.calculate_move_control(self_pose, goal[:2])
                    move_control.append(0.0)  # add shoot control
                    action = move_control
                    break
        else:
            move_control = self.calculate_move_control(self_pose, enemy_pose[:2])
            dist_to_enemy = np.sqrt((self_pose[0] - enemy_pose[0]) ** 2 + (self_pose[1] - enemy_pose[1]) ** 2)
            if dist_to_enemy < 1.5:
                move_control[0] = np.random.uniform(-0.5, 0.5)
                move_control[1] = np.random.uniform(-0.5, 0.5)
            move_control.append(1.0)
            action = move_control
        return action

    def calculate_move_control(self, self_pose, target_position):
        delta_x = target_position[0] - self_pose[0]
        delta_y = target_position[1] - self_pose[1]
        x_in_robot = delta_x * np.cos(self_pose[2]) + delta_y * np.sin(self_pose[2])
        y_in_robot = -delta_x * np.sin(self_pose[2]) + delta_y * np.cos(self_pose[2])
        theta_in_robot = np.arctan2(y_in_robot, x_in_robot)

        vel_x = 1.0 * x_in_robot
        vel_y = 1.0 * y_in_robot
        vel_w = theta_in_robot

        return [vel_x, vel_y, vel_w]
    # --------------------------------------------------------------------------------------------
