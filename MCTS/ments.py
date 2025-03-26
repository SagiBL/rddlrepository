import time
import math
import pyRDDLGym
from MCTS import agent
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from collections import deque
from binarytree import build


alfa = 100
gama = 0.98
explore_MENTS=1
default_min_reward = -10000                    #the worst simulated reward for random agent
sim_reward = 0
red_time = 4
min_green = 6
max_green = 60


class Node:    #define the format of the nodes
    def __init__(self, parent, state, change):
        self.parent = parent
        self.N = 1
        self.child_stay = None
        self.child_change = None
        if parent is not None:
            self.depth = parent.depth + 1
            self.reward_to_node = parent.reward_to_node
        else:
            self.depth = 0
            self.reward_to_node = 0
        self.state = state
        self.q_soft_stay = -10000
        self.q_soft_change = -10000
        self.IsChange = change
        self.v_soft = 0
        self.reward_stay = 0
        self.reward_change = 0

    def Jpolicy_stay(self, explore=explore_MENTS):
        stay_prob = random.uniform(0, 1)
        change_prob = 1 - stay_prob
        lamda = (explore * 2) / (1 + math.log(self.N, 2))
        val_stay = (1 - lamda) * math.exp((self.q_soft_stay - self.v_soft) / alfa) + lamda * stay_prob
        val_change = (1 - lamda) * math.exp((self.q_soft_change - self.v_soft) / alfa) + lamda * change_prob
        return val_stay > val_change

class MCTS:
    def __init__(self, state, depth_of_root, explore=explore_MENTS, instance=0, min_reward=default_min_reward):    #initialize the tree
        self.root_state = deepcopy(state)
        self.root_node = Node(None, self.root_state, False)
        self.root_node.depth = depth_of_root
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.stay = {'advance___i0':0}
        self.change = {'advance___i0':1}
        self.explore = explore
        self.instance = instance
        self.min_reward = min_reward


    def step(self):   # preforms one step of expansion and simulates it
        node = self.root_node
        while node.child_stay is not None or node.child_change is not None:   #choose the best child and advance state accordingly
            if node.child_stay is None:
                node = node.child_change
            elif node.child_change is None:
                node = node.child_stay
            elif node.Jpolicy_stay(self.explore):
                node = node.child_stay
            else:
                node = node.child_change


        if node.state['signal___i0'] % 2 == 0:
            if node.state['signal-t___i0'] < red_time: #if it's still red light
                tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
                node.reward_stay /= self.min_reward
                node.child_stay = Node(node, tmp_state, False)  # create the stay_child
                node.q_soft_stay = self.simulate(tmp_state, node.child_stay.depth) + node.child_stay.reward_to_node
                self.root_node = self.back_propagate(node)
            else: #if it's done being red light
                tmp_state, node.reward_change = self.one_step(node.state, self.change)
                node.reward_change /= self.min_reward
                node.child_change = Node(node, tmp_state, True)  # create the change_child
                node.q_soft_change = self.simulate(tmp_state, node.child_change.depth) + node.child_change.reward_to_node
                self.root_node = self.back_propagate(node)
        else:
            if node.state['signal-t___i0'] < min_green: #if it's too early to change
                tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
                node.reward_stay /= self.min_reward
                node.child_stay = Node(node, tmp_state, False)  # create the stay_child
                node.q_soft_stay = self.simulate(tmp_state, node.child_stay.depth) + node.child_stay.reward_to_node            #
                self.root_node = self.back_propagate(node)
            elif node.state['signal-t___i0'] == max_green: #if reached the maximum time without changing
                tmp_state, node.reward_change = self.one_step(node.state, self.change)
                node.reward_change /= self.min_reward
                node.child_change = Node(node, tmp_state, True)  # create the change_child
                node.q_soft_change = self.simulate(tmp_state, node.child_change.depth) + node.child_change.reward_to_node          #
                self.root_node = self.back_propagate(node)
            else: #both changing and staying are legal moves
                tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
                node.reward_stay /= self.min_reward
                node.child_stay = Node(node, tmp_state, False)  # create the stay_child
                node.q_soft_stay = self.simulate(tmp_state, node.child_stay.depth) + node.child_stay.reward_to_node

                tmp_state, node.reward_change = self.one_step(node.state, self.change)
                node.reward_change /= self.min_reward
                node.child_change = Node(node, tmp_state, True)  # create the change_child
                node.q_soft_change = self.simulate(tmp_state, node.child_change.depth) + node.child_change.reward_to_node
                node.N += 1
                self.root_node = self.back_propagate(node)


    def one_step(self, state, action):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', instance=self.instance)
        _ = tmp_env.reset()
        tmp_env.set_state(state)
        next_state, reward, terminated, truncated, _ = tmp_env.step(action)
        tmp_env.close()
        return next_state, reward


    def simulate(self, state, depth):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', instance=self.instance)
        _ = tmp_env.reset()
        tmp_env.set_state(state)
        agent1 = agent.RandomAgent(
            action_space=tmp_env.action_space,
            num_actions=tmp_env.max_allowed_actions)
        simulated_reward = 0
        values = [1, 0]
        probabilities = [0.1, 0.9]
        for step in range(tmp_env.horizon-depth):
            #print("inner step ", step)
            action = agent1.sample_action(state)
            variable = random.choices(values, probabilities)[0]
            if variable and state['signal___i0'] % 2 == 1 and max_green > state['signal-t___i0'] >= min_green:
                action = self.stay
            next_state, reward, terminated, truncated, _ = tmp_env.step(action)
            simulated_reward = simulated_reward + reward
            state = next_state
            if truncated or terminated:
                break
        tmp_env.close()
        return simulated_reward - self.min_reward


    def back_propagate(self, node):
        from_change = node.IsChange
        node.v_soft = alfa * math.log(math.exp(node.q_soft_stay / alfa) + math.exp(node.q_soft_change / alfa))
        while node.parent is not None:
            node = node.parent
            node.N += 1

            if from_change:
                node.q_soft_change = node.child_change.v_soft
                if node.child_stay is None:
                    node.v_soft = node.q_soft_change
                else:
                    node.v_soft = alfa * math.log(math.exp(node.q_soft_stay / alfa) + math.exp(node.q_soft_change / alfa))

            else:
                node.q_soft_stay = node.child_stay.v_soft
                if node.child_change is None:
                    node.v_soft = node.q_soft_stay
                else:
                    node.v_soft = alfa * math.log(math.exp(node.q_soft_stay/alfa) + math.exp(node.q_soft_change/alfa))

            from_change = node.IsChange
        return node


    def search(self, time_limit: int):
        start_time = time.process_time()

        while time.process_time() - start_time < time_limit:
            self.step()
            self.num_rollouts += 1

        self.run_time = time.process_time() - start_time


    def best_action(self):  # returns the best move for the next iteration
        if self.root_node.q_soft_stay < self.root_node.q_soft_change:
            return self.change
        else:
            return self.stay

    ###from here the code is only for displaying the results

    def statistics(self):      #return the overall calculation statistics
        return self.num_rollouts, self.run_time


    def print_tree(self):
        fig, ax = plt.subplots(figsize=(50, 10))
        plot_binary_tree(self.root_node, ax=ax)

        # Set aspect, remove axes, and display the tree
        ax.set_aspect('equal')
        ax.axis('off')  # Remove axes
        plt.savefig('binary_tree'+ str(self.root_node.depth) +'.png', format='png')


def plot_binary_tree(root, x=0, y=0, layer=1, width=100000.0, ax=None):
    """Recursively plot the binary tree using matplotlib."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(50, 10))

    # If the current node is not None
    if root:
        ax.text(x, y, str(root.N), ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='skyblue', edgecolor='black', boxstyle='round,pad=1'))

        if root.child_change:
            ax.plot([x, x - width], [y - 2500, y - 5000], color="black", lw=2)  # Draw edge to left child
            plot_binary_tree(root.child_change, x - width, y - 5000, layer + 1, (width / 1.6) + 200, ax)

        if root.child_stay:
            ax.plot([x, x + width], [y - 2500, y - 5000], color="black", lw=2)  # Draw edge to right child
            plot_binary_tree(root.child_stay, x + width, y - 5000, layer + 1, (width / 1.6) + 200, ax)

    return ax
