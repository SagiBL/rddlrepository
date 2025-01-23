import time
import math
import pyRDDLGym
from MCTS import random_agent #import RandomAgent
import matplotlib.pyplot as plt
from copy import deepcopy
import random

# import pygraphviz as pgv
from collections import deque
from binarytree import build

explore_c=1000
default_min_reward = -10000                    #the worst simulated reward for random agent
sim_reward = 0
red_time = 4
min_green = 6
max_green = 60

class Node:    #define the format of the nodes
    def __init__(self, parent, state):
        self.parent = parent
        self.N = 1
        self.total_reward = 0
        self.child_stay = None
        self.child_change = None
        if parent is not None:
            self.depth = parent.depth + 1
            self.reward_to_node = parent.reward_to_node
        else:
            self.depth = 0
            self.reward_to_node = 0
        self.state = state



    def value(self, explore=explore_c, min_reward=default_min_reward):  #calculate the UCB
        if self.parent is None:
            return 100
        else:
            val = -min_reward+(self.total_reward / self.N) + explore * math.sqrt(math.log(self.parent.N) / self.N)
            return val


class MCTS:
    def __init__(self, state, depth_of_root, explore=explore_c, instance=0, min_reward=default_min_reward):    #initialize the tree
        self.root_state = deepcopy(state)
        self.root_node = Node(None, self.root_state)
        self.root_node.depth = depth_of_root
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.stay = {'advance___i0':0}    #this is dumb but it works for now
        self.change = {'advance___i0':1}
        self.explore = explore
        self.instance = instance
        self.min_reward = min_reward


    def step(self):   # preforms one step of expansion and simulates it
        node = self.root_node
        while node.child_stay is not None or node.child_change is not None:   #choose the best child and advance state accordingly
            if node.child_stay is None:
                node = node.child_change
                #print("must change")
            elif node.child_change is None:
                node = node.child_stay
                #print("must stay")
            elif node.child_stay.value(self.explore, self.min_reward) < node.child_change.value(self.explore, self.min_reward):
                #if node.depth == 11:
                #    print("N =",node.N,"stay value =", node.child_stay.value(self.explore), "change value =", node.child_change.value(self.explore))
                node = node.child_change
                #print("decide to change")
            else:
                #if node.depth == 11:
                #    print("N =",node.N,"stay value =", node.child_stay.value(self.explore), "change value =", node.child_change.value(self.explore))
                node = node.child_stay
                #print("decide to stay")

        # print("finish")
        # print(node.state['signal___i0'])
        # print(node.state['signal-t___i0'])

        if node.state['signal___i0'] % 2 == 0:
            if node.state['signal-t___i0'] < red_time: #if it's still red light
                tmp_state, step_reward = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state)  # create the stay_child
                node.child_stay.reward_to_node += step_reward
                node.child_stay.total_reward = self.simulate(tmp_state, node.child_stay.depth) + node.child_stay.reward_to_node
                self.root_node = self.back_propagate(node.child_stay, node.child_stay.total_reward)
            else: #if it's done being red light
                tmp_state, step_reward = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state)  # create the change_child
                node.child_change.reward_to_node += step_reward
                node.child_change.total_reward = self.simulate(tmp_state, node.child_change.depth) + node.child_change.reward_to_node
                self.root_node = self.back_propagate(node.child_change, node.child_change.total_reward)
        else:
            if node.state['signal-t___i0'] < min_green: #if it's too early to change
                tmp_state, step_reward = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state)  # create the stay_child
                node.child_stay.reward_to_node += step_reward
                node.child_stay.total_reward = self.simulate(tmp_state, node.child_stay.depth) + node.child_stay.reward_to_node             #
                self.root_node = self.back_propagate(node.child_stay, node.child_stay.total_reward)
            elif node.state['signal-t___i0'] == max_green: #if reached the maximum time without changing
                tmp_state, step_reward = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state)  # create the change_child
                node.child_change.reward_to_node += step_reward
                node.child_change.total_reward = self.simulate(tmp_state, node.child_change.depth) + node.child_change.reward_to_node           #
                self.root_node = self.back_propagate(node.child_change, node.child_change.total_reward)
            else: #both changing and staying are legal moves
                tmp_state, step_reward = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state)  # create the stay_child
                node.child_stay.reward_to_node += step_reward
                node.child_stay.total_reward = self.simulate(tmp_state, node.child_stay.depth) + node.child_stay.reward_to_node
                self.root_node = self.back_propagate(node.child_stay, node.child_stay.total_reward)
                tmp_state, step_reward = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state)  # create the change_child
                node.child_change.reward_to_node += step_reward
                node.child_change.total_reward = self.simulate(tmp_state, node.child_change.depth) + node.child_change.reward_to_node
                self.root_node = self.back_propagate(node.child_change, node.child_change.total_reward)


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
        agent = random_agent.RandomAgent(
            action_space=tmp_env.action_space,
            num_actions=tmp_env.max_allowed_actions)
        simulated_reward = 0
        values = [1, 0]
        probabilities = [0.3, 0.7]
        for step in range(tmp_env.horizon-depth):
            #print("inner step ", step)
            action = agent.sample_action(state)
            variable = random.choices(values, probabilities)[0]
            if variable and state['signal___i0'] % 2 == 1 and max_green > state['signal-t___i0'] >= min_green:
                action = self.stay
            next_state, reward, terminated, truncated, _ = tmp_env.step(action)
            simulated_reward = simulated_reward + reward
            state = next_state
            if truncated or terminated:
                break
        tmp_env.close()
        return simulated_reward


    def back_propagate(self, node, reward):
        while node.parent is not None:
            node = node.parent
            node.N += 1
            node.total_reward += reward
        return node



    def search(self, time_limit: int):
        start_time = time.process_time()

        while time.process_time() - start_time < time_limit:
            self.step()
            self.num_rollouts += 1

        self.run_time = time.process_time() - start_time


    def best_action(self):    #returns the best move for the next iteration
        if self.root_node.child_stay.value(explore=0, min_reward=0) < self.root_node.child_change.value(explore=0, min_reward=0):
            return self.change
        else:
            return self.stay


    def statistics(self):      #return the overall calculation statistics
        return self.num_rollouts, self.run_time

    def bfs_traversal(self,results,values,visits):
        """Perform breadth-first traversal and return a list of values."""
        root = self.root_node
        queue = deque([root])  # Initialize the queue with the root node

        for i in range(2000):
            current_node = queue.popleft()  # Dequeue the front node
            if current_node is None:
                results.append(0)
                values.append(0)
                visits.append(0)
                queue.append(None)
                queue.append(None)
            else:
                results.append(int(-self.min_reward+(current_node.total_reward / current_node.N)))  # Visit the node
                values.append(current_node.value(self.explore, self.min_reward))
                visits.append(current_node.N)
                queue.append(current_node.child_change)                    # Enqueue the left and right children
                queue.append(current_node.child_stay)

        return results,values,visits

    def build_tree(self, visits):
        # Creating binary tree from given list
        binary_tree = build(visits)
        # print('Binary tree from list :\n',
        #      binary_tree)
        print('\nList from binary tree :',
              binary_tree.values)

        # fig, ax = plt.subplots(figsize=(50, 10))
        # plot_binary_tree(self.root_node, ax=ax)
        #
        # # Set aspect, remove axes, and display the tree
        # ax.set_aspect('equal')
        # ax.axis('off')  # Remove axes
        # plt.savefig('binary_tree.png', format='png')

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
