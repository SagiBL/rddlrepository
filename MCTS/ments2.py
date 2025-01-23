import time
import math
import random
from copy import deepcopy
import pyRDDLGym
from MCTS import random_agent #import RandomAgent
import matplotlib.pyplot as plt

# import pygraphviz as pgv
from collections import deque
from binarytree import build

alfa = 0.02
explore_MENTS = 1
gama = 0.95
max_reward = -12000
max_step_reward = 0
red_time = 4
min_green = 6
max_green = 60
default_action_number = 2

class Node:    #define the format of the nodes
    def __init__(self, parent, state, change):
        self.parent = parent
        self.N = 0
        self.reward_stay = 0
        self.reward_change = 0
        self.child_stay = None
        self.child_change = None
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.state = state
        self.q_soft_stay = -100
        self.q_soft_change = -100
        self.IsChange = change

        self.v_soft = 0
        self.actions = 1
        if self.state['signal___i0'] % 2 == 1 and max_green > self.state['signal-t___i0'] >= min_green:
            self.actions = 2

    def Jpolicy_stay(self, explore=explore_MENTS):
        stay_prob = random.uniform(0, 1)
        change_prob = 1-stay_prob
        lamda = (explore*self.actions) / (1 + math.log(self.N,2))
        val_stay = (1 - lamda) * math.exp((self.q_soft_stay - self.v_soft) / alfa) + lamda * stay_prob
        val_change = (1 - lamda) * math.exp((self.q_soft_change - self.v_soft) / alfa) + lamda * change_prob
        if self.depth == 620:
            print("N =", self.N, "lamda =",lamda)
            print("|| val_stay =",val_stay,"|| val_change =", val_change)
            print("stay_prob =",stay_prob,"|| change_prob =", change_prob)
            print("q_soft_stay =", math.exp((self.q_soft_stay - self.v_soft) / alfa), "|| q_soft_change =", math.exp((self.q_soft_change - self.v_soft) / alfa))
            print()
        return val_stay > val_change


class MCTS:
    def __init__(self, state, depth_of_root, explore=explore_MENTS):    #initialize the tree
        self.root_state = deepcopy(state)
        self.root_node = Node(None, self.root_state, False)
        self.root_node.depth = depth_of_root
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.stay = {'advance___i0':0}    #this is dumb but it works for now
        self.change = {'advance___i0':1}
        self.explore = explore


    def step(self):   # preforms one step of expansion and simulates it
        node = self.root_node
        while node.child_stay is not None or node.child_change is not None:   #choose the best child and advance state accordingly
            if node.child_stay is None:
                node = node.child_change
                #print("must change")
            elif node.child_change is None:
                node = node.child_stay
                #print("must stay")
            elif node.Jpolicy_stay(explore=self.explore):
                node = node.child_stay
                #print("decide to stay")
            else:
                node = node.child_change
                #print("decide to change")


        if node.state['signal___i0'] % 2 == 0:
            if node.state['signal-t___i0'] < red_time: #if it's still red light
                tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state, False)  # create the stay_child
                node.reward_stay /= 1000
                self.root_node = self.back_propagate(node,False)
                self.root_node = self.back_propagate(node, False)

            else: #if it's done being red light
                tmp_state, node.reward_change = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state, True)  # create the change_child
                node.reward_change /= 1000
                self.root_node = self.back_propagate(node,True)
                self.root_node = self.back_propagate(node,True)

        else:
            if node.state['signal-t___i0'] < min_green: #if it's too early to change
                tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state, False)  # create the stay_child
                node.reward_stay /= 1000
                self.root_node = self.back_propagate(node,False)
                self.root_node = self.back_propagate(node,False)

            elif node.state['signal-t___i0'] >= max_green: #if reached the maximum time without changing
                tmp_state, node.reward_change = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state, True)  # create the change_child
                node.reward_change /= 1000
                self.root_node = self.back_propagate(node,True)
                self.root_node = self.back_propagate(node,True)

            else: #both changing and staying are legal moves
                tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state, False)  # create the stay_child
                node.reward_stay /= 1000
                self.root_node = self.back_propagate(node, False)

                tmp_state, node.reward_change = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state, True)  # create the change_child
                node.reward_change /= 1000
                self.root_node = self.back_propagate(node,True)
                #print("stay_reward",node.reward_stay,"|| change_reward",node.reward_change)


    def one_step(self, state, action):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
        _ = tmp_env.reset()
        tmp_env.set_state(state)
        next_state, reward, terminated, truncated, _ = tmp_env.step(action)
        tmp_env.close()
        return next_state, reward

    def simulate(self, state):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
        _ = tmp_env.reset()
        tmp_env.set_state(state)
        agent = random_agent.RandomAgent(
            action_space=tmp_env.action_space,
            num_actions=tmp_env.max_allowed_actions)
        simulated_reward = 0
        for step in range(100):# -self.root_node.depth+1):
            #print("inner step ", step)
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = tmp_env.step(action)
            #print("step",step,"reward",reward)
            simulated_reward = simulated_reward + ((max_step_reward+reward)/1000) * (gama ** step)
            #print("normalized reward =", ((80+reward)/1000))
            state = next_state
            if truncated or terminated:
                break
        tmp_env.close()
        return simulated_reward


    def calc_g(self, node):
        G = self.simulate(node.state)
        return G


    def back_first_time(self, node, from_change):
        node.N += 1
        G = self.calc_g(node)
        if from_change:
            node.q_soft_change = node.reward_change + gama * G
        else:
            node.q_soft_stay = node.reward_stay + gama * G
        node.v_soft = alfa * math.log(math.exp(node.q_soft_stay / alfa) + math.exp(node.q_soft_change / alfa))
        return node.IsChange

    def back_propagate(self, node, from_change):
        from_change = self.back_first_time(node, from_change)
        while node is not None:
            if node.parent is not None:
                node = node.parent
            else:
                return node
            node.N += 1
            if from_change:
                node.q_soft_change = node.reward_change + gama * node.child_change.v_soft
            else:
                node.q_soft_stay = node.reward_stay + gama * node.child_stay.v_soft
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


    def statistics(self):      #return the overall calculation statistics
        return self.num_rollouts, self.run_time

    def bfs_traversal(self,results,values,visits):
        """Perform breadth-first traversal and return a list of values."""
        root = self.root_node
        queue = deque([root])  # Initialize the queue with the root node

        for i in range(2000):
            current_node = queue.popleft()  # Dequeue the front node
            if current_node is None:
                visits.append(0)
                queue.append(None)
                queue.append(None)
            else:
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

        fig, ax = plt.subplots(figsize=(50, 10))
        plot_binary_tree(self.root_node, ax=ax)

        # Set aspect, remove axes, and display the tree
        ax.set_aspect('equal')
        ax.axis('off')  # Remove axes
        plt.savefig('binary_tree.png', format='png')

def plot_binary_tree(root, x=0, y=0, layer=1, width=50000.0, ax=None):
    """Recursively plot the binary tree using matplotlib."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(50, 10))

    # If the current node is not None
    if root:
        ax.text(x, y, str(root.N), ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='skyblue', edgecolor='black', boxstyle='round,pad=1'))

        if root.child_change:
            ax.plot([x, x - width], [y - 2500, y - 5000], color="black", lw=2)  # Draw edge to left child
            plot_binary_tree(root.child_change, x - width, y - 5000, layer + 1, (width / 1.6)+20, ax)

        if root.child_stay:
            ax.plot([x, x + width], [y - 2500, y - 5000], color="black", lw=2)  # Draw edge to right child
            plot_binary_tree(root.child_stay, x + width, y - 5000, layer + 1, (width / 1.6)+20, ax)

    return ax






# import time
# import math
# import random
# from copy import deepcopy
# import pyRDDLGym
# from MCTS import random_agent #import RandomAgent
# import matplotlib.pyplot as plt
#
# # import pygraphviz as pgv
# from collections import deque
# from binarytree import build
#
# alfa = 0.02
# explore_MENTS = 1
# gama = 0.95
# max_reward = -12000
# max_step_reward = 0
# red_time = 4
# min_green = 6
# max_green = 60
# default_action_number = 2
#
# class Node:    #define the format of the nodes
#     def __init__(self, parent, state, change):
#         self.parent = parent
#         self.N = 0
#         self.reward_stay = 0
#         self.reward_change = 0
#         self.child_stay = None
#         self.child_change = None
#         if parent is not None:
#             self.depth = parent.depth + 1
#         else:
#             self.depth = 0
#         self.state = state
#         self.q_soft_stay = -100
#         self.q_soft_change = -100
#         self.IsChange = change
#
#         self.v_soft = 0
#         self.actions = 1
#         if self.state['signal___i0'] % 2 == 1 and max_green > self.state['signal-t___i0'] >= min_green:
#             self.actions = 2
#
#     def Jpolicy_stay(self, explore=explore_MENTS):
#         stay_prob = random.uniform(0, 1)
#         change_prob = 1-stay_prob
#         lamda = (explore*self.actions) / (1 + math.log(self.N,2))
#         val_stay = (1 - lamda) * math.exp((self.q_soft_stay - self.v_soft) / alfa) + lamda * stay_prob
#         val_change = (1 - lamda) * math.exp((self.q_soft_change - self.v_soft) / alfa) + lamda * change_prob
#         return val_stay > val_change
#
#
# class MCTS:
#     def __init__(self, state, depth_of_root, explore=explore_MENTS):    #initialize the tree
#         self.root_state = deepcopy(state)
#         self.root_node = Node(None, self.root_state, False)
#         self.root_node.depth = depth_of_root
#         self.run_time = 0
#         self.node_count = 0
#         self.num_rollouts = 0
#         self.stay = {'advance___i0':0}    #this is dumb but it works for now
#         self.change = {'advance___i0':1}
#         self.explore = explore
#
#
#     def step(self):   # preforms one step of expansion and simulates it
#         node = self.root_node
#         while node.child_stay is not None or node.child_change is not None:   #choose the best child and advance state accordingly
#             if node.child_stay is None:
#                 node = node.child_change
#                 #print("must change")
#             elif node.child_change is None:
#                 node = node.child_stay
#                 #print("must stay")
#             elif node.Jpolicy_stay(explore=self.explore):
#                 node = node.child_stay
#                 #print("decide to stay")
#             else:
#                 node = node.child_change
#                 #print("decide to change")
#
#
#         if node.state['signal___i0'] % 2 == 0:
#             if node.state['signal-t___i0'] < red_time: #if it's still red light
#                 tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
#                 node.child_stay = Node(node, tmp_state, False)  # create the stay_child
#                 node.reward_stay = (max_step_reward+node.reward_stay)/1000
#                 self.root_node = self.back_propagate(node,False)
#
#             else: #if it's done being red light
#                 tmp_state, node.reward_change = self.one_step(node.state, self.change)
#                 node.child_change = Node(node, tmp_state, True)  # create the change_child
#                 node.reward_change = (max_step_reward+node.reward_change)/1000
#                 self.root_node = self.back_propagate(node,True)
#
#         else:
#             if node.state['signal-t___i0'] < min_green: #if it's too early to change
#                 tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
#                 node.child_stay = Node(node, tmp_state, False)  # create the stay_child
#                 node.reward_stay = (max_step_reward+node.reward_stay)/1000
#                 self.root_node = self.back_propagate(node,False)
#
#             elif node.state['signal-t___i0'] >= max_green: #if reached the maximum time without changing
#                 tmp_state, node.reward_change = self.one_step(node.state, self.change)
#                 node.child_change = Node(node, tmp_state, True)  # create the change_child
#                 node.reward_change = (max_step_reward+node.reward_change)/1000
#                 self.root_node = self.back_propagate(node,True)
#
#             else: #both changing and staying are legal moves
#                 tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
#                 node.child_stay = Node(node, tmp_state, False)  # create the stay_child
#                 node.reward_stay = (max_step_reward+node.reward_stay)/1000
#                 self.root_node = self.back_propagate(node,False)
#
#                 tmp_state, node.reward_change = self.one_step(node.state, self.change)
#                 node.child_change = Node(node, tmp_state, True)  # create the change_child
#                 node.reward_change = (max_step_reward+node.reward_change)/1000
#                 self.root_node = self.back_propagate(node,True)
#                 #print("stay_reward",node.reward_stay,"|| change_reward",node.reward_change)
#
#
#     def one_step(self, state, action):
#         tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
#         _ = tmp_env.reset()
#         tmp_env.set_state(state)
#         next_state, reward, terminated, truncated, _ = tmp_env.step(action)
#         tmp_env.close()
#         return next_state, reward
#
#     def simulate(self, state):
#         tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
#         _ = tmp_env.reset()
#         tmp_env.set_state(state)
#         agent = random_agent.RandomAgent(
#             action_space=tmp_env.action_space,
#             num_actions=tmp_env.max_allowed_actions)
#         simulated_reward = 0
#         for step in range(100):# -self.root_node.depth+1):
#             #print("inner step ", step)
#             action = agent.sample_action(state)
#             next_state, reward, terminated, truncated, _ = tmp_env.step(action)
#             #print("step",step,"reward",reward)
#             simulated_reward = simulated_reward + ((max_step_reward+reward)/1000) * (gama ** step)
#             #print("normalized reward =", ((80+reward)/1000))
#             state = next_state
#             if truncated or terminated:
#                 break
#         tmp_env.close()
#         return simulated_reward
#
#
#     def calc_g(self, node):
#         original_node = node
#         G = self.simulate(node.state)
#         return original_node, G
#
#     def back_propagate(self, node, from_change):
#         first_time = True
#         from_change = from_change
#         while node.parent is not None:
#             node.N += 1
#             if first_time:
#                 node, G = self.calc_g(node)
#                 if from_change:
#                     node.q_soft_change = node.reward_change + gama * G
#                 else:
#                     node.q_soft_stay = node.reward_stay + gama * G
#                 first_time = False
#             else:
#                 if from_change:
#                     node.q_soft_change = node.reward_change + gama * node.child_change.v_soft
#                 else:
#                     node.q_soft_stay = node.reward_stay + gama * node.child_stay.v_soft
#             node.v_soft = alfa * math.log(math.exp(node.q_soft_stay/alfa) + math.exp(node.q_soft_change/alfa))
#             from_change = node.IsChange
#             node = node.parent
#         node.N += 1
#         if first_time:
#             node, G = self.calc_g(node)
#             if from_change:
#                 node.q_soft_change = node.reward_change + gama * G
#             else:
#                 node.q_soft_stay = node.reward_stay + gama * G
#         else:
#             if from_change:
#                 node.q_soft_change = node.reward_change + gama * node.child_change.v_soft
#             else:
#                 node.q_soft_stay = node.reward_stay + gama * node.child_stay.v_soft
#         node.v_soft = alfa * math.log(math.exp(node.q_soft_stay / alfa) + math.exp(node.q_soft_change / alfa))
#         return node
#
#
#     def search(self, time_limit: int):
#         start_time = time.process_time()
#
#         while time.process_time() - start_time < time_limit:
#             self.step()
#             self.num_rollouts += 1
#
#         self.run_time = time.process_time() - start_time
#
#     def best_action(self):  # returns the best move for the next iteration
#         if self.root_node.q_soft_stay < self.root_node.q_soft_change:
#             return self.change
#         else:
#             return self.stay
#
#
#     def statistics(self):      #return the overall calculation statistics
#         return self.num_rollouts, self.run_time
#
#     def bfs_traversal(self,results,values,visits):
#         """Perform breadth-first traversal and return a list of values."""
#         root = self.root_node
#         queue = deque([root])  # Initialize the queue with the root node
#
#         for i in range(2000):
#             current_node = queue.popleft()  # Dequeue the front node
#             if current_node is None:
#                 visits.append(0)
#                 queue.append(None)
#                 queue.append(None)
#             else:
#                 visits.append(current_node.N)
#                 queue.append(current_node.child_change)                    # Enqueue the left and right children
#                 queue.append(current_node.child_stay)
#
#         return results,values,visits
#
#     def build_tree(self, visits):
#         # Creating binary tree from given list
#         binary_tree = build(visits)
#         # print('Binary tree from list :\n',
#         #      binary_tree)
#         print('\nList from binary tree :',
#               binary_tree.values)
#
#         fig, ax = plt.subplots(figsize=(50, 10))
#         plot_binary_tree(self.root_node, ax=ax)
#
#         # Set aspect, remove axes, and display the tree
#         ax.set_aspect('equal')
#         ax.axis('off')  # Remove axes
#         plt.savefig('binary_tree.png', format='png')
#
# def plot_binary_tree(root, x=0, y=0, layer=1, width=50000.0, ax=None):
#     """Recursively plot the binary tree using matplotlib."""
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(50, 10))
#
#     # If the current node is not None
#     if root:
#         ax.text(x, y, str(root.N), ha='center', va='center', fontsize=8,
#                 bbox=dict(facecolor='skyblue', edgecolor='black', boxstyle='round,pad=1'))
#
#         if root.child_change:
#             ax.plot([x, x - width], [y - 2500, y - 5000], color="black", lw=2)  # Draw edge to left child
#             plot_binary_tree(root.child_change, x - width, y - 5000, layer + 1, (width / 1.6)+20, ax)
#
#         if root.child_stay:
#             ax.plot([x, x + width], [y - 2500, y - 5000], color="black", lw=2)  # Draw edge to right child
#             plot_binary_tree(root.child_stay, x + width, y - 5000, layer + 1, (width / 1.6)+20, ax)
#
#     return ax
#
