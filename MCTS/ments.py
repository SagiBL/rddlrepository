# import time
# import math
# from copy import deepcopy
# import pyRDDLGym
# from MCTS import random_agent #import RandomAgent
#
# # import pygraphviz as pgv
# from collections import deque
# from binarytree import build
#
# alfa = 1
# explore_MENTS = 3
# gama = 0.98
# # max_reward = -17644.851216448555
# max_reward = -12000
# sim_reward = 0
# red_time = 4
# min_green = 6
# max_green = 60
#
# class Node:    #define the format of the nodes
#     def __init__(self, parent, state):
#         self.parent = parent
#         self.N = 1
#         self.reward = 0
#         self.child_stay = None
#         self.child_change = None
#         if parent is not None:
#             self.depth = parent.depth + 1
#         else:
#             self.depth = 0
#         self.state = state
#         self.q_soft = 0
#         self.v_soft = 0
#         self.v_soft_argument = 0
#
#
#     def MENTS_value(self, explore=explore_MENTS):
#         if self.parent is None:
#             return 100
#         else:
#             lamda = explore/(1+math.log(self.N))
#             val = (1-lamda)*math.exp((self.q_soft-self.v_soft)/alfa) + lamda*0.5
#             return val
#
# class MCTS:
#     def __init__(self, state, depth_of_root, explore=explore_MENTS):    #initialize the tree
#         self.root_state = deepcopy(state)
#         self.root_node = Node(None, self.root_state)
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
#             elif node.child_stay.MENTS_value(self.explore) < node.child_change.MENTS_value(self.explore):
#                 node = node.child_change
#                 #print("decide to change")
#             else:
#                 node = node.child_stay
#                 #print("decide to stay")
#
#         # print("finish")
#         # print(node.state['signal___i0'])
#         # print(node.state['signal-t___i0'])
#
#         if node.state['signal___i0'] % 2 == 0:
#             if node.state['signal-t___i0'] < red_time: #if it's still red light
#                 tmp_state = self.one_step(node.state, self.stay)
#                 node.child_stay = Node(node, tmp_state)  # create the stay_child
#                 node.child_stay.reward = self.simulate(tmp_state)             #
#                 node.child_stay.q_soft = node.child_stay.reward + gama*node.reward
#             else: #if it's done being red light
#                 tmp_state = self.one_step(node.state, self.change)
#                 node.child_change = Node(node, tmp_state)  # create the change_child
#                 node.child_change.reward = self.simulate(tmp_state)  #
#                 node.child_change.q_soft = node.child_change.reward + gama * node.reward
#         else:
#             if node.state['signal-t___i0'] < min_green: #if it's too early to change
#                 tmp_state = self.one_step(node.state, self.stay)
#                 node.child_stay = Node(node, tmp_state)  # create the stay_child
#                 node.child_stay.reward = self.simulate(tmp_state)  #
#                 node.child_stay.q_soft = node.child_stay.reward + gama * node.reward
#             elif node.state['signal-t___i0'] == max_green: #if reached the maximum time without changing
#                 tmp_state = self.one_step(node.state, self.change)
#                 node.child_change = Node(node, tmp_state)  # create the change_child
#                 node.child_change.reward = self.simulate(tmp_state)  #
#                 node.child_change.q_soft = node.child_change.reward + gama * node.reward
#             else: #both changing and staying are legal moves
#                 tmp_state = self.one_step(node.state, self.stay)
#                 node.child_stay = Node(node, tmp_state)  # create the stay_child
#                 node.child_stay.reward = self.simulate(tmp_state)  #
#                 node.child_stay.q_soft = node.child_stay.reward + gama * node.reward
#                 tmp_state = self.one_step(node.state, self.change)
#                 node.child_change = Node(node, tmp_state)  # create the change_child
#                 node.child_change.reward = self.simulate(tmp_state)  #
#                 node.child_change.q_soft = node.child_change.reward + gama * node.reward
#
#         self.root_node, node.v_soft_argument = self.back_propagate(node, node.reward) #doing it like that make it so it simulates and gives a reward to the children, but don't backpropogate it upwards until it will reach them again
#         node.v_soft = alfa*math.log2(node.v_soft_argument+0.000001)
#
#     def one_step(self, state, action):
#         tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
#         _ = tmp_env.reset()
#         tmp_env.set_state(state)
#         next_state, reward, terminated, truncated, _ = tmp_env.step(action)
#         tmp_env.close()
#         return next_state
#
#
#     def simulate(self, state):
#         tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
#         _ = tmp_env.reset()
#         tmp_env.set_state(state)
#         agent = random_agent.RandomAgent(
#             action_space=tmp_env.action_space,
#             num_actions=tmp_env.max_allowed_actions)
#         simulated_reward = 0
#         for step in range(tmp_env.horizon-self.root_node.depth+1):
#             #print("inner step ", step)
#             action = agent.sample_action(state)
#             next_state, reward, terminated, truncated, _ = tmp_env.step(action)
#             simulated_reward = simulated_reward + reward
#             state = next_state
#             if truncated or terminated:
#                 break
#         tmp_env.close()
#         return simulated_reward
#
#
#     def back_propagate(self, node, reward):
#         node.N += 1
#         arg=0
#         while node.parent is not None:
#             v = node.v_soft
#             arg += math.exp(node.q_soft/alfa)
#             node = node.parent
#             node.N += 1
#             node.q_soft = reward + gama*v
#
#         arg += math.exp(node.q_soft / alfa)
#         return node, arg
#
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
#
#     def best_action(self):    #returns the best move for the next iteration
#         if self.root_node.child_stay.q_soft < self.root_node.child_change.q_soft:
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
#                 results.append(0)
#                 values.append(0)
#                 visits.append(0)
#                 queue.append(None)
#                 queue.append(None)
#             else:
#                 results.append(int(-max_reward+(current_node.total_reward / current_node.N)))  # Visit the node
#                 values.append(current_node.value(self.explore))
#                 visits.append(current_node.N)
#                 queue.append(current_node.child_change)                    # Enqueue the left and right children
#                 queue.append(current_node.child_stay)
#
#         return results,values,visits
#
#     def build_tree(self,result):
#         # Creating binary tree from given list
#         binary_tree = build(result)
#         print('Binary tree from list :\n',
#               binary_tree)
#         print('\nList from binary tree :',
#               binary_tree.values)
#
#
#     # def graphics(self,node, graph=None):
#     #     if graph is None:
#     #         graph = pgv.AGraph(directed=True)
#     #
#     #         # Create a node in the graph for the current tree node
#     #     if node is not None:
#     #         graph.add_node((node.total_reward / node.N))
#     #
#     #         # If there is a change child, add an edge and recurse
#     #         if node.child_change is not None:
#     #             graph.add_edge((node.total_reward / node.N), (node.child_change.total_reward / node.child_change.N))
#     #             graphics(node.child_change, graph)
#     #
#     #         # If there is a stay child, add an edge and recurse
#     #         if node.child_stay is not None:
#     #             graph.add_edge((node.total_reward / node.N), (node.child_stay.total_reward / node.child_stay.N))
#     #             graphics(node.child_stay, graph)
#     #
#     #     return graph
#     #
#     # def visualize_tree(self):
#     #     graph = graphics(self.root_node)
#     #     graph.layout(prog='dot')  # Layout the graph using dot
#     #     graph.draw('binary_tree.png')
#     #
#     # def build_list(self, node, list=None):
#     #     if node is not None:
#     #         list.append((node.total_reward / node.N))
#     #
#     #         # If there is a change child, add an edge and recurse
#     #         if node.child_change is not None:
#     #             build_list(node.child_change, list)
#     #
#     #         # If there is a stay child, add an edge and recurse
#     #         if node.child_stay is not None:
#     #             graph.add_edge((node.total_reward / node.N), (node.child_stay.total_reward / node.child_stay.N))
#     #             graphics(node.child_stay, graph)
#     #
#     #     else:
#     #         list.append(None)
#     #
#     #     return graph


import time
import math
from copy import deepcopy
import pyRDDLGym
from MCTS import random_agent #import RandomAgent

# import pygraphviz as pgv
from collections import deque
from binarytree import build

alfa = 1
explore_MENTS = 3
gama = 0.98
# max_reward = -17644.851216448555
max_reward = -12000
sim_reward = 0
red_time = 4
min_green = 6
max_green = 60

class Node:    #define the format of the nodes
    def __init__(self, parent, state):
        self.parent = parent
        self.N = 1
        self.total_reward = 0
        self.state_reward = 0
        self.child_stay = None
        self.child_change = None
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.state = state
        self.q_soft = 0
        self.v_soft = 0
        self.G = 0

    def value(self, explore=explore_MENTS):
        if self.parent is None:
            return 100
        else:
            lamda = explore / (1 + math.log(self.N))
            val = (1 - lamda) * math.exp((self.q_soft - self.v_soft) / alfa) + lamda * 0.5
            return val


class MCTS:
    def __init__(self, state, depth_of_root, explore=explore_MENTS):    #initialize the tree
        self.root_state = deepcopy(state)
        self.root_node = Node(None, self.root_state)
        self.root_node.depth = depth_of_root
        self.root_node.G = 100
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
            elif node.child_stay.value(self.explore) < node.child_change.value(self.explore):
                node = node.child_change
                #print("decide to change")
            else:
                node = node.child_stay
                #print("decide to stay")

        # print("finish")
        # print(node.state['signal___i0'])
        # print(node.state['signal-t___i0'])

        if node.state['signal___i0'] % 2 == 0:
            if node.state['signal-t___i0'] < red_time: #if it's still red light
                tmp_state = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state)  # create the stay_child
                node.child_stay.total_reward = self.simulate(tmp_state)             #
            else: #if it's done being red light
                tmp_state = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state)  # create the change_child
                node.child_change.total_reward = self.simulate(tmp_state)           #
        else:
            if node.state['signal-t___i0'] < min_green: #if it's too early to change
                tmp_state = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state)  # create the stay_child
                node.child_stay.total_reward = self.simulate(tmp_state)             #
            elif node.state['signal-t___i0'] == max_green: #if reached the maximum time without changing
                tmp_state = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state)  # create the change_child
                node.child_change.total_reward = self.simulate(tmp_state)           #
            else: #both changing and staying are legal moves
                tmp_state = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state)  # create the stay_child
                node.child_stay.total_reward = self.simulate(tmp_state)
                tmp_state = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state)  # create the change_child
                node.child_change.total_reward = self.simulate(tmp_state)

        self.root_node = self.back_propagate(node, node.total_reward) #doing it like that make it so it simulates and gives a reward to the children, but don't backpropogate it upwards until it will reach them again


    def one_step(self, state, action):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
        _ = tmp_env.reset()
        tmp_env.set_state(state)
        next_state, reward, terminated, truncated, _ = tmp_env.step(action)
        tmp_env.close()
        return next_state


    def simulate(self, state):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
        _ = tmp_env.reset()
        tmp_env.set_state(state)
        agent = random_agent.RandomAgent(
            action_space=tmp_env.action_space,
            num_actions=tmp_env.max_allowed_actions)
        simulated_reward = 0
        for step in range(tmp_env.horizon-self.root_node.depth+1):
            #print("inner step ", step)
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = tmp_env.step(action)
            simulated_reward = simulated_reward + reward
            state = next_state
            if truncated or terminated:
                break
        tmp_env.close()
        return simulated_reward


    def back_propagate(self, node, reward):
        initial_node = node
        arg = 0
        node.N += 1
        node.state_reward = reward

        if node.parent is not None:
            node.G = reward + gama * node.parent.G
            node.q_soft = node.G

        while node.parent is not None:
            arg += math.exp(node.q_soft/alfa)
            node = node.parent

        node = initial_node
        node.v_soft = alfa * math.log2(arg+0.000001)

        while node.parent is not None:
            v = node.v_soft
            node = node.parent
            node.N += 1
            node.q_soft = node.state_reward + gama*v

        return node


    def search(self, time_limit: int):
        start_time = time.process_time()

        while time.process_time() - start_time < time_limit:
            self.step()
            self.num_rollouts += 1

        self.run_time = time.process_time() - start_time

    def best_action(self):  # returns the best move for the next iteration

        if self.root_node.child_stay.q_soft < self.root_node.child_change.q_soft:
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
                results.append(int(-max_reward+(current_node.total_reward / current_node.N)))  # Visit the node
                values.append(current_node.value(self.explore))
                visits.append(current_node.N)
                queue.append(current_node.child_change)                    # Enqueue the left and right children
                queue.append(current_node.child_stay)

        return results,values,visits

    def build_tree(self,result):
        # Creating binary tree from given list
        binary_tree = build(result)
        print('Binary tree from list :\n',
              binary_tree)
        print('\nList from binary tree :',
              binary_tree.values)


    # def graphics(self,node, graph=None):
    #     if graph is None:
    #         graph = pgv.AGraph(directed=True)
    #
    #         # Create a node in the graph for the current tree node
    #     if node is not None:
    #         graph.add_node((node.total_reward / node.N))
    #
    #         # If there is a change child, add an edge and recurse
    #         if node.child_change is not None:
    #             graph.add_edge((node.total_reward / node.N), (node.child_change.total_reward / node.child_change.N))
    #             graphics(node.child_change, graph)
    #
    #         # If there is a stay child, add an edge and recurse
    #         if node.child_stay is not None:
    #             graph.add_edge((node.total_reward / node.N), (node.child_stay.total_reward / node.child_stay.N))
    #             graphics(node.child_stay, graph)
    #
    #     return graph
    #
    # def visualize_tree(self):
    #     graph = graphics(self.root_node)
    #     graph.layout(prog='dot')  # Layout the graph using dot
    #     graph.draw('binary_tree.png')
    #
    # def build_list(self, node, list=None):
    #     if node is not None:
    #         list.append((node.total_reward / node.N))
    #
    #         # If there is a change child, add an edge and recurse
    #         if node.child_change is not None:
    #             build_list(node.child_change, list)
    #
    #         # If there is a stay child, add an edge and recurse
    #         if node.child_stay is not None:
    #             graph.add_edge((node.total_reward / node.N), (node.child_stay.total_reward / node.child_stay.N))
    #             graphics(node.child_stay, graph)
    #
    #     else:
    #         list.append(None)
    #
    #     return graph