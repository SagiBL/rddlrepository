import time
import math
from copy import deepcopy
import pyRDDLGym
import agent1
#import pygraphviz as pgv
from collections import deque
from binarytree import build

explore_c=10000
#max_reward = -17644.851216448555
max_reward = -10000
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
        else:
            self.depth = 0
        self.state = state


    def value(self, explore=explore_c):  #calculate the UCB
        if self.parent is None:
            return 100
        else:
            val = int(-max_reward+(self.total_reward / self.N) + explore * math.sqrt(math.log(self.parent.N) / self.N))
            return val


class MCTS:
    def __init__(self, state):    #initialize the tree
        self.root_state = deepcopy(state)
        self.root_node = Node(None, self.root_state)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.stay = {'advance___i0':0}    #this is dumb but it works for now
        self.change = {'advance___i0':1}


    def step(self):   # preforms one step of expansion and simulates it
        node = self.root_node
        while node.child_stay is not None or node.child_change is not None:   #choose the best child and advance state accordingly
            if node.child_stay is None:
                node = node.child_change
            elif node.child_change is None:
                node = node.child_stay
            elif node.child_stay.value() < node.child_change.value():
                node = node.child_change
                if node.parent.parent is not None:
                    print("went to change")
            else:
                node = node.child_stay
                if node.parent.parent is not None:
                    print("went to stay")

            # if ((node.state["signal___i0"]%2 == 0) and (node.state["signal-t___i0"] < 4)) or ((node.state["signal___i0"] % 2 == 1) and (node.state["signal-t___i0"] < 60)) :
            #     tmp_state = self.one_step(node.state, self.stay)
            #     node.child_stay = Node(node, tmp_state)     #create the stay_child
            #     sim_reward = self.simulate(node.state)
            #     self.back_propagate(node,sim_reward)
            #     return self.stay
            #
            # elif ((node.state["signal___i0"] % 2 == 0) and (node.state["signal-t___i0"] >= 4)) or ((node.state["signal___i0"] % 2 == 1) and (node.state["signal-t___i0"] >= 6)):
            #     tmp_state = self.one_step(node.state, self.change)
            #     node.child_change = Node(node, tmp_state)   #create the change_child
            #     sim_reward = self.simulate(node.state)
            #     self.back_propagate(node,sim_reward)
            #     return self.change
        print(node.state['signal___i0'])
        print(node.state['signal-t___i0'])
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


    # def simulate(self,state):       #rollout the result to get final cumulative reward
    #     simulated_reward = 0
    #     for step in range(1,200):#for 200 step, to simulate the original 200 steps intersection
    #         action = self.RandomAgent.sample_action(state)
    #         next_state, reward, terminated, truncated, _ = self.env.step(action)
    #         simulated_reward = simulated_reward + reward
    #         state = next_state
    #         if truncated or terminated:
    #             break
    #     return simulated_reward

    def simulate(self, state):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
        _ = tmp_env.reset()
        tmp_env.set_state(state)
        agent = agent1.RandomAgent(
            action_space=tmp_env.action_space,
            num_actions=tmp_env.max_allowed_actions)
        simulated_reward = 0
        for step in range(tmp_env.horizon):
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = tmp_env.step(action)
            simulated_reward = simulated_reward + reward
            state = next_state
            if truncated or terminated:
                break
        tmp_env.close()
        return simulated_reward


    def back_propagate(self, node, reward):
        while node.parent is not None:
            node.N += 1
            node.total_reward += reward
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
        if self.root_node.child_stay.value() < self.root_node.child_change.value():
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
                values.append(7)
                visits.append(0)
                queue.append(None)
                queue.append(None)
            else:
                results.append(int(-max_reward+(current_node.total_reward / current_node.N)))  # Visit the node
                values.append(current_node.value())
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