import time
import math
import random
from copy import deepcopy
import pyRDDLGym
from MCTS import random_agent #import RandomAgent

# import pygraphviz as pgv
from collections import deque
from binarytree import build

alfa = 0.1
explore_MENTS = 1
gama = 0.8
max_reward = -12000
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
        self.q_soft_stay = 0
        self.q_soft_change = 0
        self.IsChange = change

        self.v_soft = 0
        self.G = 0
        # self.actions = 1
        # if self.state['signal___i0'] % 2 == 1 and max_green > self.state['signal-t___i0'] >= min_green:
        #     self.actions = 2

    def Jpolicy_stay(self, debugg, explore=explore_MENTS):
        stay_prob = random.uniform(0, 1)
        change_prob = 1-stay_prob
        lamda = (explore*default_action_number) / (1 + math.log(self.N))
        val_stay = (1 - lamda) * math.exp((self.q_soft_stay - self.v_soft) / alfa) + lamda * stay_prob
        val_change = (1 - lamda) * math.exp((self.q_soft_change - self.v_soft) / alfa) + lamda * change_prob
        if debugg:
            x=1
            #print(x)
        return val_stay > val_change


class MCTS:
    def __init__(self, state, depth_of_root, explore=explore_MENTS):    #initialize the tree
        self.root_state = deepcopy(state)
        self.root_node = Node(None, self.root_state, False)
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
        debugg = True
        while node.child_stay is not None or node.child_change is not None:   #choose the best child and advance state accordingly
            if node.child_stay is None:
                node = node.child_change
                #print("must change")
            elif node.child_change is None:
                node = node.child_stay
                #print("must stay")
            elif node.Jpolicy_stay(debugg=debugg, explore=self.explore):
                node = node.child_stay
                #print("decide to stay")
            else:
                node = node.child_change
                #print("decide to change")
            debugg = False

        # print("finish")
        # print(node.state['signal___i0'])
        # print(node.state['signal-t___i0'])

        if node.state['signal___i0'] % 2 == 0:
            if node.state['signal-t___i0'] < red_time: #if it's still red light
                tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state, False)  # create the stay_child

            else: #if it's done being red light
                tmp_state, node.reward_change = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state, True)  # create the change_child

        else:
            if node.state['signal-t___i0'] < min_green: #if it's too early to change
                tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state, False)  # create the stay_child

            elif node.state['signal-t___i0'] >= max_green: #if reached the maximum time without changing
                tmp_state, node.reward_change = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state, True)  # create the change_child

            else: #both changing and staying are legal moves
                tmp_state, node.reward_stay = self.one_step(node.state, self.stay)
                node.child_stay = Node(node, tmp_state, False)  # create the stay_child

                tmp_state, node.reward_change = self.one_step(node.state, self.change)
                node.child_change = Node(node, tmp_state, True)  # create the change_child
                #print("stay_reward",node.reward_stay,"|| change_reward",node.reward_change)


        node.reward_stay = node.reward_stay/100
        node.reward_change = node.reward_change/100 #to make the values a bit smaller
        self.root_node = self.back_propagate(node) #doing it like that make it so it simulates and gives a reward to the children, but don't backpropogate it upwards until it will reach them again


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
        for step in range(tmp_env.horizon):# -self.root_node.depth+1):
            #print("inner step ", step)
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = tmp_env.step(action)
            simulated_reward = simulated_reward + reward
            state = next_state
            if truncated or terminated:
                break
        tmp_env.close()
        return simulated_reward

    def calc_g(self, node):
        original_node = node
        G = 0
        while node.parent is not None:
            node = node.parent
            if node.IsChange:
                G = G + node.reward_change * (gama ** node.depth)
            else:
                G = G + node.reward_stay * (gama ** node.depth)
        return original_node, G

    def back_propagate(self, node):
        first_time = True
        while node.parent is not None:
            node.N += 1
            if first_time:
                node, G = self.calc_g(node)
                if node.IsChange:
                    node.q_soft_stay = node.reward_change + gama * G
                else:
                    node.q_soft_change = node.reward_stay + gama * G
                first_time = False
            else:
                if node.child_stay is not None: node.q_soft_stay = node.reward_stay + gama * node.child_stay.v_soft
                if node.child_change is not None: node.q_soft_change = node.reward_change + gama * node.child_change.v_soft
            node.v_soft = alfa * math.log(math.exp(node.q_soft_stay/alfa) + math.exp(node.q_soft_change/alfa))
            node = node.parent
        node.N += 1
        if first_time:
            node.q_soft_stay = node.reward_stay
            node.q_soft_change = node.reward_change
        else:
            node.q_soft_stay = node.reward_stay + gama * node.child_stay.v_soft
            node.q_soft_change = node.reward_change + gama * node.child_change.v_soft
        node.v_soft = alfa * math.log(math.exp(node.q_soft_stay / alfa) + math.exp(node.q_soft_change / alfa))
        return node

    # def back_propagate(self, node):
    #     first_time = True
    #     while node.parent is not None:
    #         if node.IsChange:
    #             G = node.parent.q_soft_change
    #         else:
    #             G = node.parent.q_soft_stay
    #         node.N += 1
    #         if first_time:
    #             node.q_soft_stay = node.reward_stay + gama * G
    #             node.q_soft_change = node.reward_change + gama * G
    #             first_time = False
    #         else:
    #             if node.child_stay is not None: node.q_soft_stay = node.reward_stay + gama * node.child_stay.v_soft
    #             if node.child_change is not None: node.q_soft_change = node.reward_change + gama * node.child_change.v_soft
    #         node.v_soft = alfa * math.log(math.exp(node.q_soft_stay/alfa) + math.exp(node.q_soft_change/alfa))
    #         node = node.parent
    #     node.N += 1
    #     if first_time:
    #         node.q_soft_stay = node.reward_stay
    #         node.q_soft_change = node.reward_change
    #     else:
    #         node.q_soft_stay = node.reward_stay + gama * node.child_stay.v_soft
    #         node.q_soft_change = node.reward_change + gama * node.child_change.v_soft
    #     node.v_soft = alfa * math.log(math.exp(node.q_soft_stay / alfa) + math.exp(node.q_soft_change / alfa))
    #     return node

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

    def build_tree(self,visits):
        # Creating binary tree from given list
        binary_tree = build(visits)
        print('Binary tree from list :\n',
              binary_tree)
        print('\nList from binary tree :',
              binary_tree.values)