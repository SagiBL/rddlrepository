import random
import time
import math
from copy import deepcopy
import pyRDDLGym
import agent1
from test import tmp_state

explore_c=2
max_reward = 17644.851216448555
sim_reward = 0

class Node:    #define the format of the nodes
    def __init__(self, parent, state):
        self.parent = parent
        self.N = 0
        self.total_reward = 0
        self.child_stay = None
        self.child_change = None
        self.depth = parent.depth + 1
        self.state = state


    def value(self, explore=explore_c):  #calculate the UCB
        return -max_reward+(self.total_reward / self.N) + explore * math.sqrt(math.log(self.parent.N) / self.N)


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

        while (len(node.child_stay)+len(node.child_change)) != 0:   #choose the best child and advance state accordingly
            if len(node.child_stay)==0:
                node = node.child_change
            elif len(node.child_change)==0:
                node = node.child_stay
            elif node.child_stay.value < node.child_change.value:
                node = node.child_change
            else:
                node = node.child_stay
        if node.depth>200 :             #the simulation reached maximal depth
            return "sim depth over"
        else:
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

            if node.state['signal___i0'] % 2 == 0:
                if node.state['signal-t___i0'] < 4: #if it's still red light
                    tmp_state = self.one_step(node.state, self.stay)
                    node.child_stay = Node(node, tmp_state)  # create the stay_child
                else: #if it's done being red light
                    tmp_state = self.one_step(node.state, self.change)
                    node.child_change = Node(node, tmp_state)  # create the change_child
            else:
                if node.state['signal-t___i0'] < 6: #if it's too early to change
                    tmp_state = self.one_step(node.state, self.stay)
                    node.child_stay = Node(node, tmp_state)  # create the stay_child
                elif node.state['signal-t___i0'] == 60: #if reached the maximum time without changing
                    tmp_state = self.one_step(node.state, self.change)
                    node.child_change = Node(node, tmp_state)  # create the change_child
                else: #both changing and staying are legal moves
                    tmp_state = self.one_step(node.state, self.stay)
                    node.child_stay = Node(node, tmp_state)  # create the stay_child
                    tmp_state = self.one_step(node.state, self.change)
                    node.child_change = Node(node, tmp_state)  # create the change_child

    def one_step(self, state, action):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
        _ = tmp_env.reset()
        _ = tmp_env.set_state(state)
        next_state, reward, terminated, truncated, _ = tmp_env.step(action)
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

    def simulate(self,state):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
        _ = tmp_env.reset()
        _ = tmp_env.set_state(state)
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
        while node is not None:
            node.N += 1
            node.total_reward += reward
            node = node.parent



    def search(self, time_limit: int):
        start_time = time.process_time()

        while time.process_time() - start_time < time_limit:
            self.step()
            self.num_rollouts += 1

        self.run_time = time.process_time() - start_time


    def best_action(self):    #returns the best move for the next iteration
        if self.root_node.child_stay.value < self.root_node.child_change.value:
            return self.change
        else:
            return self.stay


    def statistics(self):      #return the overall calculation statistics
        return self.num_rollouts, self.run_time
