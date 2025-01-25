from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
from MCTS import agent
import numpy as np
import pyRDDLGym.core.policy

exp_arr = [500]
search_time = 1
instance = 1

def test(exp_arr, search_time, instance, min_reward):
    rewards_arr = []
    exp_arr = exp_arr
    for explore in exp_arr:
        env = pyRDDLGym.make('TrafficBLX_SimplePhases', instance=instance)

        agent2 = agent.mcts_Agent(
            action_space=env.action_space,
            num_actions=env.max_allowed_actions,
            explore=explore,
            search_time=search_time,
            instance=instance,
            min_reward=min_reward)

        cmlt_reward = 0
        state, _ = env.reset()
        for step in range(env.horizon):
            print("step =", step, "| reward =", cmlt_reward)
            env.render(to_display=True)
            action = agent2.sample_action(state, step)
            next_state, reward, terminated, truncated, _ = env.step(action)
            cmlt_reward = cmlt_reward + reward
            state = next_state
            if truncated or terminated:
                break
        print("for explore =",explore,"the reward is", cmlt_reward)
        env.close()
        rewards_arr.append(cmlt_reward)

    return rewards_arr



def find_min_max_reward(instance):
    print("calculating min_reward ...")
    rewards_arr = []

    for i in range(100):
        env = pyRDDLGym.make('TrafficBLX_SimplePhases', instance=instance)

        agent1 = agent.RandomAgent(
            action_space=env.action_space,
            num_actions=env.max_allowed_actions)

        cmlt_reward = 0
        state, _ = env.reset()
        for step in range(env.horizon):
            action = agent1.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            cmlt_reward = cmlt_reward + reward
            state = next_state
            if truncated or terminated:
                break
        env.close()
        rewards_arr.append(cmlt_reward)

    rewards_arr = np.array(rewards_arr)
    return np.min(rewards_arr), np.max(rewards_arr)


#min_reward, max_reward = find_min_max_reward(instance)
min_reward = -13000
rewards_arr = test(exp_arr, search_time, instance, min_reward)
#print("instance =", instance, "|| min_reward =", min_reward, "|| max_reward =", max_reward)
print("explore =", exp_arr)
print("reward =", rewards_arr)

