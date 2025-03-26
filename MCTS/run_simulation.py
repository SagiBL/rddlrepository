from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
from MCTS import agent
import numpy as np
import pyRDDLGym.core.policy

seed = None
search_time = 30
instance = 0
use_uct = False     ### choose to use uct or ments
                    ### True if uct and False if ments
exp_arr = [500] if use_uct else [1]
ToPrint = False     ### choose to print

def test(exp_arr, search_time, instance, min_reward):
    rewards_arr = []
    exp_arr = exp_arr
    for explore in exp_arr:
        env = pyRDDLGym.make('TrafficBLX_SimplePhases', instance=instance)

        MCTSagent = agent.mcts_Agent(
            action_space=env.action_space,
            num_actions=env.max_allowed_actions,
            explore=explore,
            search_time=search_time,
            instance=instance,
            min_reward=min_reward,
            use_uct = use_uct,
            ToPrint=ToPrint)

        cmlt_reward = 0
        state, _ = env.reset(seed=seed)
        for step in range(env.horizon):
            env.render(to_display=True)
            action = MCTSagent.sample_action(state, step)
            print_action = "stay" if action=={'advance___i0': 0} else "change"
            print("step =", step, "| reward =", cmlt_reward,"| action =", print_action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            cmlt_reward = cmlt_reward + reward
            state = next_state
            if truncated or terminated:
                break
        env.close()
        rewards_arr.append(cmlt_reward)
        print("cmlt_reward =", cmlt_reward)
    return rewards_arr



def find_min_max_reward(instance):
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


print("calculating min_reward ...")
min_reward, max_reward = find_min_max_reward(instance)
print("instance =", instance, "|| min_reward =", min_reward, "|| max_reward =", max_reward)

rewards_arr = test(exp_arr, search_time, instance, min_reward)
print("explore =", exp_arr)
print("reward =", rewards_arr)

