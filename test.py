from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
import agent1

import pyRDDLGym.core.policy
#exp_arr = [1,5,10,20,100,500,1000,2000,5000,10000]
exp_arr = [100]
rewards_arr = []
for explore in exp_arr:
    env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)

    agent = agent1.mcts_Agent(
        action_space=env.action_space,
        num_actions=env.max_allowed_actions,
        explore=explore,
        search_time=10)

    cmlt_reward = 0
    state, _ = env.reset()
    for step in range(1,200):
        print("step",step)
        #env.render(to_display=True)
        action = agent.sample_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        cmlt_reward = cmlt_reward + reward
        state = next_state
        if truncated or terminated:
            break
    #print(f'Episode ended with cumulative reward {cmlt_reward}')
    print("for explore =",explore,"the reward is", cmlt_reward)
    env.close()
    rewards_arr.append(cmlt_reward)
print(exp_arr)
print(rewards_arr)