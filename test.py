from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
from MCTS import agent

import pyRDDLGym.core.policy
#exp_arr = [400,400,400,400,400,100,100,100,100,100,1,5,10,100,500,1000,2000,5000,10000]
exp_arr = [0.2]

rewards_arr = []
for explore in exp_arr:
    env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)

    agent = agent.mcts_Agent(
        action_space=env.action_space,
        num_actions=env.max_allowed_actions,
        explore=explore,
        search_time=30)

    cmlt_reward = 0
    state, _ = env.reset()
    for step in range(env.horizon):
        print("step =", step,"| reward =", cmlt_reward)
        #env.render(to_display=True)
        action = agent.sample_action(state,step)
        next_state, reward, terminated, truncated, _ = env.step(action)
        cmlt_reward = cmlt_reward + reward
        state = next_state
        if truncated or terminated:
            break
    #print(f'Episode ended with cumulative reward {cmlt_reward}')
    #print("for explore =",explore,"the reward is", cmlt_reward)
    env.close()
    rewards_arr.append(cmlt_reward)
print(exp_arr)
print(rewards_arr)
#print("average for explore=100",(rewards_arr[0]+rewards_arr[1]+rewards_arr[2]+rewards_arr[3]+rewards_arr[4])/5)
#print("average for explore=100",(rewards_arr[5]+rewards_arr[6]+rewards_arr[7]+rewards_arr[8]+rewards_arr[9])/5)
