from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
import agent1

import pyRDDLGym.core.policy

stay = {'advance___i0': 0}
change = {'advance___i0': 1}

seed_arr = [1,2,3,4,5]
for seed in seed_arr:
    print()
    print("***********")
    print("testing seed", seed)
    print("***********")
    exp_arr = [100,200,300]
    rewards_arr = []
    for explore in exp_arr:
        env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)

        agent = agent1.mcts_Agent(
            action_space=env.action_space,
            num_actions=env.max_allowed_actions,
            explore=explore,
            search_time=10)

        cmlt_reward = 0
        state, _ = env.reset(seed=seed)
        for step in range(env.horizon):
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            cmlt_reward = cmlt_reward + reward
            state = next_state
            if truncated or terminated:
                break
        env.close()
        rewards_arr.append(cmlt_reward)
    print(exp_arr)
    print(rewards_arr)
    print("average for MCTS",(sum(rewards_arr)/len(rewards_arr)))

    env = pyRDDLGym.make('TrafficBLX_SimplePhases', 1)

    cmlt_reward = 0
    state, _ = env.reset(seed=seed)
    for step in range(env.horizon):
        next_state, reward, terminated, truncated, _ = env.step(change)
        cmlt_reward = cmlt_reward + reward
        state = next_state
        if truncated or terminated:
            break
    print(f'Fixed episode ended with cumulative reward {cmlt_reward}')
    env.close()
