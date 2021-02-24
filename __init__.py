from PBRSMA_E001_Agent import Agent
from PBRSMA_E001_Env import MultiAgentEnv
import numpy as np
import pandas as pd
if __name__ == '__main__':
    #Do experiments both when PBRS is False or True
    PBRS = [False, True]
    for PBRS_flag in PBRS:
        data_1 = pd.DataFrame()
        data_2 = pd.DataFrame()
        for i in range(0, 1):
            if PBRS_flag:
                print("PBRS_Exp-----------------------------------")
            else:
                print("Non-PBRS_Exp---------------------------------")
            env = MultiAgentEnv(PBRS_flag)
            env.do_experiment()
            print(env.moves_to_goal[0], "agent_1")
            print(env.moves_to_goal[1], "agent_2")
            print(env.agents[0].q_table)
            print("q_table_pure: ")
            print(env.agents[0].q_table_pure)
            data_1.insert(data_1.shape[1], i + 1, env.moves_to_goal[0])
            data_2.insert(data_2.shape[1], i + 1, env.moves_to_goal[1])
        col_mean_1 = np.mean(data_1.values, axis=1)
        col_mean_2 = np.mean(data_2.values, axis=1)
        col_name = ["agent_1", "agent_2"]
        data = pd.DataFrame(columns=col_name)
        data["agent_1"] = col_mean_1
        data["agent_2"] = col_mean_2
        if PBRS_flag:
            data.to_csv("PBSR_result1.csv")
        else:
            data.to_csv("result1.csv")