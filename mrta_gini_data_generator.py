import pickle
import torch
import numpy as np
import random

if __name__ == "__main__":
    n_samples = 1
    max_n_agent = 10


    agents_ids = torch.randint(0, 4, (n_samples, 1))

    groups = torch.randint(1, 3, (n_samples, 1))

    max_range = 4
    max_capacity = 10
    max_speed = .06

    dist = torch.randint(1, 5, (n_samples, 1))

    data = []

    n_tasks = 1000

    group_list = [1, 2]
    instance_list = [0, 1, 2]
    ratio_deadline_list = [1, 2, 3, 4]
    robotSize_list = (torch.tensor([2,3,5,7])*10).tolist()
    pkl_file_names = []
    for G in group_list:
        for D in ratio_deadline_list:
            for R in robotSize_list:
                for I in instance_list:
                    n_agents = R
                    agents_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100)
                    loc = torch.FloatTensor(n_tasks, 2).uniform_(0, 1)
                    workload = torch.FloatTensor(n_tasks).uniform_(.2, .2)
                    d_low = (((loc[:, None, :].expand((n_tasks, max_n_agent, 2)) - agents_location[None].expand(
                        (n_tasks, max_n_agent, 2))).norm(2, -1).max() / max_speed) + 20).to(torch.int64) + 1
                    d_high = ((35) * (45) * 100 / (380) + d_low).to(torch.int64) + 1
                    d_low = d_low * (.5 * G)
                    d_high = ((d_high * (.5 * G) / 10).to(torch.int64) + 1) * 10
                    deadline_normal = (torch.rand(n_tasks, 1) * (d_high - d_low) + d_low).to(torch.int64) + 1
                    n_norm_tasks = int(D * 25*n_tasks/100)
                    rand_mat = torch.rand(n_tasks, 1)
                    k = n_norm_tasks # For the general case change 0.25 to the percentage you need
                    k_th_quant = torch.topk(rand_mat.T, k, largest=False)[0][:, -1:]
                    bool_tensor = rand_mat <= k_th_quant
                    normal_dist_tasks = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))

                    slack_tasks = (normal_dist_tasks - 1).to(torch.bool).to(torch.int64)

                    normal_dist_tasks_deadline = normal_dist_tasks * deadline_normal

                    slack_tasks_deadline = slack_tasks * d_high

                    deadline_final = (normal_dist_tasks_deadline + slack_tasks_deadline)

                    robots_start_location = (torch.randint(0, 101, (n_agents, 2)).to(torch.float) / 100).to(
                        device=deadline_final.device)

                    robots_work_capacity = (torch.randint(1, 3, (n_agents, 1), dtype=torch.float,
                                                         device=deadline_final.device).view(-1) / 100)

                    loc_data = {
                        'loc': loc.numpy(),
                        'workload': workload.numpy(),
                        'deadline': deadline_final.view(-1).numpy(),
                        'n_agents': R,
                    }

                    robot_data = {
                        'robots_loc': robots_start_location.numpy(),
                        'robots_capacity': robots_work_capacity.numpy()
                    }

                    data = {
                        'loc_data': loc_data,
                        'robot_data': robot_data
                    }
                    agent_name = "a" + str(R) + "i0" + str(I)
                    data_name = "data/mrta_gini/n_1000/" +"r" + str(G) + str(D) + agent_name + '_n_1000'
                    pkl_file_name = data_name + ".pkl"
                    pkl_file_names.append(pkl_file_name)
                    pickle.dump(data, open(pkl_file_name, "wb"))
    all_files = "data/mrta_gini/gini_data_sets_n_1000.pkl"
    file_n = open(all_files, 'wb')
    pickle.dump(pkl_file_names, file_n)
    file_n.close()


    # for i in range(n_samples):
    #     n_agents = n_agents_available[agents_ids[i, 0].item()].item()
    #     agents_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100)
    #
    #     loc = torch.FloatTensor(n_tasks, 2).uniform_(0, 1)
    #     workload = torch.FloatTensor(n_tasks).uniform_(.2, .2).numpy()
    #     d_low = (((loc[:, None, :].expand((n_tasks, max_n_agent, 2)) - agents_location[None].expand((n_tasks, max_n_agent, 2))).norm(2, -1).max()/max_speed) + 20).to(torch.int64) + 1
    #     d_high = ((35)*(45)*100/(380) + d_low).to(torch.int64) + 1
    #     d_low = d_low*(.5*groups[i, 0])
    #     d_high = ((d_high * (.5 * groups[i, 0])/10).to(torch.int64) + 1)*10
    #     deadline_normal = (torch.rand(n_tasks, 1) * (d_high - d_low) + d_low).to(torch.int64) + 1
    #
    #     n_norm_tasks = dist[i, 0]*25
    #     rand_mat = torch.rand(n_tasks, 1)
    #     k = n_norm_tasks.item()  # For the general case change 0.25 to the percentage you need
    #     k_th_quant = torch.topk(rand_mat.T, k, largest=False)[0][:, -1:]
    #     bool_tensor = rand_mat <= k_th_quant
    #     normal_dist_tasks = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))
    #
    #     slack_tasks = (normal_dist_tasks - 1).to(torch.bool).to(torch.int64)
    #
    #     normal_dist_tasks_deadline = normal_dist_tasks*deadline_normal
    #
    #     slack_tasks_deadline = slack_tasks*d_high
    #
    #     deadline_final = normal_dist_tasks_deadline + slack_tasks_deadline
    #
    #     robots_start_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100).to(
    #         device=deadline_final.device)
    #
    #     robots_work_capacity = torch.randint(1, 3, (max_n_agent, 1), dtype=torch.float,
    #                                          device=deadline_final.device).view(-1) / 100
    #
    #     case_info = {
    #                 'loc': loc,
    #                 'depot': torch.FloatTensor(1,2).uniform_(0, 1),
    #                 'deadline':deadline_final.to(torch.float).view(-1),
    #                 'workload': workload,
    #                 'initial_size':100,
    #                 'n_agents': torch.tensor([[n_agents]]),
    #                 'max_n_agents': torch.tensor([[max_n_agent]]),
    #                 'max_range':max_range,
    #                 'max_capacity':max_capacity,
    #                 'max_speed':max_speed,
    #                 'enable_capacity_constraint':False,
    #                 'enable_range_constraint':False,
    #                 'robots_start_location': robots_start_location,
    #                 'robots_work_capacity': robots_work_capacity
    #             }
    #
    #     data.append(case_info)
    #
    #
    # dt = 0
    # pickle.dump(data, open("mrta_gini_validation_init.pkl", 'wb'))