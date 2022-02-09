"""
Author: Steve Paul
Date: 10/27/21 """

import os
import pickle
import torch
from torch import nn
from utils.data_utils import check_extension, save_dataset

# n_samples = 1000
# n_tasks = 100
# n_machines = 30
# n_jobs = 10
# time_low = 10
# time_high = 100



# tot_jobs = []
# for j in range(1, n_jobs+1):
#     n_ops = (task_job_mapping == j).to(torch.float32).sum()
#
#     op_ids = torch.arange(1,n_ops.item()+1)
#     tot_jobs.append(op_ids)



n_samples = 100
job_smaples_infos = [[5,20,5], [10,50,10], [20,100,20], [50,200,50], [100,500,100]]

time_low = 10
time_high = 100


for job_smaples_info in job_smaples_infos:
    n_tasks = job_smaples_info[1]
    n_machines = job_smaples_info[2]
    n_jobs = job_smaples_info[0]

# (data["task_job_mapping"].expand(2, 100, 100) != data["task_job_mapping"].permute(0, 2, 1)).to(
#     torch.float32)

    job_list = []

    for i in range(n_samples):
        task_machine_accessibility = torch.randint(0, 2, (n_machines, n_tasks))
        for i in range(n_tasks):
            task_machine_accessibility[:, i] = torch.randint(0, 2, (1, n_machines))
            if task_machine_accessibility[:, i].max() == 0:
                task_machine_accessibility[torch.randint(0, n_machines, (1,)), i] = 1
        # first job will be waiting with index 0
        task_machine_accessibility = torch.cat(
            (torch.ones((n_machines, 1), dtype=torch.int64), task_machine_accessibility), dim=1)

        task_machine_time = torch.randint(time_low, time_high + 1,
                                          (n_machines, n_tasks + 1)) * task_machine_accessibility
        task_machine_time[:, 0] = 0

        task_job_mapping = torch.randint(1, n_jobs + 1, (1, n_tasks))  # wait not considered
        while task_job_mapping.unique().shape[0] != n_jobs:
            task_job_mapping = torch.randint(1, n_jobs + 1, (1, n_tasks))
        job_nums = torch.arange(1, n_jobs + 1)

        n_ops_in_jobs = (task_job_mapping.expand((n_jobs, n_tasks)).T == job_nums).to(torch.float32).sum(
            0)  # wait not considered
        operations_availability = torch.zeros((n_tasks)).to(torch.float32)

        # 0-idle, 1-wait, 2- engaged, 4- done (no more operations available)
        machine_status = torch.zeros((n_machines, 1))

        data = {}
        data["n_tasks"] = n_tasks
        data["n_machines"] = n_machines
        data["n_jobs"] = n_jobs
        data["time_low"] = time_low
        data["time_high"] = time_high
        data["task_machine_accessibility"] = task_machine_accessibility
        data["task_machine_time"] = task_machine_time
        data["task_job_mapping"] = task_job_mapping
        data["job_nums"] = job_nums
        data["n_ops_in_jobs"] = n_ops_in_jobs
        data["machine_status"] = machine_status

        adjacency = (data["task_job_mapping"].expand(n_tasks, n_tasks) != data["task_job_mapping"].permute(1, 0)).to(
            torch.float32)

        ops_nz = (
        torch.triu((data["task_job_mapping"].expand(n_tasks, n_tasks) == data["task_job_mapping"].permute(1, 0)).to(
            torch.float32), diagonal=1))

        operations_next = torch.zeros((n_tasks, 1)).to(torch.float32)

        for j in range(n_tasks):
            sp = ops_nz[j, :].nonzero(as_tuple=True)
            if task_job_mapping[0, j] not in task_job_mapping[0, 0:j]:
                operations_availability[j] = 1
            if sp[0].size()[0] > 0:
                operations_next[j, 0] = sp[0][0] + 1
                for el in sp[0]:
                    adjacency[j, el] = 1

        operations_availability = torch.cat((torch.ones((1), dtype=torch.float32), operations_availability), dim=0)

        # indz = ops_nz.nonzero(as_tuple=True)

        data["adjacency"] = adjacency  # wait not considered
        data["operations_next"] = operations_next  # wait not considered
        data["operations_availability"] = operations_availability

        job_list.append(data)
    datadir = 'Data/JSP/'
    os.makedirs(datadir, exist_ok=True)
    with open(datadir+'J'+str(n_jobs)+'_T_'+str(n_tasks)+'_M_'+str(n_machines)+'.pickle', 'wb') as fl:
        pickle.dump(job_list, fl, protocol=pickle.HIGHEST_PROTOCOL)

