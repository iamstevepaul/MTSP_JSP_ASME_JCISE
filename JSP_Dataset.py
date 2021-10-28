"""
Author: Steve Paul
Date: 10/27/21 """

import os
import pickle
import torch
from torch import nn

n_samples = 1000
n_tasks = 100
n_machines = 30
n_jobs = 10
time_low = 10
time_high = 100



# tot_jobs = []
# for j in range(1, n_jobs+1):
#     n_ops = (task_job_mapping == j).to(torch.float32).sum()
#
#     op_ids = torch.arange(1,n_ops.item()+1)
#     tot_jobs.append(op_ids)

job_list = []


for i in range(n_samples):
    task_machine_accessibility = torch.randint(0, 2, (n_machines, n_tasks))
    task_machine_time = torch.randint(time_low, time_high + 1, (n_machines, n_tasks)) * task_machine_accessibility
    task_job_mapping = torch.randint(1, n_jobs + 1, (1, n_tasks))
    job_nums = torch.arange(1, n_jobs + 1)

    n_ops_in_jobs = (task_job_mapping.expand((n_jobs, n_tasks)).T == job_nums).to(torch.float32).sum(0)

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

    job_list.append(data)

dt = 0
