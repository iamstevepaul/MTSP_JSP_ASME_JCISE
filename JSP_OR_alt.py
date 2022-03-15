"""
Author: Steve Paul 
Date: 2/8/22 """
#!/usr/bin/env python3
# Copyright 2010-2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Solves a flexible jobshop problems with the CP-SAT solver.

A jobshop is a standard scheduling problem when you must sequence a
series of task_types on a set of machines. Each job contains one task_type per
machine. The order of execution and the length of each job on each
machine is task_type dependent.

The objective is to minimize the maximum completion time of all
jobs. This is called the makespan.
"""

# overloaded sum() clashes with pytype.
# pytype: disable=wrong-arg-types

import collections
import os
from ortools.sat.python import cp_model
import pickle
import csv

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """Called at each new solution."""
        print('Solution %i, time = %f s, objective = %i' %
              (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1


def flexible_jobshop(jobs, num_machines):
    time_in_Seconds = 50
    """Solve a small flexible jobshop problem."""
    # Data part.
    # jobs = [  # task = (processing_time, machine_id)
    #     [  # Job 0
    #         [(3, 0), (1, 1), (5, 2)],  # task 0 with 3 alternatives
    #         [(2, 0), (4, 1), (6, 2)],  # task 1 with 3 alternatives
    #         [(2, 0), (3, 1), (1, 2)],  # task 2 with 3 alternatives
    #     ],
    #     [  # Job 1
    #         [(2, 0), (3, 1), (4, 2)],
    #         [(1, 0), (5, 1), (4, 2)],
    #         [(2, 0), (1, 1), (4, 2)],
    #     ],
    #     [  # Job 2
    #         [(2, 0), (1, 1), (4, 2)],
    #         [(2, 0), (3, 1), (4, 2)],
    #         [(3, 0), (1, 1), (5, 2)],
    #     ],
    # ]
    #
    # jobs = [  # task = (processing_time, machine_id)
    #     [  # Job 0
    #         [(3, 0), (1, 1), (5, 2)],  # task 0 with 3 alternatives
    #         [(2, 0), (4, 1)],  # task 1 with 3 alternatives
    #         [(3, 1), (1, 2)],  # task 2 with 3 alternatives
    #     ],
    #     [  # Job 1
    #         [(2, 0), (4, 2)],
    #         [(1, 0), (5, 1), (4, 2)],
    #         [(2, 0), (1, 1), (4, 2)],
    #     ],
    #     [  # Job 2
    #         [(2, 0)],
    #         [(2, 0), (3, 1), (4, 2)],
    #         [(3, 0), (1, 1), (5, 2)],
    #     ],
    # ]

    # jobs = [  # task = (machine_id, processing_time). accessibility can be added here
    #     [[(0, 3), (1, 2)], [(1, 2), (2,3)], [(2, 2)]],  # Job0
    #     [[(0, 2)], [(1,2),(2, 1)], [(1, 4)]],  # Job1
    #     [[(1, 4)], [(2, 3)]]  # Job2
    # ]

    num_jobs = len(jobs)
    all_jobs = range(num_jobs)

    # num_machines = n_machines
    all_machines = range(num_machines)

    # Model the flexible jobshop problem.
    model = cp_model.CpModel()

    horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration
    worst = 0
    for job in jobs:
        for task in job:
            ls = []
            for alternative in task:
                # max_task_duration = max(max_task_duration, alternative[0])
                ls.append(alternative[0])
            worst += max(ls)

    print('Horizon = %i' % horizon)

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = '_j%i_t%i' % (job_id, task_id)
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration,
                                       'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end,
                                            'interval' + suffix_name)

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar('presence' + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence,
                        'interval' + alt_suffix)
                    l_presences.append(l_presence)

                    # Link the master variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.Add(sum(l_presences) == 1)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_in_Seconds
    solution_printer = SolutionPrinter()
    status = solver.Solve(model)

    # Print final solution.
    for job_id in all_jobs:
        print('Job %i:' % job_id)
        for task_id in range(len(jobs[job_id])):
            start_value = solver.Value(starts[(job_id, task_id)])
            machine = -1
            duration = -1
            selected = -1
            for alt_id in range(len(jobs[job_id][task_id])):
                if solver.Value(presences[(job_id, task_id, alt_id)]):
                    duration = jobs[job_id][task_id][alt_id][0]
                    machine = jobs[job_id][task_id][alt_id][1]
                    selected = alt_id
            print(
                '  task_%i_%i starts at %i (alt %i, machine %i, duration %i)' %
                (job_id, task_id, start_value, selected, machine, duration))

    print('Solve status: %s' % solver.StatusName(status))
    print('Optimal objective value: %f' % (solver.ObjectiveValue()/worst))
    print('Statistics')
    print('  - conflicts : %i' % solver.NumConflicts())
    print('  - branches  : %i' % solver.NumBranches())
    print('  - wall time : %f s' % solver.WallTime())
    return solver.WallTime(), solver.ObjectiveValue(), worst

data_dir = 'Data/JSP/'
job_smaples_infos = [[5,20,5], [10,50,10], [20,100,20], [50,200,50], [100,500,100]]
ind = 3
n_jobs = job_smaples_infos[ind][0]
n_tasks = job_smaples_infos[ind][1]
n_machines = job_smaples_infos[ind][2]
file_name = data_dir +'J'+ str(n_jobs)+'_T_'+str(n_tasks)+'_M_'+str(n_machines)+'.pickle'
with open(file_name, 'rb') as handle:
    datas = pickle.load(handle)
results = []
for data in datas:

    n_tasks = data["n_tasks"]
    n_machines = data["n_machines"]
    n_jobs = data["n_jobs"]
    jobs = []
    for i in range(n_jobs):
        tasks_in_job =  []
        for j in range(n_tasks):
            if data["task_job_mapping"][0,j] == i+1:
                # tasks_in_job.append(j)
                task_machine_info = []
                for k in range(n_machines):
                    if data["task_machine_accessibility"][k, j+1] == 1:
                        task_machine_info.append((data["task_machine_time"][k, j+1].item(), k))
                tasks_in_job.append(task_machine_info)

        jobs.append(tasks_in_job)

    walltime, optimal, worst = flexible_jobshop(jobs,n_machines)
    results.append([walltime, optimal, worst])
    ft = 0
datadir = "ORTOOLS/"
os.makedirs(datadir, exist_ok=True)
file_name = datadir+"Results_J_"+str(n_jobs)+"_T_"+str(n_tasks)+"_M"+str(n_machines)+".csv"

with open(file_name, 'w') as csvfile:
    wrtr = csv.writer(csvfile)
    wrtr.writerows(results)
# with open(file_name,'wb') as handle:
#     pickle.dump
