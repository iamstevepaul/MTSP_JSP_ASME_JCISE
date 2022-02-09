"""
Author: Steve Paul 
Date: 2/5/22 """
"""Minimal jobshop example."""
import collections
from ortools.sat.python import cp_model


def main():
    """Minimal jobshop problem."""
    # Data.
    # jobs_data = [  # task = (machine_id, processing_time). accessibility can be added here
    #     [[(0, 3)], [(1, 2)], [(2, 2)]],  # Job0
    #     [[(0, 2)], [(2, 1)], [(1, 4)]],  # Job1
    #     [[(1, 4)], [(2, 3)]]  # Job2
    # ]

    jobs_data = [  # task = (machine_id, processing_time). accessibility can be added here
        [[(0, 3), (1, 2)], [(1, 2), (2,3)], [(2, 2)]],  # Job0
        [[(0, 2)], [(1,2),(2, 1)], [(1, 4)]],  # Job1
        [[(1, 4)], [(2, 3)]]  # Job2
    ]

    # create a new matrix for accessibility
    # an operation can only be done by the machines accessible to the task
    # add constraints that an operation can only be done once - done
    # the operation of any machine should not overlap - done
    # constraints for precedence - done
    # start time and end time greater than 0 - done
    # start time < end time - done
    # start time  = end time - duration - done
    # all jobs should be done or all operations should be done - done

    machines_count = 1 + max(mac[0] for job in jobs_data for task in job for mac in task)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(mac[1] for job in jobs_data for task in job for mac in task)

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)
    co=0
    task_agg_list = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            suffix = '_%i_%i' % (job_id, task_id)
            task_agg = model.NewIntVar(0, 1, 'task_agg'+suffix)

            for mac_id, mac in enumerate(task):

                machine = mac[0]
                duration = mac[1]

                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                    'interval' + suffix)
                all_tasks[job_id, task_id] = task_type(start=start_var,
                                                       end=end_var,
                                                       interval=interval_var)
                mac_bool = model.NewBoolVar(str(machine)+suffix)
                machine_to_intervals[machine].append(interval_var)
                task_agg_list[co].append(mac_bool)
            co = co+1
    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    for tk in range(len(task_agg_list)):
        expre = sum(task_agg_list[tk])
        model.Add(expre == 1)

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Solution:')
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                for mac_id, mac in enumerate(task):
                    machine = mac[0]
                    assigned_jobs[machine].append(
                        assigned_task_type(start=solver.Value(
                            all_tasks[job_id, task_id].start),
                                           job=job_id,
                                           index=task_id,
                                           duration=mac[1]))

        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_task_%i' % (assigned_task.job,
                                           assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-15s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-15s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
        print(output)
    else:
        print('No solution found.')

    # Statistics.
    print('\nStatistics')
    print('  - conflicts: %i' % solver.NumConflicts())
    print('  - branches : %i' % solver.NumBranches())
    print('  - wall time: %f s' % solver.WallTime())


if __name__ == '__main__':
    main()