import torch
from typing import NamedTuple
import numpy as np
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateJSP(NamedTuple):
    # Fixed input
    n_tasks: torch.Tensor
    n_machines : torch.Tensor
    n_jobs: torch.Tensor
    time_low : torch.Tensor
    time_high : torch.Tensor
    task_machine_accessibility : torch.Tensor
    task_machine_time : torch.Tensor
    task_job_mapping : torch.Tensor
    job_nums : torch.Tensor
    n_ops_in_jobs : torch.Tensor
    adjacency : torch.Tensor

    operations_status: torch.Tensor # takes values 0, 1, or 2
    current_time : torch.Tensor
    machine_taking_decision : torch.Tensor # ID of the machine taking decision
    machines_current_operation: torch.Tensor

    machine_taking_decision_operationId: torch.Tensor # ID of the operation where the machine taking a decision is currently is
    machine_taking_decision_jobId: torch.Tensor #ID of the job where the machine taking decision is currently is
    machines_initial_decision_sequence: torch.Tensor

    machine_idle: torch.Tensor

    machines_operation_finish_time_pred: torch.Tensor # time at which the machine finishes its current job

    i: torch.Tensor
    ids: torch.Tensor



    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_[:,:,self.n_depot:]
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        # if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
        #     return self._replace(
        #         ids=self.ids[key],
        #         prev_a=self.prev_a[key],
        #         visited_=self.visited_[key],
        #         lengths=self.lengths[key],
        #         cur_coord=self.cur_coord[key],
        #     )
        return super(StateJSP, self).__getitem__(key)


    @staticmethod
    def initialize(input,
                   visited_dtype = torch.uint8):

        n_tasks = input["n_tasks"]
        n_machines = input["n_machines"]
        n_jobs = input["n_jobs"]
        time_low = input["time_low"]
        time_high = input["time_high"]
        task_machine_accessibility = input["task_machine_accessibility"]
        task_machine_time = input["task_machine_time"]
        task_job_mapping = input["task_job_mapping"]
        job_nums = input["job_nums"]
        n_ops_in_jobs = input["n_ops_in_jobs"]
        adjacency = input["adjacency"]
        n_samples = (input["n_tasks"].size())[0]

        operations_status = torch.zeros((n_samples, n_tasks[0].item()+1)) # 0 for idle
        current_time = torch.zeros((n_samples,1), dtype=torch.float32)
        machine_taking_decision = torch.zeros((n_samples,1)).to(torch.int64)
        machines_current_operation = torch.zeros((n_samples,n_machines[0].item())).to(torch.int64)
        machine_taking_decision_operationId = torch.zeros((n_samples,1)).to(torch.int64)
        machine_taking_decision_jobId = torch.zeros((n_samples,1)).to(torch.int64)
        machines_initial_decision_sequence = torch.arange(0, n_machines[0].item())

        machine_idle = torch.ones((n_samples,n_machines[0].item())).to(torch.int64)





        return StateJSP(
            n_tasks = n_tasks,
            n_machines = n_machines,
            n_jobs = n_jobs,
            time_low = time_low,
            time_high = time_high,
            task_machine_accessibility = task_machine_accessibility,
            task_machine_time = task_machine_time,
            task_job_mapping = task_job_mapping,
            job_nums = job_nums,
            n_ops_in_jobs = n_ops_in_jobs,
            adjacency = adjacency,
            operations_status = operations_status,
            current_time = current_time,
            machine_taking_decision=machine_taking_decision,
            machines_current_operation = machines_current_operation,
            machine_taking_decision_operationId = machine_taking_decision_operationId,
            machine_taking_decision_jobId = machine_taking_decision_jobId,
            machines_initial_decision_sequence = machines_initial_decision_sequence,
            machines_operation_finish_time_pred = torch.zeros((n_samples,1)),
            machine_idle=machine_idle,
            ids=torch.arange(n_samples, dtype=torch.int64)[:, None],
            i=torch.zeros(1, dtype=torch.int64),
        )

    def get_final_cost(self):


        # torch.mul(len, self.)
        return []

    def update(self, selected):
        # print('************** New decision **************')

        ###

        # assign an operation to all machines based on accessibility and priority
        # update the status of thpse operations to be 1
        #update the job id for the machines
        #update the operation id for the machines
        #update the task finish time for the machines
        # find the machine that finish its operation next
        #update the current time
        #update the status of that machine to be 2
        #update machine_taking_decision_operationId to selected
        # continue until all the operations status are 2

        selected = selected[:, None]  # Add dimension for step

        machine_idle = self.machine_idle

        current_time = self.current_time
        operations_status = self.operations_status
        machine_taking_decision = self.machine_taking_decision
        machine_taking_decision_operationId = self.machine_taking_decision_operationId
        operations_status[self.ids, machine_taking_decision_operationId] = 2
        operations_status[self.ids, selected] = 1

        time_to_complete = self.task_machine_time[self.ids, machine_taking_decision, selected]
        completion_time = current_time + time_to_complete

        machines_current_operation = self.machines_current_operation
        machines_current_operation[self.ids, machine_taking_decision] = selected
        machine_taking_decision_operationId = selected

        machines_operation_finish_time_pred = self.machines_operation_finish_time_pred
        machines_operation_finish_time_pred[self.ids, machine_taking_decision] = completion_time

        machine_taking_decision_jobId = self.machine_taking_decision_jobId
        # machine_taking_decision_jobId = self.task_job_mapping[self.ids, :]

        sorted_time, indices = torch.sort(machines_operation_finish_time_pred)
        current_time = sorted_time[self.ids,0]
        machine_taking_decision = indices[self.ids,0]




        return self._replace(
            current_time =current_time,
            machine_taking_decision = machine_taking_decision.to(torch.int64),
            machine_taking_decision_operationId = machine_taking_decision_operationId,
            machines_current_operation =machines_current_operation,
            operations_status = operations_status,
            i=self.i + 1
        )

    def all_finished(self):
        # return self.i.item() >= self.demand.size(-1) and self.visited.all()
        return ((self.operations_status == 2).to(torch.uint8)).all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.machines_current_operation[self.ids, self.machine_taking_decision]

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        # (self.operations_status != 0)[:, None] and self.task_machine_accessibility[self.ids, self.machine_taking_decision] == 0
        # if self.visited_.dtype == torch.uint8:
        #     visited_loc = self.visited_[:, :, 1:]
        # else:
        #     visited_loc = mask_long2bool(self.visited_)
        #
        # mask_loc = visited_loc.to(torch.bool)  # | exceeds_cap
        #
        # # robot_taking_decision = self.robot_taking_decision
        #
        # # Cannot visit the depot if just visited and still unserved nodes
        # mask_depot = torch.tensor(torch.ones((mask_loc.size()[0], 1)).clone().detach(), dtype=torch.bool, device=mask_loc.device) #(self.robots_current_destination[self.ids, robot_taking_decision] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        n_samples, n_operations = self.operations_status.size()
        full_mask = torch.zeros((n_samples, 1, n_operations)).to(torch.bool)
        full_mask[:, :, 1:] = torch.logical_and((self.operations_status[:,1:] != 0)[:, None], self.task_machine_accessibility[self.ids, self.machine_taking_decision] == 0)

        return full_mask

    def construct_solutions(self, actions):
        return actions
