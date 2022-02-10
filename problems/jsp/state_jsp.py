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
    operations_availability: torch.Tensor
    operations_next: torch.Tensor
    current_time : torch.Tensor
    machine_taking_decision : torch.Tensor # ID of the machine taking decision
    machines_current_operation: torch.Tensor

    machine_taking_decision_operationId: torch.Tensor # ID of the operation where the machine taking a decision is currently is
    machine_taking_decision_jobId: torch.Tensor #ID of the job where the machine taking decision is currently is
    machines_initial_decision_sequence: torch.Tensor
    machine_status: torch.Tensor

    operations_machines_assignment: torch.Tensor

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
        device = input["n_tasks"].device
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

        operations_status = torch.zeros((n_samples, n_tasks[0].item()), device=device)
        current_time = torch.zeros((n_samples,1), dtype=torch.float32, device=device)
        machine_taking_decision = torch.zeros((n_samples,1), device=device).to(torch.int64)
        machines_current_operation = torch.zeros((n_samples,n_machines[0].item()), device=device).to(torch.int64)
        machine_taking_decision_operationId = torch.zeros((n_samples,1), device=device).to(torch.int64)
        machine_taking_decision_jobId = torch.zeros((n_samples,1), device=device).to(torch.int64)
        machines_initial_decision_sequence = torch.arange(0, n_machines[0].item(), device=device)
        operations_machines_assignment = torch.zeros((n_samples, n_machines[0].item(), n_tasks[0].item()), device=device)
        machine_status = input["machine_status"]

        machine_idle = torch.ones((n_samples,n_machines[0].item()), device=device).to(torch.int64)





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
            operations_next = input["operations_next"],
            operations_availability = input["operations_availability"],
            current_time = current_time,
            machine_taking_decision=machine_taking_decision, ### can be removed
            machines_current_operation = machines_current_operation,
            machine_taking_decision_operationId = machine_taking_decision_operationId, ### can be removed
            machine_taking_decision_jobId = machine_taking_decision_jobId, ### can be removed
            machines_initial_decision_sequence = machines_initial_decision_sequence, ### can be removed
            machines_operation_finish_time_pred = torch.zeros((n_samples, n_machines[0].item()), device=device),
            operations_machines_assignment = operations_machines_assignment,
            machine_status=machine_status,
            machine_idle=machine_idle, ### can be removed
            ids=torch.arange(n_samples, dtype=torch.int64, device=device)[:, None],
            i=torch.zeros(1, dtype=torch.int64, device=device),
        )

    def get_final_cost(self):


        # torch.mul(len, self.)
        return []

    def update(self, selected):
        # print('************** New decision **************') #check after 3rd iteration
        # print(selected[1,:])
        ids_wait = ((selected[:,1] == 0).nonzero()).squeeze(dim=1)
        ids_non_wait = ((selected[:,1] != 0).nonzero()).squeeze(dim=1)
        machine_selected = selected[:, 0]
        task_selected = selected[:, 1]
        # machine selected and action
        current_time = self.current_time
        operations_status = self.operations_status
        machine_status = self.machine_status
        machines_operation_finish_time_pred = self.machines_operation_finish_time_pred
        machines_current_operation = self.machines_current_operation
        operations_availability = self.operations_availability
        machines_current_operation[self.ids.squeeze(), machine_selected] = task_selected
        if ids_wait.size()[0] > 0:
            machine_status[ids_wait, machine_selected[ids_wait]] = 1 #status setting as waiting

        if ids_non_wait.size()[0] > 0:
            machine_status[ids_non_wait, machine_selected[ids_non_wait]] = 2 # status changed as occupied

            time_for_operation = self.task_machine_time[ids_non_wait, machine_selected[ids_non_wait], task_selected[ids_non_wait]]
            finish_time_prediction = current_time[ids_non_wait] + time_for_operation.unsqueeze(dim=1)

            machines_operation_finish_time_pred[ids_non_wait, machine_selected[ids_non_wait]] = finish_time_prediction.squeeze(dim=1)
            operations_availability[ids_non_wait,task_selected[ids_non_wait]] = 0 # making the task no longer available for other machines
            operations_status[ids_non_wait, task_selected[ids_non_wait]-1] = 1 #1 for engaged

        # ids_with_idle = (machine_status == 0).nonzero()

        ids_with_no_idle = (machine_status.squeeze().prod(dim=1) != 0).nonzero().squeeze(dim=1) # prod is used to detect the presence of 0 in dimension 1
        if ids_with_no_idle.size()[0] > 0:
            #find ids with 2 as min update 1s - need to make sure atleast one machine is working while there is availabiltty

            ids_with_atleast_one_occupied = ((machine_status[ids_with_no_idle,:] == 1).to(torch.float32).squeeze(dim=2).prod(dim=1) == 0).nonzero().squeeze(dim=1)
            ids_with_atleast_one_occupied = ids_with_no_idle[ids_with_atleast_one_occupied]
            if ids_with_atleast_one_occupied.size()[0] > 0:
            # finding the min time for operating mchines
                next_time_data = ((machine_status[ids_with_atleast_one_occupied,:] == 1).to(torch.float32).squeeze(dim=2) * 1000000 + machines_operation_finish_time_pred[ids_with_atleast_one_occupied,:]).min(dim=1)
                current_time[ids_with_atleast_one_occupied] = next_time_data.values.unsqueeze(dim=1)
                min_task_index = next_time_data.indices
                machine_status[ids_with_atleast_one_occupied, min_task_index] = 1
                finished_operations = machines_current_operation[ids_with_atleast_one_occupied, min_task_index]
                operations_status[ids_with_atleast_one_occupied, finished_operations-1] = 2 # 2 for finished
                operations_availability[ids_with_atleast_one_occupied, finished_operations] = 0

                ids_next_not_idle = (self.operations_next[ids_with_atleast_one_occupied, finished_operations-1, 0].to(torch.int64) !=0).nonzero().squeeze(dim=1)
                # ids_next_not_idle = ids_with_atleast_one_occupied[ids_next_not_idle]
                if ids_next_not_idle.size()[0] > 0:
                    operations_availability[ids_with_atleast_one_occupied[ids_next_not_idle], self.operations_next[ids_with_atleast_one_occupied[ids_next_not_idle], finished_operations[ids_next_not_idle]-1, 0].to(torch.int64)] = 1

                machines_operation_finish_time_pred[ids_with_atleast_one_occupied] = ((machine_status[ids_with_atleast_one_occupied] == 2).to(torch.float32).squeeze() * machines_operation_finish_time_pred[
                    ids_with_atleast_one_occupied]) + ((machine_status[ids_with_atleast_one_occupied] == 1).to(torch.float32).squeeze() * next_time_data.values.unsqueeze(dim=1))




        return self._replace(
            current_time =current_time,
            operations_status = operations_status,
            machines_operation_finish_time_pred=machines_operation_finish_time_pred,
            operations_availability=operations_availability,
            machine_status = machine_status,
            machines_current_operation = machines_current_operation,
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
