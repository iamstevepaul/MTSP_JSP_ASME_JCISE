from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.jsp.state_jsp import StateJSP
from utils.beam_search import beam_search


class JSP(object):

    NAME = 'mrta'  # Capacitated Vehicle Routing Problem

    # VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size, loc_vec_size = dataset['loc'].size()
        # print(batch_size, graph_size, loc_vec_size)
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"


        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        cost = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

        return cost

    @staticmethod
    def make_dataset(*args, **kwargs):
        return JSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateJSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = JSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(data):
    loc = data['loc_data']['loc']
    workload = data['loc_data']['workload']
    deadline = data['loc_data']['deadline']
    initial_size = 100
    n_agents = len(data['robot_data']['robots_capacity'])
    max_capacity = 10
    max_range = 4
    max_speed = .07
    enable_capacity_constraint =  False
    enable_range_constraint = True
    grid_size = 1

    return [{
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'deadline': torch.tensor(deadline, dtype=torch.float),
        'depot': torch.zeros((1, 2)),
        'workload': torch.tensor(workload, dtype=torch.float),
        'initial_size':initial_size,
        'n_agents':n_agents,
        'max_n_agents': torch.tensor([[n_agents]]),
        'max_range':max_range,
        'max_capacity':max_capacity,
        'max_speed':max_speed,
        'enable_capacity_constraint':enable_capacity_constraint,
        'enable_range_constraint':enable_range_constraint,
        'robots_start_location': torch.tensor(data['robot_data']['robots_loc'], dtype=torch.float),
        'robots_work_capacity': torch.tensor(data['robot_data']['robots_capacity'], dtype=torch.float)

    }]


class JSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0,
                 n_depot = 1,
                 initial_size = None,
                 deadline_min = None,
                 deadline_max=None,
                 n_agents = 20,
                 max_range = 4,
                 max_capacity = 10,
                 max_speed = .01,
                 enable_capacity_constraint = False,
                 enable_range_constraint=True,
                 distribution=None):
        super(JSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = make_instance(data)#[make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            n_samples = num_samples
            n_tasks = 100
            n_machines = 30
            n_jobs = 10
            time_low = 10
            time_high = 100


            job_list = []

            for i in range(n_samples):
                task_machine_accessibility = torch.randint(0, 2, (n_machines, n_tasks))
                task_machine_time = torch.randint(time_low, time_high + 1,
                                                  (n_machines, n_tasks)) * task_machine_accessibility
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

            self.data = job_list


        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
