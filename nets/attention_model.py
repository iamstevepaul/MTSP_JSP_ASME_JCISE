import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches
import time

from nets.graph_encoder import GraphAttentionEncoder, CCN, CCN3, GCAPCN, GCAPCN_K_1_P_2_L_3, GCAPCN_K_1_P_2_L_2, GCAPCN_K_1_P_2_L_1, GCAPCN_K_2_P_3_L_1, GCAPCN_K_2_P_2_L_1, GCAPCN_K_2_P_1_L_1, GCAPCN_K_3_P_1_L_1, GCAPCN_K_1_P_1_L_1, GCAPCN_K_2_P_3_L_2
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_vrp = problem.NAME == 'cvrp'
        self.is_mrta = problem.NAME == 'mrta'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        torch.autograd.set_detect_anomaly(True)
        self.robots_state_query_embed = nn.Linear(2, embedding_dim)
        self.robot_taking_decision_query = nn.Linear(2, embedding_dim)

        # Problem specific context parameters (placeholder and step context dimension)
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect


        step_context_dim = 30#embedding_dim + 1


        node_dim = 2# x, y, demand / prize

        if self.is_mrta:
            # n_robot = 20
            # step_context_dim = embedding_dim + 1 + 1 + 1 + n_robot*(embedding_dim + 1 + 1)#embedding_dim + 2



            step_context_dim_new = embedding_dim#embedding_dim + embedding_dim

        # Special embedding projection for depot node
        self.init_embed_depot = nn.Linear(2, embedding_dim)

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        n_machines = 20
        n_tasks = 100


        self.embedder = GCAPCN_K_2_P_2_L_1(
                    n_dim=embedding_dim,
                    node_dim=n_machines+1
                )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings_task = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context_task = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_node_embeddings_machine = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context_machine = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.project_step_context = nn.Linear(step_context_dim_new, embedding_dim, bias=False)
        self.project_context_cur_loc = nn.Linear(2, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out_machine = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out_task = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.state_embedding = nn.Linear(2*(n_tasks+n_machines)+1, embedding_dim)
        self.machine_encoding = nn.Linear(2*(n_tasks+1), embedding_dim)
        self.machine_context_encoding = nn.Linear(2*embedding_dim, embedding_dim)

        wait_encoding = [nn.Linear(2*embedding_dim, 2*embedding_dim),nn.Linear(2*embedding_dim, 2*embedding_dim),nn.Linear(2*embedding_dim, embedding_dim)]
        self.machine_wait_encoding = nn.Sequential(*wait_encoding)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            import time
            # start_time = time.time()
            # embeddings, _ = self.embedder(self._init_embed(input))
            embeddings, _ = self.embedder(input)
            # end_time = time.time() - start_time

        _log_p_machine, _log_p_task, pi, cost = self._inner(input, embeddings)
        # cos, mask = self.problem.get_costs(input, pi)
        mask = None
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p_machine, _log_p_task, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p_machine, _log_p_task, a, mask):

        # Get log_p corresponding to selected actions
        _log_p_machine = _log_p_machine.gather(2, a[:,:,0].unsqueeze(-1)).squeeze(-1)
        _log_p_task = _log_p_task.gather(2, a[:,:, 1].unsqueeze(-1)).squeeze(-1)
        log_p = _log_p_machine + _log_p_task
        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        features = ('deadline','workload')
        # print(self.init_embed(torch.cat((
        #             input['loc'],
        #             *(input[feat][:, :, None] for feat in features)
        #         ), -1)))

        return torch.cat(
            (
                self.init_embed_depot(input['depot'])[:, :],
                self.init_embed(torch.cat((
                    input['loc'],
                    *(input[feat][:, :, None] for feat in features)
                ), -1))
            ),
            1
        )
        # TSP

    def get_action(self, state):
        # state.machine_status[0,1,0]=1
        # (state.machine_status == 0).to(torch.int64).argmax(dim=0)
        if state.i <4:
            machines_selected = (
                        (state.machine_status == 0).to(torch.int64) * torch.rand(state.machine_status.shape)).argmax(
                dim=1)
        else:

            machines_selected = ((state.machine_status != 2).to(torch.int64)*torch.rand(state.machine_status.shape)).argmax(dim=1)
        selected_machine_operation_accessibility = state.task_machine_accessibility[state.ids.squeeze(), machines_selected.squeeze(),:]
        operations_Avail = (selected_machine_operation_accessibility*state.operations_availability)
        operations_selected = (operations_Avail*torch.rand(state.operations_availability.shape)).argmax(dim=1).unsqueeze(dim=1)

        return torch.cat((machines_selected, operations_selected), dim=1)

    def get_current_state_encoding(self, current_state):
        vec = torch.cat((current_state.machines_current_operation.to(torch.float32),current_state.operations_status, current_state.operations_availability,
                         current_state.machines_operation_finish_time_pred),1)
        return self.state_embedding(vec)

    def get_machine_encoding(self, current_state):
        return self.machine_encoding(torch.cat((current_state.task_machine_accessibility.to(torch.float32) ,current_state.task_machine_time.to(torch.float32)), 2))



    def _inner(self, input, embeddings):

        outputs_machine = []
        outputs_task = []
        sequences = []

        state = self.problem.make_state(input)
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step


        batch_size = state.ids.size(0)

        machine_encodings = self.get_machine_encoding(state)
        fixed_machine = self._precompute(machine_encodings,entity="machine")


        # Perform decoding steps
        i = 0
        # print(state.visited_)
        # print(state.all_finished().item())
        # print(state.visited_.all().item())

        #initial tasks
        while not (self.shrink_size is None and not (state.all_finished().item() == 0) or state.i == 10000):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                # unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                # if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                #     # Filter states
                #     state = state[unfinished]
                #     fixed = fixed[unfinished]

            # Only the required ones goes here, so we should
            #  We need a variable that track which all tasks are available
            current_state_encoding = self.get_current_state_encoding(state).unsqueeze(dim=1)

            entity = 'machine'
            log_p_machine, machine_mask = self._get_log_p(fixed_machine, state, query=current_state_encoding, entity=entity)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            machine_selected = self._select_node(log_p_machine.exp()[:, 0, :], machine_mask[:, 0, :])  # Squeeze out steps dimension
            selected_machine_encodings = machine_encodings[state.ids.squeeze(), machine_selected].unsqueeze(dim=1)

            task_selection_context = self.machine_context_encoding(torch.cat((current_state_encoding, selected_machine_encodings), 2))
            waitng_embedding = self.machine_wait_encoding(torch.cat((current_state_encoding, selected_machine_encodings), 2))
            full_task_encoding = torch.cat((waitng_embedding, embeddings),1)
            fixed_task = self._precompute(full_task_encoding, entity="task")

            entity = 'task'
            log_p_task, task_mask = self._get_log_p(fixed_task, state, query=task_selection_context,
                                                          entity=entity, machine_selected=machine_selected)

            task_selected = self._select_node(log_p_task.exp()[:, 0, :], task_mask[:, 0, :])




            actions = torch.cat((machine_selected[:,None], task_selected[:,None]),1)

            state = state.update(actions)


            # # Now make log_p, selected desired output size by 'unshrinking'
            # if self.shrink_size is not None and state.ids.size(0) < batch_size:
            #     log_p_, selected_ = log_p, selected
            #     log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
            #     selected = selected_.new_zeros(batch_size)

                # log_p[state.ids[:, 0]] = log_p_
                # selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs_machine.append(log_p_machine[:, 0, :])
            outputs_task.append(log_p_task[:, 0, :])
            sequences.append(actions)
            # print(state.all_finished().item() == 0)
            i += 1
        # print(state.tasks_done_success, cost)
        # Collected lists, return Tensor

        # nor = (state.task_machine_time.permute(0, 2, 1)[:, 1:, :] == 0)*1000 + (state.task_machine_time.permute(0, 2, 1))[:, 1:, :]
        worst = state.task_machine_time.permute(0, 2, 1)[:, 1:, :].max(dim=2).values.sum(dim=1).unsqueeze(dim=1)
        cost = torch.div(state.current_time, worst) + ((state.operations_status != 2).to(torch.float32).sum(dim=1) * 1000).unsqueeze(dim=1) # makespan #((mk < mk.max()).to(torch.float32)*mk).max(1)[0][:, None]/state.n_agents


        return torch.stack(outputs_machine, 1),  torch.stack(outputs_task, 1), torch.stack(sequences, 1), cost

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner_eval(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            # (input, self.embedder(input)[0]),  # Pack input with embeddings (additional input) - for CCN encoding
            (input, self.embedder(self._init_embed(input))[0]), ## for MHA encoding
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1, entity="task"):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        if entity=="task":
            fixed_context = self.project_fixed_context_task(graph_embed)[:, None, :]
        else:
            fixed_context = self.project_fixed_context_task(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        if entity == "machine":
            glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
                self.project_node_embeddings_machine(embeddings[:, None, :, :]).chunk(3, dim=-1)
        else:
            glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
                self.project_node_embeddings_machine(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, query:torch.Tensor, normalize=True, entity = 'machine', machine_selected=None):

        # Compute query = context node embedding
        # query = fixed.context_node_projected + \
        #         self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state)) ### this has to be cross checked for the context inputs

        # query = self.project_step_context(query)

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        # mask = state.get_mask()
        if entity == "machine":
            if state.i < state.n_machines[0]:
                mask = (state.machine_status != 0).permute(0, 2, 1)
            else:
                ids_with_all_wait = (state.machine_status.squeeze().prod(dim=1) == 1).nonzero().squeeze(dim=1)
                mask = (state.machine_status == 2).permute(0,2,1)
                if ids_with_all_wait.size()[0] > 0:
                    #masking machines with nothing available to them
                    m1 = (state.task_machine_accessibility[ids_with_all_wait,:,1:]*state.operations_availability[ids_with_all_wait,None,1:].expand(state.task_machine_accessibility[ids_with_all_wait,:,1:].shape)).sum(dim=2).unsqueeze(dim=1) == 0
                    # m2 = (state.machine_status[ids_with_all_wait,:,:] == 2).permute(0,2,1)
                    mask[ids_with_all_wait,:,:] = m1# torch.bitwise_or(m1,m2)
                ids_with_not_all_wait = (state.machine_status.squeeze().prod(dim=1) != 1).nonzero().squeeze(dim=1)
                if ids_with_not_all_wait.size()[0] > 0: # ids with not all wait, mask machine with status 2
                    mask[ids_with_not_all_wait,:,:] = (state.machine_status[ids_with_not_all_wait, :, :] == 2).permute(0, 2, 1)
        else:
            selected_machine_operation_accessibility = state.task_machine_accessibility[state.ids.squeeze(),
                                                       machine_selected.squeeze(), :]
            m1 = (selected_machine_operation_accessibility * state.operations_availability).unsqueeze(dim=1) == 0
            mask = (selected_machine_operation_accessibility * state.operations_availability).unsqueeze(dim=1) == 0

            ids_not_all_unavailable = ((m1.squeeze(dim=1).to(torch.float32)[:, 1:]).sum(dim=1) < state.n_tasks).nonzero().squeeze(dim=1)
            if ids_not_all_unavailable.size()[0] > 0: # if atelast one non wait task is there, then set mask wait
                mask[ids_not_all_unavailable,0,0] = True

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, entity)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        if torch.isnan(log_p).any():
            dt = 0

        assert not torch.isnan(log_p).any()

        # need to do masking  here

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

            # Embedding of previous node + remaining capacity
        accessible_operations = state.task_machine_accessibility
        batch_size, n_machines, n_op = accessible_operations.size()
        operations_status = state.operations_status
        operations_availability = state.operations_availability





        return torch.cat((accessible_operations.to(torch.float32), operations_status.to(torch.float32)[:,None,:].expand(batch_size, n_machines, n_op), operations_availability[:,None,:].expand(batch_size, n_machines, n_op)), -1)





    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, entity = 'machine'):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        # if self.mask_inner:
        #     assert self.mask_logits, "Cannot mask inner without masking logits"
        #     compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        if entity == 'machine':
            glimpse = self.project_out_machine(
                heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))
        else:
            glimpse = self.project_out_task(
                heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        # if self.tanh_clipping > 0:
        #     logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
