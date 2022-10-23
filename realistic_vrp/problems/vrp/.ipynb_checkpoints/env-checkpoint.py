import torch
from utils.boolmask import mask_long2bool, mask_long_scatter
import gym


class VRPEnv(gym.Env):

    def __init__(self, instance,visited_dtype=torch.uint8):
        super(VRPEnv, self).__init__()
        self.depot = instance['depot']
        self.coordinates = instance['coordinates']
        self.edge_features = instance['edge_features']
        self.demand = instance['demand']
        batch_size, n_loc, _ = self.coordinates.size()
        self.ids = torch.arange(batch_size, dtype=torch.int64, device=self.coordinates.device)[:, None]
        self.prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=self.coordinates.device)
        self.used_capacity = self.demand.new_zeros(batch_size, 1)
        self.used_capacity = self.demand.new_zeros(batch_size, 1),
        self.visited_ = (  # Visited as mask is easier to understand, as long more memory efficient
            # Keep visited_ with depot so we can scatter efficiently
            torch.zeros(
                batch_size, 1, n_loc + 1,
                dtype=torch.uint8, device=self.coordinates.device
            )
            if visited_dtype == torch.uint8
            else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=self.coordinates.device)  #

        )
        self.lengths = torch.zeros(batch_size, 1, device=self.coordinates.device)
        self.cur_coord = input['depot'][:, None, :]
        self.i = torch.zeros(1, dtype=torch.int64, device=self.coordinates.device)



    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    def initialize(self):
        obs = {
            'coordinates': self.coordinates,
            'edge_features': self.edge_features,
            'demand':self.demand,
            'ids': self.ids,
            'prev_a': self.prev_a,
            'used_capacity': self.used_capacity,
            'visited': self.visited_,
            'lengths': self.lengths,
            'cur_coord': self.cur_coord,
            'i': self.i
            }
        return obs

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coordinates[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)


    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coordinates[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def construct_solutions(self, actions):
        return actions
