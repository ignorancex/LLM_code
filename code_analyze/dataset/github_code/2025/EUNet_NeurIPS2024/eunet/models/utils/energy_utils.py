import torch
import dgl
import dgl.function as fn


def euler_inertia(cur_state, prev_state, mass, dt):
    next_x1 = cur_state[:, 0:3]
    next_x0 = prev_state[:, 0:3]
    next_v = (next_x1 - next_x0) / dt
    energy = 0.5 * mass * torch.sum(next_v*next_v, dim=-1, keepdim=True)
    return energy

def euler_gravity(cur_state, mass, gravity):
    cur_x = cur_state[:, 0:3]
    g = gravity[:, -3:]

    energy = -1 * mass * torch.sum(g*cur_x, dim=-1, keepdim=True)
    return energy

def euler_energy(
    cur_state, prev_state, prevprev_state,
    cur_potential, prev_potential,
    gravity, mass,
    cur_dis_potential=None, prev_dis_potential=None, dt=1/30, f_ext=None, **kwargs):
    delta_pred = cur_potential - prev_potential
    if prev_dis_potential is not None:
        delta_pred += prev_dis_potential
    
    cur_inertia = euler_inertia(cur_state, prev_state, mass, dt)
    prev_inertia = euler_inertia(prev_state, prevprev_state, mass, dt)
    delta_inertia = cur_inertia - prev_inertia

    cur_gra_e = euler_gravity(cur_state, mass, gravity)
    prev_gra_e = euler_gravity(prev_state, mass, gravity)
    delta_gra_e = cur_gra_e - prev_gra_e

    delta_label = -(delta_inertia + delta_gra_e)
    return delta_label.sum(dim=0, keepdim=True), delta_pred.sum(dim=0, keepdim=True)


def propagate_with_weight(neighbor_node_field, edge_field, out_field):
    def func(edges):
        src_mask = edges.src[neighbor_node_field]
        assert src_mask.max() == 1.0 and src_mask.min() == 0.0
        src_weight = 1/(src_mask + 1) # To [1, 2]
        edge_energy = edges.data[edge_field]
        weighted_energy = src_weight * edge_energy
        return {out_field: weighted_energy}
    return func

def collect_unique_energy(
        hop_mask, first_neighor_mask, edges, edge_data, noised_edge_data,
        node_filed='mask', neighbor_node_field='neighbor_mask', edge_field='energy', noised_edge_field='noised_energy'):
    def func(e_filed, n_n_field, n_field):
        # Apply the weight for edge energy
        weighted_e_field = f'weighted_{e_filed}'
        g.apply_edges(propagate_with_weight(n_n_field, e_filed, weighted_e_field))
        ## Collect these energy
        g.send_and_recv(g.edges(), fn.copy_e(weighted_e_field, weighted_e_field), fn.sum(weighted_e_field, weighted_e_field))
        ## Clean unused ones
        g.ndata[weighted_e_field] = g.ndata[weighted_e_field] * g.ndata[n_n_field]
        ## Forward to the noised nodes
        n_collect_field = f'collected_{e_filed}'
        g.send_and_recv(g.edges(), fn.copy_u(weighted_e_field, n_collect_field), fn.sum(n_collect_field, n_collect_field))
        node_wise_energy = g.ndata[n_collect_field] * g.ndata[n_field]
        return node_wise_energy

    # the api recieve: src -> dst
    src = torch.cat([edges[:, 0:1], edges[:, 1:2]], dim=0).squeeze(-1)
    dst = torch.cat([edges[:, 1:2], edges[:, 0:1]], dim=0).squeeze(-1)
    g_edata = torch.cat([edge_data, edge_data], dim=0)
    noised_g_edata = torch.cat([noised_edge_data, noised_edge_data], dim=0)
    g = dgl.graph((src, dst))
    g.edata[edge_field] = g_edata
    g.edata[noised_edge_field] = noised_g_edata

    g.ndata[node_filed] = hop_mask
    g.ndata[neighbor_node_field] = first_neighor_mask

    node_wise_energy = func(edge_field, neighbor_node_field, node_filed)
    noised_node_wise_energy = func(noised_edge_field, neighbor_node_field, node_filed)
    return node_wise_energy, noised_node_wise_energy
    
def comp_energy(
        cur_pred,
        cur_state, prev_state,
        cur_noised_state, cur_pred_noise,
        cur_gravity,
        mass, hop_mask, f_connectivity_edges, first_neighbor_mask, dt=1/30):
    orig_energy, noised_energy = collect_unique_energy(
        hop_mask, first_neighbor_mask, f_connectivity_edges, cur_pred, cur_pred_noise)

    cur_inertia = euler_inertia(cur_state, prev_state, mass, dt=dt)
    noised_inertia = euler_inertia(cur_noised_state, prev_state, mass, dt=dt)

    cur_gra_e = euler_gravity(cur_state, mass, cur_gravity)
    noised_gra_e = euler_gravity(cur_noised_state, mass, cur_gravity)

    cur_total_e = orig_energy+cur_inertia+cur_gra_e
    noised_total_e = noised_energy+noised_inertia+noised_gra_e

    return cur_total_e, noised_total_e