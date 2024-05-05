import pandas as pd
import networkx as nx
import numpy as np
import random

chic_choc = pd.read_csv('chic_choc_data.csv')
costs = pd.read_csv('costs.csv')
too_exp = [107, 1684, 1912, 3437]

G = nx.from_pandas_edgelist(chic_choc, 'user', 'friend')
G0 = G.copy


def list_intersection(a, b):
    a = set(a)
    return a.intersection(b)


def search_in_array(array, a):
    for tuple in array:
        if tuple[0] == a:
            return tuple[1]
    return 0


def graphHistory(G):
    graph_history = []
    G1 = G.copy(as_view=False)
    graph_history.append(G)
    for time in range(1, 7):
        Gt_1 = graph_history[time - 1]
        Gt = Gt_1.copy(as_view=False)
        for node1 in Gt_1:
            for node2 in Gt_1:
                if node2 == node1: continue
                if Gt_1.has_edge(node1, node2): continue
                mutual_neighbors = len(list_intersection(list(Gt_1.neighbors(node1)), list(Gt_1.neighbors(node2))))
                p = 1 - 0.99 ** np.log(mutual_neighbors) if mutual_neighbors > 0 else 0
                p1 = random.uniform(0, 1)
                if p >= p1:
                    Gt.add_edge(node1, node2)
                else:
                    continue
        graph_history.append(Gt)
    return graph_history


def chic_choc_full_edges_creation():
    chic_choc_data_full_edges = pd.read_csv('chic_choc_data.csv')
    chic_choc_data_full_edges = pd.concat(
        [chic_choc_data_full_edges, chic_choc_data_full_edges.rename(columns={'user': 'friend', 'friend': 'user'})])
    return chic_choc_data_full_edges


def data_per_vertex_creation():
    chic_choc_data_full_edges = chic_choc_full_edges_creation()
    num_of_friends = chic_choc_data_full_edges.groupby('user').size().reset_index(name='num_of_friends')
    cost_df = pd.read_csv('costs.csv').astype(int)

    data_per_vertex = pd.merge(num_of_friends, cost_df, on='user', how='left').fillna(0)
    data_per_vertex.columns = ['user', 'num_of_friends', 'cost']
    data_per_vertex['cost'] = data_per_vertex['cost'].astype(int)

    return data_per_vertex


def top_k_center_influence(original_data, data_influ, data_small, top_k):
    center_counts_for_influ = {}
    for user in data_small.itertuples(index=False):
        user_id = user.user
        for friend in original_data[original_data.user == user_id]['friend'].astype(int):
            if int(friend) in data_influ['user'].astype(int).tolist():
                if int(friend) in center_counts_for_influ:
                    center_counts_for_influ[int(friend)] += 1
                else:
                    center_counts_for_influ[int(friend)] = 0

    sorted_nodes = sorted(center_counts_for_influ.keys(), key=lambda x: center_counts_for_influ[x], reverse=True)

    res = list(set(sorted_nodes) - set(too_exp))

    return res[:top_k]


def expected_normalized_influence(edges, list_of_potential_influ, top_k):
    influence_probs = {}
    for user in list_of_potential_influ.itertuples(index=False):
        influence_probs[user.user] = 0

    friend_counts = {}
    for user in list_of_potential_influ.itertuples(index=False):
        user_id = user.user
        friend_counts[user_id] = 0
        for friend in edges[edges.user == user_id]['friend'].astype(int):
            if int(friend) in list_of_potential_influ['user'].astype(int).tolist():
                friend_counts[user_id] += 1

    for user in list_of_potential_influ.itertuples(index=False):
        influence_probs[user.user] = 1 - (friend_counts[user.user] / user.num_of_friends)

    sorted_nodes = sorted(influence_probs.keys(), key=lambda x: influence_probs[x], reverse=True)

    res = list(set(sorted_nodes) - set(too_exp))

    return res[:top_k]


def count_of_loyal_followers(graph, data_with_friends, top_k):
    chic_choc = graph.to_numpy()
    per_vertex = data_with_friends.to_numpy()
    users = per_vertex[:, 0]

    col1 = chic_choc[:, 0]
    col2 = chic_choc[:, 1]

    strong_influencers = {}
    for user in users:
        key = float("nan")
        for j in range(len(col1)):
            if col1[j] == user:
                key = col2[j]
            elif col2[j] == user:
                key = col1[j]
            else:
                key = float("nan")
            if not np.isnan(key):
                if key not in strong_influencers:
                    strong_influencers[key] = 0
                else:
                    strong_influencers[key] += 1

    sorted_nodes = sorted(strong_influencers, key=strong_influencers.get, reverse=True)
    res = list(set(sorted_nodes)-set(too_exp))
    return res[:top_k]


def criterion_time_zero(original_data, data_per_vertex):
    data = data_per_vertex.drop('cost', axis=1)
    condition = data['num_of_friends'] > 120
    data_above_condition_friends = data[condition]
    condition_few = data['num_of_friends'] < 8
    data_few_friends = data[condition_few]

    graph = original_data.sort_values(['user', 'friend'])
    data_per = data_per_vertex.sort_values(['num_of_friends', 'cost'], ascending=False)

    list_of_high_ad = expected_normalized_influence(original_data, data_above_condition_friends, 10)
    list_of_high_center = top_k_center_influence(original_data, data_above_condition_friends, data_few_friends, 10)
    list_of_high_loyal = count_of_loyal_followers(graph, data_few_friends, 10)

    return (list_of_high_ad, list_of_high_center, list_of_high_loyal)


def criteria():
    chic_choc_full_edges = chic_choc_full_edges_creation()
    data_per_vertex_full = data_per_vertex_creation()
    list_of_high_ad, list_of_high_center, list_of_high_loyal = criterion_time_zero(chic_choc_full_edges,
                                                                                   data_per_vertex_full)
    res = []
    res.extend(list_of_high_loyal)
    res.extend(list_of_high_center)
    res.extend(list_of_high_ad)
    return list(set(res))


def potential_nodes(G):
    a = nx.eigenvector_centrality(G)
    fin_a = sorted(a, key=a.get, reverse=True)[0:10]
    b = nx.closeness_centrality(G)
    fin_b = sorted(b, key=b.get, reverse=True)[0:10]
    c = criteria()  # [686, 0, 3101, 348, 2839, 3363, 2754, 414, 376, 483, 3980, 698]
    fin = list(set(fin_a + fin_b) | set(c))
    fin = list(set(fin) - {107, 1684, 1912, 3437})
    return fin


def get_influencers_cost(influencers: list) -> int:
    return sum(
        [costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in
         influencers])


def infected(influencers, graph_history):
    infected = []
    temp = []
    infected.extend(influencers)
    for time in range(1, 7):
        GT_nodes = graph_history[time].nodes()
        for node in list(GT_nodes):
            user_neighbors = list(graph_history[time].neighbors(node))
            n_t = len(user_neighbors)
            b_t = len(list_intersection(user_neighbors, infected))
            p1 = random.uniform(0, 1)
            p = b_t / n_t
            if p >= p1:
                temp.append(node)
        infected.extend(temp)
        temp.clear()
    cost = get_influencers_cost(influencers)
    return len(list(set(infected))) - cost


def hill_climbing(graph_history, possible_influencers):
    added = []
    possible = []
    possible = possible_influencers.copy()
    infectedd = 0
    a = [] #
    max_i = []
    for i in range(1, 6):
        for possible_influencer in possible:
            a.append(possible_influencer)
            infected_f = infected(a, graph_history)
            if infected_f > infectedd:
                infectedd = infected_f
                max_i = possible_influencer
            a.remove(possible_influencer)
        if max_i not in added:
            added.append(max_i)
            possible.remove(max_i)
        a.extend(added)
        a = list(set(a))
    while len(added) < 5:
        added.append(possible[0])
        possible.pop(0)
    print(added)
    return added


p = potential_nodes(G)
print(p)
graph_history = graphHistory(G)
hill_climbing(graph_history, p)
