from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .partition import MAX_COST, Partition


@dataclass
class VertexChoice:
    node: int
    target_k: int
    score: float
    follower_count: int
    cost: int
    followers: List[int] = field(default_factory=list)


@dataclass
class GroupChoice:
    nodes: List[int]
    score: float
    cost: int
    follower_gain: int
    followers_final: List[Tuple[int, int]] = field(default_factory=list)
    insert_edges: List[Tuple[int, int]] = field(default_factory=list)
    anchor_cost: Dict[int, Tuple[int, int]] = field(default_factory=dict)


class Master:
    def __init__(self, partition: Partition) -> None:
        self.partition = partition
        self.anchor_edges: List[Tuple[int, int]] = []
        self.nfs = 0
        self.acost = 0

    def anchoring(self, budget: int, mode: int, check_path: str) -> None:
        t_begin = time.process_time()
        self.partition.p_decomposition()
        vertex_time = 0.0
        group_time = 0.0
        round_idx = 0
        while self.acost < budget:
            print(f" -- Anchoring round: {round_idx}")
            vertex_choice: Optional[VertexChoice] = None
            group_choice: Optional[GroupChoice] = None

            vertex_start = time.process_time()
            if mode != 1:
                vertex_choice = self.get_anchor_choice(budget)
            vertex_time += time.process_time() - vertex_start

            group_start = time.process_time()
            if mode != 0:
                group_choice = self.get_group_choice(budget)
            group_time += time.process_time() - group_start

            if vertex_choice is None and group_choice is None:
                break

            vertex_score = vertex_choice.score if vertex_choice else -1.0
            group_score = group_choice.score if group_choice else -1.0

            if vertex_score > group_score:
                if vertex_choice is None or vertex_choice.cost + self.acost > budget:
                    break
                self.apply_vertex_choice(vertex_choice)
                print("node")
            else:
                if group_choice is None or group_choice.cost + self.acost > budget:
                    break
                self.apply_group_choice(group_choice)
                print("group")
            round_idx += 1
        total = time.process_time() - t_begin
        print(f"the anchoring time is:{total:.6f}s.")
        print(f"the group time is:{group_time:.6f}s.")
        print(f"the vertex time is:{vertex_time:.6f}s.")
        self.final_insert(check_path)

    def apply_vertex_choice(self, choice: VertexChoice) -> None:
        self.acost += choice.cost
        self.nfs += choice.follower_count
        node = choice.node
        ids = sorted(range(len(self.partition.datagraph.adj_list)), key=lambda x: self.partition.coreness[x], reverse=True)
        remaining = choice.cost
        pos = 0
        while remaining > 0 and pos < len(ids):
            candidate = ids[pos]
            while pos < len(ids):
                candidate = ids[pos]
                if candidate == node:
                    pos += 1
                    continue
                if self.partition.coreness[candidate] >= choice.target_k:
                    if (node, candidate) not in self.partition.datagraph.hash_edges:
                        break
                    pos += 1
                    continue
                if (node, candidate) in self.partition.datagraph.hash_edges:
                    pos += 1
                    continue
                break
            if pos >= len(ids):
                break
            if self._register_edge(candidate, node):
                remaining -= 1
            pos += 1
        self.partition.p_maintenance(node, choice.target_k, choice.followers)

    def apply_group_choice(self, choice: GroupChoice) -> None:
        self.acost += choice.cost
        self.nfs += choice.follower_gain
        for u, v in choice.insert_edges:
            self._register_edge(u, v)
        ids = sorted(range(len(self.partition.datagraph.adj_list)), key=lambda x: self.partition.coreness[x], reverse=True)
        for node, (need_edges, target_k) in choice.anchor_cost.items():
            remaining = need_edges
            pos = 0
            while remaining > 0 and pos < len(ids):
                candidate = ids[pos]
                while pos < len(ids):
                    candidate = ids[pos]
                    if candidate == node or candidate in choice.anchor_cost:
                        pos += 1
                        continue
                    if self.partition.coreness[candidate] >= target_k:
                        if (node, candidate) not in self.partition.datagraph.hash_edges:
                            break
                        pos += 1
                        continue
                    if (node, candidate) in self.partition.datagraph.hash_edges:
                        pos += 1
                        continue
                    break
                if pos >= len(ids):
                    break
                if self._register_edge(candidate, node):
                    remaining -= 1
                pos += 1
        self.partition.group_maintenance(choice.nodes, choice.followers_final)

    def get_anchor_choice(self, budget: int) -> Optional[VertexChoice]:
        best_choice: Optional[VertexChoice] = None
        best_cost = -1
        for vid in range(len(self.partition.datagraph.adj_list)):
            followers_g: List[int] = []
            self.collect_followers_vertex(vid, followers_g)
            followers_g.sort(key=lambda u: self.partition.coreness[u])
            posk = len(followers_g) - 1
            nbr_items = sorted(self.partition.nbrk[vid].items())
            for nbr_core, _ in nbr_items:
                if nbr_core < self.partition.coreness[vid] or nbr_core == self.partition.max_coreness:
                    continue
                target_k = nbr_core + 1
                while posk >= 0 and self.partition.coreness[followers_g[posk]] > target_k - 1:
                    posk -= 1
                poskk = posk
                while poskk >= 0 and self.partition.coreness[followers_g[poskk]] >= target_k - 1:
                    poskk -= 1
                cnt_followers = posk + 1
                cost = target_k
                for idx in range(poskk + 1, posk + 1):
                    follower = followers_g[idx]
                    if (vid, follower) in self.partition.datagraph.hash_edges:
                        cost -= 1
                for nb in self.partition.datagraph.adj_list[vid]:
                    if self.partition.coreness[nb] >= target_k:
                        cost -= 1
                cnt_followers += target_k - self.partition.coreness[vid]
                if cost <= 0:
                    cost = MAX_COST
                if cost + self.acost > budget:
                    continue
                score = cnt_followers / cost
                if best_choice is None or score > best_choice.score or (score == best_choice.score and cost > best_cost):
                    followers_subset = followers_g[: posk + 1]
                    best_choice = VertexChoice(
                        node=vid,
                        target_k=target_k,
                        score=score,
                        follower_count=cnt_followers,
                        cost=cost,
                        followers=list(followers_subset),
                    )
                    best_cost = cost
            posk = len(followers_g) - 1
        return best_choice

    def get_group_choice(self, budget: int) -> Optional[GroupChoice]:
        candidate_group = []
        for vid, core in enumerate(self.partition.coreness):
            if core == self.partition.max_coreness:
                continue
            if self.partition.deggt[vid] < core + 1:
                continue
            candidate_group.append(vid)
        best_choice: Optional[GroupChoice] = None
        best_cost = -1
        for u in candidate_group:
            group: List[int] = [u]
            vis_group: Dict[int, int] = {u: 1}
            k_val = self.partition.coreness[u]
            for v in self.partition.datagraph.adj_list[u]:
                if self.partition.coreness[v] == k_val:
                    group.append(v)
                    vis_group[v] = 1
            r: Dict[int, int] = {}
            budget_map: Dict[int, int] = {}
            for v in group:
                r[v] = k_val + 1
                budget_map[v] = 0
                for w in self.partition.datagraph.adj_list[v]:
                    if self.partition.coreness[w] > k_val or w in vis_group:
                        r[v] -= 1
                        if w in vis_group:
                            budget_map[v] += 1
            for v in list(group):
                if vis_group.get(v, 0) and budget_map[v] < r[v]:
                    vis_group[v] = 0
                    r.pop(v, None)
                    budget_map.pop(v, None)
                    self.group_shrink(v, r, budget_map, vis_group)
            for v in list(r):
                if r[v] < 0:
                    r[v] = 0
            group = [v for v in group if vis_group.get(v, 0)]
            for v in list(vis_group):
                if vis_group[v] == 0:
                    vis_group.pop(v, None)
            new_group: List[int] = []
            visited_group: Dict[int, int] = {}
            r_extra: Dict[int, int] = {}
            budget_extra: Dict[int, int] = {}
            for v in group:
                for w in self.partition.datagraph.adj_list[v]:
                    if self.partition.coreness[w] > self.partition.coreness[v] or w in vis_group:
                        continue
                    if w in visited_group:
                        continue
                    visited_group[w] = 1
                    cost_val = k_val + 1
                    budget_val = 0
                    update_r: List[int] = []
                    for z in self.partition.datagraph.adj_list[w]:
                        if self.partition.coreness[z] > k_val:
                            cost_val -= 1
                        if z in vis_group:
                            if r.get(z, 0) > 0:
                                budget_val += 1
                                update_r.append(z)
                            cost_val -= 1
                    if cost_val < 0:
                        cost_val = 0
                    r_extra[w] = cost_val
                    budget_extra[w] = budget_val
                    if budget_val >= cost_val:
                        new_group.append(w)
                        vis_group[w] = 1
                        for ttt in update_r:
                            if r.get(ttt, 0) > 0:
                                r[ttt] -= 1
                                if r[ttt] == 0:
                                    for ttnbr in self.partition.datagraph.adj_list[ttt]:
                                        if ttnbr not in vis_group and ttnbr in budget_extra:
                                            budget_extra[ttnbr] -= 1
                        r[w] = cost_val
                        budget_map[w] = budget_val
                        self.group_expand(w, r_extra, budget_extra, visited_group, r, budget_map, vis_group, new_group)
            group.extend(new_group)
            r2: Dict[int, int] = {}
            for v in group:
                r2[v] = k_val + 1
                for w in self.partition.datagraph.adj_list[v]:
                    if self.partition.coreness[w] > k_val or w in vis_group:
                        r2[v] -= 1
            for v in r2:
                if r2[v] < 0:
                    r2[v] = 0
            group_cost = 0
            group_anchor_cost_tmp: Dict[int, Tuple[int, int]] = {}
            group_insert_edges_tmp: List[Tuple[int, int]] = []
            followers_gain: List[Tuple[int, int]] = []
            coreness_gain = self.get_group_followers(group, followers_gain, vis_group, k_val + 1)
            for i, w in enumerate(group):
                if r.get(w, 0) == 0:
                    continue
                for j in range(i + 1, len(group)):
                    v = group[j]
                    if r.get(w, 0) == 0:
                        break
                    if r.get(v, 0) == 0:
                        continue
                    if (w, v) not in self.partition.datagraph.hash_edges:
                        group_cost += 1
                        group_insert_edges_tmp.append((w, v))
                        r[w] = r.get(w, 0) - 1
                        r[v] = r.get(v, 0) - 1
                if r.get(w, 0) and r[w] > 0:
                    group_cost += r[w]
                    group_anchor_cost_tmp[w] = (r[w], k_val + 1)
            if group_cost > budget - self.acost:
                continue
            effective_cost = group_cost if group_cost != 0 else MAX_COST
            score = coreness_gain / effective_cost if effective_cost else 0.0
            if best_choice is None or score > best_choice.score or (score == best_choice.score and effective_cost > best_cost):
                best_choice = GroupChoice(
                    nodes=list(group),
                    score=score,
                    cost=effective_cost,
                    follower_gain=coreness_gain,
                    followers_final=list(followers_gain),
                    insert_edges=list(group_insert_edges_tmp),
                    anchor_cost=dict(group_anchor_cost_tmp),
                )
                best_cost = effective_cost
        return best_choice

    def collect_followers_vertex(self, anchor: int, followers_i: List[int]) -> None:
        num_n = self.partition.max_layer + 100.0
        heap: List[Tuple[float, int]] = []
        heapq.heappush(heap, (self.partition.coreness[anchor] + self.partition.shells[anchor] / num_n, anchor))
        degplus: Dict[int, int] = {}
        status: Dict[int, int] = {}
        in_queue: Dict[int, int] = {anchor: 1}
        while heap:
            _, v = heapq.heappop(heap)
            in_queue.pop(v, None)
            degplus_v = 0
            if v == anchor and v not in status:
                for u in self.partition.datagraph.adj_list[v]:
                    if u in status:
                        continue
                    if (
                        self.partition.coreness[v] == self.partition.coreness[u]
                        and self.partition.shells[v] < self.partition.shells[u]
                        and u not in in_queue
                    ):
                        heapq.heappush(
                            heap,
                            (self.partition.coreness[u] + self.partition.shells[u] / num_n, u),
                        )
                        in_queue[u] = 1
                    if self.partition.coreness[u] > self.partition.coreness[v] and u not in in_queue:
                        heapq.heappush(
                            heap,
                            (self.partition.coreness[u] + self.partition.shells[u] / num_n, u),
                        )
                        in_queue[u] = 1
                continue
            for u in self.partition.datagraph.adj_list[v]:
                if self.partition.coreness[u] >= self.partition.coreness[v] + 1 or u == anchor:
                    degplus_v += 1
                elif self.partition.coreness[v] == self.partition.coreness[u] and u != anchor:
                    if u in status:
                        if status[u] == 0:
                            degplus_v += 1
                    else:
                        if self.partition.shells[u] > self.partition.shells[v]:
                            degplus_v += 1
                        elif self.partition.shells[u] <= self.partition.shells[v] and u in in_queue:
                            degplus_v += 1
            if degplus_v >= self.partition.coreness[v] + 1:
                status[v] = 0
                degplus[v] = degplus_v
                for u in self.partition.datagraph.adj_list[v]:
                    if u == anchor:
                        continue
                    if u in status:
                        continue
                    if (
                        self.partition.coreness[v] == self.partition.coreness[u]
                        and self.partition.shells[v] < self.partition.shells[u]
                        and u not in in_queue
                    ):
                        heapq.heappush(
                            heap,
                            (self.partition.coreness[u] + self.partition.shells[u] / num_n, u),
                        )
                        in_queue[u] = 1
            else:
                status[v] = 1
                degplus[v] = degplus_v
                self.shrink_vertex(v, degplus, status)
        for u, flag in status.items():
            if u == anchor:
                continue
            if flag == 0 and degplus.get(u, 0) >= self.partition.coreness[u] + 1:
                followers_i.append(u)

    def shrink_vertex(self, v: int, degplus: Dict[int, int], status: Dict[int, int]) -> None:
        shrink_nodes: List[int] = []
        for u in self.partition.datagraph.adj_list[v]:
            if u in status and status[u] == 0 and self.partition.coreness[u] == self.partition.coreness[v]:
                degplus[u] -= 1
                if degplus[u] < self.partition.coreness[u] + 1:
                    status[u] = 1
                    shrink_nodes.append(u)
        for u in shrink_nodes:
            self.shrink_vertex(u, degplus, status)

    def group_expand(
        self,
        w: int,
        r_extra: Dict[int, int],
        budget_extra: Dict[int, int],
        visited_group: Dict[int, int],
        r: Dict[int, int],
        budget_map: Dict[int, int],
        vis_group: Dict[int, int],
        new_group: List[int],
    ) -> None:
        for u in self.partition.datagraph.adj_list[w]:
            if u in visited_group and r_extra.get(u, 0) > 0:
                r_extra[u] -= 1
        for u in self.partition.datagraph.adj_list[w]:
            if r.get(w, 0) == 0:
                break
            if u in visited_group:
                budget_extra[u] = budget_extra.get(u, 0) + 1
                if budget_extra[u] >= r_extra.get(u, 0) and u not in vis_group:
                    new_group.append(u)
                    vis_group[u] = 1
                    for v in self.partition.datagraph.adj_list[u]:
                        if v in vis_group and r.get(v, 0) > 0:
                            r[v] -= 1
                            if r[v] == 0:
                                for q in self.partition.datagraph.adj_list[v]:
                                    if q not in vis_group and q in budget_extra:
                                        budget_extra[q] -= 1
                    r[u] = r_extra.get(u, 0)
                    budget_map[u] = budget_extra.get(u, 0)
                    self.group_expand(u, r_extra, budget_extra, visited_group, r, budget_map, vis_group, new_group)

    def group_shrink(
        self,
        v: int,
        r: Dict[int, int],
        budget_map: Dict[int, int],
        vis_group: Dict[int, int],
    ) -> None:
        for w in self.partition.datagraph.adj_list[v]:
            if w in vis_group and vis_group[w] != 0:
                r[w] = r.get(w, 0) + 1
                budget_map[w] = budget_map.get(w, 0) - 1
                if budget_map[w] < r[w]:
                    vis_group[w] = 0
                    r.pop(w, None)
                    budget_map.pop(w, None)
                    self.group_shrink(w, r, budget_map, vis_group)

    def get_group_followers(
        self,
        group: List[int],
        followers_gain: List[Tuple[int, int]],
        vis_group: Dict[int, int],
        group_k: int,
    ) -> int:
        num_n = self.partition.max_layer + 100.0
        heap: List[Tuple[float, int]] = []
        in_queue: Dict[int, int] = {}
        degplus: Dict[int, int] = {}
        status: Dict[int, int] = {}
        survive: Dict[int, int] = {}
        mp_vis: Dict[int, float] = {}
        for v in group:
            heapq.heappush(heap, (self.partition.coreness[v] + self.partition.shells[v] / num_n, v))
            in_queue[v] = 1
        before_k = 0
        while heap:
            priority, v = heapq.heappop(heap)
            coreness_v = self.partition.coreness[v]
            if v in survive:
                coreness_v += survive[v]
            if coreness_v > before_k:
                for vid, pr in mp_vis.items():
                    heapq.heappush(heap, (pr, vid))
                    in_queue[vid] = 1
                mp_vis.clear()
                before_k = coreness_v
            in_queue.pop(v, None)
            degplus_v = 0
            if v in vis_group and v not in status:
                status[v] = 1
                survive[v] = group_k - self.partition.coreness[v]
                degplus[v] = self.partition.coreness[v] + 1
                in_queue[v] = 1
                for u in self.partition.datagraph.adj_list[v]:
                    if u in vis_group:
                        continue
                    if u not in status and u not in in_queue:
                        if (
                            self.partition.coreness[v] == self.partition.coreness[u]
                            and self.partition.shells[v] < self.partition.shells[u]
                        ):
                            heapq.heappush(
                                heap,
                                (self.partition.coreness[u] + self.partition.shells[u] / num_n, u),
                            )
                            in_queue[u] = 1
                        elif (
                            self.partition.coreness[v] < self.partition.coreness[u]
                            and self.partition.coreness[u] < group_k
                        ):
                            heapq.heappush(
                                heap,
                                (self.partition.coreness[u] + self.partition.shells[u] / num_n, u),
                            )
                            in_queue[u] = 1
                continue
            survive_v = survive.get(v, 0)
            for u in self.partition.datagraph.adj_list[v]:
                if self.partition.coreness[u] >= self.partition.coreness[v] + survive_v + 1 or u in vis_group:
                    degplus_v += 1
                elif u not in status:
                    if self.partition.coreness[u] == self.partition.coreness[v] + survive_v:
                        if survive_v > 0:
                            degplus_v += 1
                        elif self.partition.shells[u] > self.partition.shells[v] or u in in_queue:
                            degplus_v += 1
                elif status.get(u) == 1:
                    survive_u = survive.get(u, 0)
                    if (
                        self.partition.coreness[u] + survive_u > self.partition.coreness[v] + survive_v
                        or (
                            self.partition.coreness[u] + survive_u == self.partition.coreness[v] + survive_v
                            and u in in_queue
                        )
                    ):
                        degplus_v += 1
            if degplus_v >= self.partition.coreness[v] + survive_v + 1:
                for u in self.partition.datagraph.adj_list[v]:
                    if u in vis_group:
                        continue
                    if v not in status and u not in in_queue:
                        if (
                            u not in status
                            and self.partition.coreness[v] == self.partition.coreness[u]
                            and self.partition.shells[v] < self.partition.shells[u]
                        ):
                            heapq.heappush(
                                heap,
                                (self.partition.coreness[u] + self.partition.shells[u] / num_n, u),
                            )
                            in_queue[u] = 1
                    elif status.get(v) == 1 and u not in in_queue:
                        if (
                            u not in status
                            and self.partition.coreness[v] + survive_v == self.partition.coreness[u]
                        ):
                            heapq.heappush(
                                heap,
                                (self.partition.coreness[u] + self.partition.shells[u] / num_n, u),
                            )
                            in_queue[u] = 1
                status[v] = 1
                survive[v] = survive.get(v, 0) + 1
                degplus[v] = degplus_v
                mp_vis[v] = self.partition.coreness[v] + survive[v]
            else:
                status[v] = 2
                degplus[v] = degplus_v
                self.shrink_groups(v, degplus, status, vis_group, survive, mp_vis)
        gains = 0
        for u, inc in survive.items():
            if u in vis_group:
                inc_val = group_k - self.partition.coreness[u]
                followers_gain.append((u, inc_val))
                gains += inc_val
            else:
                followers_gain.append((u, inc))
                gains += inc
        return gains

    def shrink_groups(
        self,
        v: int,
        degplus: Dict[int, int],
        status: Dict[int, int],
        vis_group: Dict[int, int],
        survive: Dict[int, int],
        mp_vis: Dict[int, float],
    ) -> None:
        shrink_nodes: List[int] = []
        for u in self.partition.datagraph.adj_list[v]:
            if u in status and status[u] == 1 and u not in vis_group:
                survive_u = survive.get(u, 0)
                survive_v = survive.get(v, 0)
                if self.partition.coreness[u] + survive_u == self.partition.coreness[v] + survive_v + 1:
                    degplus[u] -= 1
                    if degplus[u] < self.partition.coreness[u] + survive_u:
                        status[u] = 2
                        if u in survive:
                            survive[u] -= 1
                            if survive[u] == 0:
                                survive.pop(u, None)
                        if u in mp_vis:
                            mp_vis.pop(u, None)
                        shrink_nodes.append(u)
        for u in shrink_nodes:
            self.shrink_groups(u, degplus, status, vis_group, survive, mp_vis)

    def _register_edge(self, u: int, v: int) -> bool:
        if (u, v) in self.partition.datagraph.hash_edges:
            return False
        self.anchor_edges.append((u, v))
        self.partition.datagraph.adj_list[u].append(v)
        self.partition.datagraph.adj_list[v].append(u)
        self.partition.datagraph.hash_edges.add((u, v))
        self.partition.datagraph.hash_edges.add((v, u))
        cu = self.partition.coreness[u]
        cv = self.partition.coreness[v]
        self.partition.nbrk[u][cv] += 1
        self.partition.nbrk[v][cu] += 1
        if cu >= cv:
            self.partition.deggt[v] += 1
        if cv >= cu:
            self.partition.deggt[u] += 1
        return True

    def final_insert(self, check_path: str) -> None:
        with open(check_path, "w", encoding="utf-8") as outfile:
            for u, v in self.anchor_edges:
                src = self.partition.datagraph.seq2id.get(u, u)
                dst = self.partition.datagraph.seq2id.get(v, v)
                outfile.write(f"{src}\t{dst}\n")
        print("Data written to insert file successfully.")
