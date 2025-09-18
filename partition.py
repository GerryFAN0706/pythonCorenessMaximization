from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Set, Tuple

from .datagraph import DataGraph

INITIAL = -1
MAX_COST = 10**9


class Partition:
    def __init__(self, datagraph: DataGraph) -> None:
        self.datagraph = datagraph
        n = len(datagraph.adj_list)
        self.coreness: List[int] = [0] * n
        self.total_order: List[int] = []
        self.order_pointer: List[int] = []
        self.shell_tag: List[int] = []
        self.anchor_verts: Set[int] = set()
        self.shells: List[int] = [0] * n
        self.deggt: List[int] = [0] * n
        self.nbrk: List[DefaultDict[int, int]] = [defaultdict(int) for _ in range(n)]
        self.max_degree = max((len(neigh) for neigh in datagraph.adj_list), default=0)
        self.max_coreness = 0
        self.max_layer = 0
        self.coreness_ins: List[int] = [0] * n
        self.pre_coreness = -1
        self.after_coreness = 0

    def _reset_order_structures(self) -> None:
        self.total_order.clear()
        self.order_pointer.clear()
        self.shell_tag.clear()

    def core_decomposition(self) -> None:
        self._reset_order_structures()
        self.max_coreness = 0
        n = len(self.datagraph.adj_list)
        self.coreness = []
        self.shells = [0] * n
        max_degs = 0
        for vid in range(n):
            self.total_order.append(vid)
            degree = len(self.datagraph.adj_list[vid])
            max_degs = max(max_degs, degree)
            self.coreness.append(degree)
            self.order_pointer.append(INITIAL)
        print(f"max_degree:{max_degs}")
        self.total_order.sort(key=lambda v: self.coreness[v])
        bi = 0
        for pos, vid in enumerate(self.total_order):
            degree = self.coreness[vid]
            self.order_pointer[vid] = pos
            while degree >= bi:
                self.shell_tag.append(pos)
                bi += 1
        for v in self.total_order:
            for u in self.datagraph.adj_list[v]:
                if self.coreness[u] > self.coreness[v]:
                    du = self.coreness[u]
                    pu = self.order_pointer[u]
                    pw = self.shell_tag[du]
                    w = self.total_order[pw]
                    if u != w:
                        self.total_order[pu], self.total_order[pw] = self.total_order[pw], self.total_order[pu]
                        self.order_pointer[u] = pw
                        self.order_pointer[w] = pu
                    self.shell_tag[du] += 1
                    self.coreness[u] -= 1
        sum_coreness = 0
        for vid in range(n):
            self.max_coreness = max(self.max_coreness, self.coreness[vid])
            sum_coreness += self.coreness[vid]
        print(f"all coreness:{sum_coreness}")
        if self.pre_coreness == -1:
            self.pre_coreness = sum_coreness
        else:
            self.after_coreness = sum_coreness
            print(f"coreness gain:{self.after_coreness - self.pre_coreness}")

    def layer_decomposition(self) -> None:
        self.shells = [0] * len(self.coreness)
        coreness_copy = sorted((c, i) for i, c in enumerate(self.coreness))
        if not coreness_copy:
            return
        k = coreness_copy[0][0]
        pos = 0
        while pos < len(coreness_copy):
            shell_tag_: List[int] = []
            shell_deg: Dict[int, int] = {}
            in_shellk: Dict[int, int] = {}
            vert_set_: Dict[int, List[int]] = {}
            while pos < len(coreness_copy) and coreness_copy[pos][0] == k:
                vertex = coreness_copy[pos][1]
                shell_tag_.append(vertex)
                shell_deg[vertex] = 0
                in_shellk[vertex] = k
                vert_set_[vertex] = []
                pos += 1
            for u in shell_tag_:
                for v in self.datagraph.adj_list[u]:
                    if self.coreness[v] >= k:
                        shell_deg[u] += 1
                        if v in in_shellk:
                            vert_set_[u].append(v)
            cnt = 1
            to_delete: deque[int] = deque()
            for u in shell_tag_:
                if shell_deg[u] <= k:
                    to_delete.append(u)
                    self.shells[u] = cnt
                    self.max_layer = max(self.max_layer, self.shells[u])
            while to_delete:
                tmp_que: deque[int] = deque()
                cnt += 1
                while to_delete:
                    u = to_delete.popleft()
                    for v in vert_set_[u]:
                        if self.shells[v] == 0:
                            shell_deg[v] -= 1
                            if shell_deg[v] <= k:
                                tmp_que.append(v)
                                self.shells[v] = cnt
                                self.max_layer = max(self.max_layer, self.shells[v])
                to_delete = tmp_que
            if pos < len(coreness_copy):
                k = coreness_copy[pos][0]

    def p_decomposition(self) -> None:
        self.core_decomposition()
        self.layer_decomposition()
        n = len(self.datagraph.adj_list)
        self.deggt = [0] * n
        self.nbrk = [defaultdict(int) for _ in range(n)]
        for u in self.total_order:
            for v in self.datagraph.adj_list[u]:
                if self.coreness[v] >= self.coreness[u]:
                    self.deggt[u] += 1
                kv = self.coreness[v]
                self.nbrk[u][kv] += 1

    def p_maintenance(self, anchor: int, k: int, followers: List[int]) -> None:
        self.anchor_verts.add(anchor)
        before_anchor = self.coreness[anchor]
        if before_anchor < len(self.coreness_ins):
            self.coreness_ins[before_anchor] += 1
        self.coreness[anchor] = k
        self.max_coreness = max(self.max_coreness, k)
        for v in self.datagraph.adj_list[anchor]:
            if self.coreness[v] > before_anchor and self.coreness[v] <= k:
                self.deggt[v] += 1
                if self.coreness[v] != k:
                    self.deggt[anchor] -= 1
            if self.coreness[v] == before_anchor:
                self.deggt[anchor] -= 1
            self.nbrk[v][before_anchor] -= 1
            if self.nbrk[v][before_anchor] == 0:
                del self.nbrk[v][before_anchor]
            self.nbrk[v][k] += 1
        for u in followers:
            if self.coreness[u] < len(self.coreness_ins):
                self.coreness_ins[self.coreness[u]] += 1
            self.coreness[u] += 1
            for v in self.datagraph.adj_list[u]:
                if self.coreness[v] == self.coreness[u]:
                    self.deggt[v] += 1
                if self.coreness[v] == self.coreness[u] - 1:
                    self.deggt[u] -= 1
                self.nbrk[v][self.coreness[u] - 1] -= 1
                if self.nbrk[v][self.coreness[u] - 1] == 0:
                    del self.nbrk[v][self.coreness[u] - 1]
                self.nbrk[v][self.coreness[u]] += 1
        self.layer_decomposition()

    def group_maintenance(self, group: List[int], followers_final: List[Tuple[int, int]]) -> None:
        for u in group:
            self.anchor_verts.add(u)
        for vid, inc in followers_final:
            before_anchor = self.coreness[vid]
            if before_anchor < len(self.coreness_ins):
                self.coreness_ins[before_anchor] += 1
            self.coreness[vid] += inc
            self.max_coreness = max(self.max_coreness, self.coreness[vid])
            for v in self.datagraph.adj_list[vid]:
                if self.coreness[v] > before_anchor and self.coreness[v] <= self.coreness[vid]:
                    self.deggt[v] += 1
                    if self.coreness[v] != self.coreness[vid]:
                        self.deggt[vid] -= 1
                if self.coreness[v] == before_anchor:
                    self.deggt[vid] -= 1
                self.nbrk[v][before_anchor] -= 1
                if self.nbrk[v][before_anchor] == 0:
                    del self.nbrk[v][before_anchor]
                self.nbrk[v][self.coreness[vid]] += 1
        self.layer_decomposition()
