from __future__ import annotations

from typing import Dict, List, Set, Tuple


class DataGraph:
    """In-memory adjacency representation with id remapping."""

    def __init__(self, file_name: str) -> None:
        self.adj_list: List[List[int]] = []
        self.id2seq: Dict[int, int] = {}
        self.seq2id: Dict[int, int] = {}
        self.hash_edges: Set[Tuple[int, int]] = set()
        self._load(file_name)

    def _ensure_vertex(self, vertex_id: int) -> int:
        if vertex_id in self.id2seq:
            return self.id2seq[vertex_id]
        seq = len(self.adj_list)
        self.id2seq[vertex_id] = seq
        self.seq2id[seq] = vertex_id
        self.adj_list.append([])
        return seq

    def _add_edge_internal(self, src_seq: int, dst_seq: int) -> None:
        if src_seq == dst_seq:
            return
        if (src_seq, dst_seq) in self.hash_edges:
            return
        self.adj_list[src_seq].append(dst_seq)
        self.adj_list[dst_seq].append(src_seq)
        self.hash_edges.add((src_seq, dst_seq))
        self.hash_edges.add((dst_seq, src_seq))

    def _parse_edge_line(self, line: str) -> Tuple[int, int] | None:
        line = line.strip()
        if not line:
            return None
        parts = line.replace('\t', ' ').split()
        if len(parts) < 2:
            return None
        try:
            src = int(float(parts[0]))
            dst = int(float(parts[1]))
        except ValueError:
            return None
        return src, dst

    def _load(self, file_name: str) -> None:
        with open(file_name, "r", encoding="utf-8") as infile:
            header = infile.readline()
            if not header:
                return
            for line in infile:
                parsed = self._parse_edge_line(line)
                if not parsed:
                    continue
                src, dst = parsed
                if src == dst:
                    continue
                src_seq = self._ensure_vertex(src)
                dst_seq = self._ensure_vertex(dst)
                self._add_edge_internal(src_seq, dst_seq)

    def add_edges(self, file_name: str) -> None:
        with open(file_name, "r", encoding="utf-8") as infile:
            for line in infile:
                parsed = self._parse_edge_line(line)
                if not parsed:
                    continue
                src, dst = parsed
                if src == dst:
                    continue
                src_seq = self._ensure_vertex(src)
                dst_seq = self._ensure_vertex(dst)
                self._add_edge_internal(src_seq, dst_seq)
