from __future__ import annotations

import argparse
import sys

from .datagraph import DataGraph
from .master import Master
from .partition import Partition


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coreness maximization via edge insertion")
    parser.add_argument("dataset", help="Path to input graph file")
    parser.add_argument("budget", type=int, help="Maximum number of edges to insert")
    parser.add_argument("insert_path", help="Output path for inserted edges")
    parser.add_argument("check", type=int, choices=[0, 1, 2], help="Mode: 0=run heuristic, 1=verify, 2=core decomposition only")
    parser.add_argument("mode", type=int, choices=[0, 1, 2], help="Algorithm mode: 0=vertex, 1=group, 2=hybrid")
    return parser.parse_args(argv)


def run(argv: list[str]) -> None:
    args = parse_args(argv)
    print(f"dataset: {args.dataset}")
    datagraph = DataGraph(args.dataset)
    partition = Partition(datagraph)
    if args.check == 1:
        partition.core_decomposition()
        partition.datagraph.add_edges(args.insert_path)
        partition.core_decomposition()
    elif args.check == 0:
        master = Master(partition)
        master.anchoring(args.budget, args.mode, args.insert_path)
        print(f"Number of Followers:{master.nfs}")
    elif args.check == 2:
        partition.core_decomposition()
    else:
        raise ValueError("Unsupported check mode")


def main() -> None:
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
