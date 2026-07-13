import time
import tracemalloc
from typing import Tuple, Dict, Any, Type, List

from environment.discrete_world import DiscreteWorld
from action.action_calculator import ActionCalculator
from search.search_algorithms import BaseSearch

class Benchmark:
    @staticmethod
    def run_benchmark(
        search_class: Type[BaseSearch],
        world: DiscreteWorld,
        calculator: ActionCalculator,
        start_state: Tuple[int, int, int],
        goal_state: Tuple[int, int, int],
        max_depth: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Runs a benchmarking suite on a specific search algorithm.
        Returns a dictionary with runtime, memory footprint, path found, and cost.
        """
        search_instance = search_class(world, calculator, **kwargs)

        tracemalloc.start()
        start_time = time.time()

        result = search_instance.search(start_state, goal_state, max_depth)

        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtime = end_time - start_time

        if result:
            path, cost, nodes_expanded = result
        else:
            path, cost, nodes_expanded = None, float('inf'), 0

        metrics = {
            'algorithm': search_class.__name__,
            'runtime_seconds': runtime,
            'peak_memory_bytes': peak_mem,
            'nodes_expanded': nodes_expanded,
            'solution_cost': cost,
            'path_length': len(path) if path else 0,
            'path_found': path is not None
        }

        return metrics
