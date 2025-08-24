import random
from collections import deque

"""
Crossover operators for permutation-encoded GAs (e.g., TSP routes).

Expected input for all methods:
- parent_1, parent_2: lists of unique, hashable items of equal length (a permutation).

Expected output for all methods:
- (child_1, child_2): two lists (same length as parents) representing valid offspring permutations.
"""


class Crossover:
    """Collection of crossover operators for permutations.

    Methods assume parents are permutations (no duplicates) and return two
    children of the same length. No validation is performed.
    """

    def __init__(self):
        return

    def order_crossover(self, parent_1, parent_2):
        """
        Order Crossover (OX).

        Args:
            parent_1, parent_2 (list): equal-length permutations.
        Returns:
            (child_1, child_2): offspring preserving relative order of
            non-segment elements from the opposite parent.
        """
        size = len(parent_1)

        # Select an arbitrary slice
        i, j = sorted(random.sample(range(size + 1), 2))

        # child 1
        child_1 = [None] * size                  # Use None instead of 0 for clarity
        child_1[i:j] = parent_1[i:j]             # copy fixed segment

        # Create set for O(1) lookup of copied items
        copied_items = set(parent_1[i:j])
        
        # Rotation of parent_2 starting at j to preserve order; filter copied items
        # Use deque for O(1) popleft operations
        rotated = parent_2[j:] + parent_2[:j]
        order = deque(item for item in rotated if item not in copied_items)

        # Fill remaining positions
        for idx in range(j, j + size):
            pos = idx % size
            if child_1[pos] is None:
                child_1[pos] = order.popleft()

        # child 2
        child_2 = [None] * size
        child_2[i:j] = parent_2[i:j]

        # Create set for O(1) lookup of copied items
        copied_items = set(parent_2[i:j])
        
        rotated = parent_1[j:] + parent_1[:j]
        order = deque(item for item in rotated if item not in copied_items)

        # Fill remaining positions
        for idx in range(j, j + size):
            pos = idx % size
            if child_2[pos] is None:
                child_2[pos] = order.popleft()

        return child_1, child_2

    def partially_mapped_crossover(self, parent_1, parent_2):
        """
        Partially Mapped Crossover (PMX).

        Args:
            parent_1, parent_2 (list): equal-length permutations.
        Returns:
            (child_1, child_2): offspring where a central segment is mapped
            between parents and the remainder keeps positional information.
        """
        size = len(parent_1)

        # Select an arbitrary segment
        i, j = sorted(random.sample(range(size + 1), 2))

        # Create position lookup maps for O(1) index operations
        pos_map_p1 = {val: idx for idx, val in enumerate(parent_1)}
        pos_map_p2 = {val: idx for idx, val in enumerate(parent_2)}
        
        # child 1
        child_1 = [None] * size
        child_1[i:j] = parent_1[i:j]
        
        # Create sets for O(1) membership testing
        segment_p1 = set(parent_1[i:j])
        segment_p2 = set(parent_2[i:j])
        used_in_child1 = set(parent_1[i:j])

        # Create mapping for conflicting elements
        mapping = {}
        for idx in range(i, j):
            val_p1 = parent_1[idx]
            val_p2 = parent_2[idx]
            if val_p2 not in segment_p1:  # val_p2 is not in the copied segment
                # Find where val_p2 should go by following the mapping chain
                target_pos = idx
                while parent_2[target_pos] in segment_p1:
                    target_pos = pos_map_p2[parent_1[pos_map_p1[parent_2[target_pos]]]]
                child_1[target_pos] = val_p2
                used_in_child1.add(val_p2)

        # Fill remaining positions from parent_2
        for idx in range(size):
            if child_1[idx] is None and parent_2[idx] not in used_in_child1:
                child_1[idx] = parent_2[idx]
                used_in_child1.add(parent_2[idx])

        # child 2
        child_2 = [None] * size
        child_2[i:j] = parent_2[i:j]
        used_in_child2 = set(parent_2[i:j])

        # Create mapping for conflicting elements
        for idx in range(i, j):
            val_p1 = parent_1[idx]
            val_p2 = parent_2[idx]
            if val_p1 not in segment_p2:  # val_p1 is not in the copied segment
                # Find where val_p1 should go by following the mapping chain
                target_pos = idx
                while parent_1[target_pos] in segment_p2:
                    target_pos = pos_map_p1[parent_2[pos_map_p2[parent_1[target_pos]]]]
                child_2[target_pos] = val_p1
                used_in_child2.add(val_p1)

        # Fill remaining positions from parent_1
        for idx in range(size):
            if child_2[idx] is None and parent_1[idx] not in used_in_child2:
                child_2[idx] = parent_1[idx]
                used_in_child2.add(parent_1[idx])

        return child_1, child_2

    def cycle_crossover(self, parent_1, parent_2):
        """
        Cycle Crossover (CX).

        Args:
            parent_1, parent_2 (list): equal-length permutations.
        Returns:
            (child_1, child_2): offspring built by alternating cycles that
            map element positions between parents.
        """
        size = len(parent_1)

        # Create position lookup maps for O(1) index operations
        pos_map_p1 = {val: idx for idx, val in enumerate(parent_1)}
        pos_map_p2 = {val: idx for idx, val in enumerate(parent_2)}

        # Initialize children
        child_1 = [None] * size
        child_2 = [None] * size
        
        # Track visited positions
        visited = [False] * size
        cycle_num = 0

        # Find and process cycles
        for start_pos in range(size):
            if visited[start_pos]:
                continue
                
            # Trace the cycle starting from start_pos
            cycle_positions = []
            pos = start_pos
            
            while not visited[pos]:
                visited[pos] = True
                cycle_positions.append(pos)
                # Follow the mapping: parent_1[pos] -> find where it is in parent_2
                val = parent_1[pos]
                pos = pos_map_p2[val]
            
            # Assign cycle elements to children based on cycle number
            if cycle_num % 2 == 0:  # even cycles
                for pos in cycle_positions:
                    child_1[pos] = parent_1[pos]
                    child_2[pos] = parent_2[pos]
            else:  # odd cycles - swap
                for pos in cycle_positions:
                    child_1[pos] = parent_2[pos]
                    child_2[pos] = parent_1[pos]
            
            cycle_num += 1

        return child_1, child_2

    def edge_recombination_crossover(self, parent_1, parent_2):
        """
        Edge Recombination Crossover (ERX).
        Preserves adjacency information from both parents by building
        an edge table and constructing offspring using shared edges.
        
        Args:
            parent_1, parent_2 (list): equal-length permutations.
        Returns:
            (child_1, child_2): offspring that preserve edge information
            from both parents.
        """
        size = len(parent_1)
        
        def build_edge_table(p1, p2):
            """Build adjacency table from both parents."""
            edge_table = {}
            
            for i in range(size):
                city = p1[i]
                if city not in edge_table:
                    edge_table[city] = set()
                
                # Add neighbors from parent 1
                prev_city = p1[(i - 1) % size]
                next_city = p1[(i + 1) % size]
                edge_table[city].add(prev_city)
                edge_table[city].add(next_city)
                
                # Add neighbors from parent 2
                p2_index = p2.index(city)
                prev_city = p2[(p2_index - 1) % size]
                next_city = p2[(p2_index + 1) % size]
                edge_table[city].add(prev_city)
                edge_table[city].add(next_city)
            
            return edge_table
        
        def construct_offspring(edge_table):
            """Construct offspring using edge table."""
            offspring = []
            remaining = set(parent_1)
            edge_table_copy = {city: neighbors.copy() for city, neighbors in edge_table.items()}
            
            # Start with random city
            current = random.choice(list(remaining))
            offspring.append(current)
            remaining.remove(current)
            
            while remaining:
                # Remove current city from all edge lists
                for city in edge_table_copy:
                    edge_table_copy[city].discard(current)
                
                # Find next city
                candidates = edge_table_copy[current] & remaining
                
                if candidates:
                    # Choose city with fewest connections (tie-breaking randomly)
                    min_connections = min(len(edge_table_copy[city] & remaining) for city in candidates)
                    best_candidates = [city for city in candidates 
                                     if len(edge_table_copy[city] & remaining) == min_connections]
                    current = random.choice(best_candidates)
                else:
                    # No connected cities available, choose randomly
                    current = random.choice(list(remaining))
                
                offspring.append(current)
                remaining.remove(current)
            
            return offspring
        
        # Build edge table
        edge_table = build_edge_table(parent_1, parent_2)
        
        # Construct two offspring
        child_1 = construct_offspring(edge_table)
        child_2 = construct_offspring(edge_table)
        
        return child_1, child_2
