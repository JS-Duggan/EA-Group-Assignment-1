import math

class loadTSP:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dimension = None
        self.coordinates = []
        self.distance_matrix = None
        self._parse_tsplib_file()
        self._compute_distance_matrix()

    def _parse_tsplib_file(self):
        with open(self.file_path, "r") as f:
            lines = f.readlines()

        reading_coords = False
        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            if line.startswith("DIMENSION"):
                self.dimension = int(line.split(":")[1].strip())

            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue

            elif line.startswith("EOF"):
                break

            elif reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    # TSPLib: node_id x y
                    _, x, y = parts
                    self.coordinates.append((float(x), float(y)))

        if len(self.coordinates) != self.dimension:
            raise ValueError(
                f"Mismatch: DIMENSION={self.dimension}, but found {len(self.coordinates)} coordinates."
            )

    # def _compute_distance_matrix(self):
    #     """Creates an NxN matrix with Euclidean distances between cities."""
    #     n = self.dimension
    #     self.distance_matrix = [[0.0] * n for _ in range(n)]
    #     for i in range(n):
    #         for j in range(n):
    #             if i != j:
    #                 self.distance_matrix[i][j] = math.dist(self.coordinates[i], self.coordinates[j])
    
    def _compute_distance_matrix(self):
        """Creates an NxN matrix with Euclidean distances between cities."""
        n = self.dimension
        self.distance_matrix = [0.0] * (n * n)
        for row in range(n):
            for col in range(n):
                if row != col:
                    dx = self.coordinates[row][0] - self.coordinates[col][0]
                    dy = self.coordinates[row][1] - self.coordinates[col][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    self.distance_matrix[row * n + col] = dist

    # def get_coordinates(self):
    #     return self.coordinates
    
    def get_distance(self, row, col):
        return self.distance_matrix[row * self.dimension + col]

    def get_distance_matrix(self):
        return self.distance_matrix


# Example usage
# if __name__ == "__main__":
#     tsp = TSP("usa13509.tsp")
#     coords = tsp.get_coordinates()
#     dist_matrix = tsp.get_distance_matrix()

#     print(f"Loaded {len(coords)} cities.")
#     print("First 5 coordinates:", coords[:5])
#     print("Distance between city 0 and 1:", dist_matrix[0][1])

if __name__ == "__main__":
    tsp = TSP("usa13509.tsp")
    print(f"Loaded {len(tsp.coordinates)} cities.")
    print("First 5 coordinates:", tsp.coordinates[:5])
    print("Distance between city 0 and 1:", tsp.get_distance(0, 1))
