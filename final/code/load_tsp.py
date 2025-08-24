import math
import numpy as np

"""Example usage:
if __name__ == "__main__":
    tsp = TSP("usa13509.tsp")
    coords = tsp.getCoordinates()
    distMatrix = tsp.getDistanceMatrix()
    print(f"Loaded {len(coords)} cities.")
    print("First 5 coordinates:", coords[:5])
    print("Distance between city 0 and 1:", distMatrix[0][1])
if __name__ == "__main__":
    tsp = TSP("usa13509.tsp")
    print(f"Loaded {len(tsp.coordinates)} cities.")
    print("First 5 coordinates:", tsp.coordinates[:5])
    print("Distance between city 0 and 1:", tsp.getDistance(0, 1))
"""

class LoadTSP:
    def __init__(self, file_path):
        """ 
        Loads and prepares the TSP instance
        
        Args:
            filePath (string): the file containing the instance of TSP that will be processed.
        """

        self.file_path = file_path
        self.dimension = None
        self.coordinates = []
        self.distance_matrix = None
        self._parse_tsplib_file()
        self._compute_distance_matrix()

    def _parse_tsplib_file(self):
        """
        Loads in the data for the TSP instance based on the filePath variable. Data is saved into a local variable coordinates.
        """
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

    def _compute_distance_matrix(self):
        """
        Creates an NxN matrix with Euclidean distances between cities.
        """
        n = self.dimension
        self.distance_matrix = np.zeros((n, n))
        for row in range(n):
            for col in range(n):
                if row != col:
                    dx = self.coordinates[row][0] - self.coordinates[col][0]
                    dy = self.coordinates[row][1] - self.coordinates[col][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    self.distance_matrix[row, col] = dist

    def get_distance(self, row, col):
        """
        Returns the distance between two cities at index row and col.
        Args:
            row, col (int): start location and end location which distance is being calculated for. Indexes into a matrix representing a 2D array.
        Returns:
            distance (int): the distance between the two locations. 
        """
        return self.distance_matrix[row, col]

    def get_distance_matrix(self):
        """
        Returns the full distance matrix of the TSP.
        
        Returns:
            distanceMatrix (Matrix(int)): the distance matrix
        """
        return self.distance_matrix
    
    def get_dimension(self):
        """
        Returns the number of nodes in the TSP.
        
        Returns:
            dimension (int): the dimension. 
        """
        return self.dimension

