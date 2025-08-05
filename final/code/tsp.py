class TSP:
  graph = [[]]
  def __init__(self, path):
    """Init class

    takes path to test case as input
    edits private variable 'graph'
    graph is 2d array, where graph[i][j] = distance between i and j
    """
    self.graph = [[]]
    return
  
  def cost(self, perm):
    """Cost
    
    given a permutation, uses graph to calculate total cost
    """
    cost = 0
    for i in range(len(perm) - 1):
      cost += self.graph[perm[i]][perm[i + 1]]
    cost += self.graph[perm[-1]][perm[0]]
      




  def swap(self, perm, cost):
    """Swap
    
    uses class variable graph, along with an initial permutation/cost to 
    test for a better cost using the swap neighbourhood search.
    """

    return solution