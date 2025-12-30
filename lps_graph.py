"""LPS Ramanujan Graph (Lubotzky-Phillips-Sarnak) construction.

This module provides a class to construct LPS expander graphs, which are
optimal expanders based on quaternion algebras over finite fields.

References:
    Lubotzky, Phillips, Sarnak. "Ramanujan Graphs"
    "Ramanujan graphs." Combinatorica 8, no. 3 (1988): 261-277.
"""

import math
import time
from itertools import product as iterproduct
import scipy.sparse
from scipy.sparse.linalg import eigsh
from sage.all import graphs, is_prime, Graph, FiniteField, GL, legendre_symbol, sqrt

class LPSGraph:
    """LPS (Lubotzky-Phillips-Sarnak) Ramanujan expander graph.

    This class constructs (p+1)-regular Ramanujan graphs X(p, q) based on
    quaternion solutions in PGL(2, q).

    Attributes:
        p (int): Prime ≡ 1 (mod 4), determines regularity (p+1).
        q (int): Prime ≡ 1 (mod 4), field size for matrix entries.
        generators (list): 2x2 matrices over GF(q) from quaternion solutions.
        graph (Graph): SageMath Graph object representing the Cayley graph X(p, q).
    """

    #Class attribute for maximum size of graph generated to prevent memory issues. 
    MAXIMUM_GRAPH_SIZE = 10**5 # Adjust based on system capabilities

    def __init__(self, beta: float = None, n: int = None, l: float = None,
                 p: int = None, q: int = None, silent: bool = False):
        """Initialize an LPS Ramanujan graph.

        Two modes of initialization:
        1. Direct: Provide primes p and q both ≡ 1 (mod 4).
        2. Parametric: Provide beta > 0, n > 0, 0 < l < 1 to auto-search for p, q.

        Args:
            beta (float): Sparsity parameter (used if p, q not provided).
            n (int): Graph size parameter (used if p, q not provided).
            l (float): Expansion parameter in (0, 1) (used if p, q not provided).
            p (int): Prime regularity parameter (≡ 1 mod 4).
            q (int): Prime field size parameter (≡ 1 mod 4).

        Raises:
            ValueError: If input parameters are invalid or primes not found.
            MemoryError: If selected primes will result in a graph that is larger than MAXIMUM_GRAPH_SIZE.
        """

        self.silent = silent

        # Mode 1: Direct initialization with primes p and q
        if p is not None and q is not None:
            try:
                # Validate p and q meet LPS constraints
                if p % 4 != 1 or q % 4 != 1 or not is_prime(p) or not is_prime(q) or p == q:
                    raise ValueError(
                        "Both p and q must be primes congruent to 1 mod 4 and p ≠ q."
                    )
            except TypeError:
                raise ValueError("Both p and q must be integers.")
            self.p = p
            self.q = q

        # Mode 2: Parametric initialization with beta, n, l
        elif beta is not None and n is not None and l is not None:
            try:
                # Validate parameter ranges
                if beta <= 0 or n <= 0 or l <= 0 or l >= 1:
                    raise ValueError(
                        "beta must be > 0, n must be > 0, and 0 < l < 1."
                    )
            except TypeError:
                raise ValueError("beta, n, and l must be numbers.")
            # Auto-search for suitable primes
            q_lower_bound = ((1 - beta) * 2 * n) ** (1 / 3)
            q_upper_bound = (2 * n) ** (1 / 3)
            self._log("Searching for prime q in range [{}, {}]...".format(q_lower_bound, q_upper_bound))
            self.q = LPSGraph.find_prime(q_lower_bound, q_upper_bound)
            p_lower_bound = (1 - beta) * (8 / l ** 2)
            p_upper_bound = 8 / (l ** 2)
            self.p = LPSGraph.find_prime(p_lower_bound, p_upper_bound)
            self._log("Searching for prime p in range [{}, {}]...".format(p_lower_bound, p_upper_bound))
            
            if self.p == self.q:
                raise ValueError(
                    "Unable to find distinct primes found within the specified parameters."
                )

        # Invalid input
        else:
            raise ValueError(
                "Either primes (p, q) must be provided, "
                "or parameters (beta, n, l) must be provided."
            )

        self._log(f"\nSelected primes: p={self.p},q={self.q}")
        
        # Check graph size before building
        graph_size = (self.q - 1)*self.q*(self.q + 1)
        if legendre_symbol(self.p, self.q) == 1:
            graph_size = graph_size/2
        if graph_size > LPSGraph.MAXIMUM_GRAPH_SIZE:
            raise MemoryError(
                f"LPS Graph for selected primes will be too large to construct, with {graph_size} vertices."
            )

        self._log("Finding generators...")
        self.generators = self.LPS_generators()
        self._log("Building graph...")
        self.graph = self.build_graph()
        self._log("Checking expansion...")
        self.is_ramanujan, self.expansion = LPSGraph.graph_is_ramanujan(self.graph)
        if not self.is_ramanujan:
            self._log("Graph constructed but expansion does not satisfy Ramanujan bound. Expansion ", self.expansion, "is not less than ", ((2 * math.sqrt(p))/(p+1)))
            raise ValueError("LPS graph could not be correctly constructed.")
        else:
            self._log(f"Constructed LPS Graph (p={self.p}, q={self.q}):")
            self._log(f"  Vertices: {self.graph.num_verts()}")
            self._log(f"  Edges: {self.graph.num_edges()}")
            self._log(f"  Expansion: {self.expansion:.4f}")


    def _log(self, message):
        """Internal helper to handle logging.
        
        Args:
            message (str): Message to self._log.    
        """
        if not self.silent:
            print(message)


    @staticmethod
    def graph_is_ramanujan(G: Graph):
        """Check whether a d-regular graph is Ramanujan.

        For a d-regular graph, the Ramanujan bound states that every
        nontrivial eigenvalue lambda satisfies |lambda| <= 2*sqrt(d-1).

        This function computes the normalized second-largest eigenvalue
        and compares it against the normalized Ramanujan threshold: 2*sqrt(d-1)/d.

        Args:
            G (Graph): A SageMath Graph object that should be regular.

        Returns:
            tuple: (is_ramanujan (bool), expansion (float)) where
                   `expansion` is the normalized second-largest eigenvalue
                   computed as (second_eigenvalue / degree).

        Raises:
            TypeError: If `G` is not a SageMath Graph object or G is not regular.
        """
        # Ensure the graph is regular before spectral checks
        if not isinstance(G, Graph):
            raise TypeError("G needs to be a SageMath Graph object")
        if not G.is_regular():
            raise TypeError("G needs to be a regular graph")

        # In Sage `G.degree()` returns a list; index 1 gives the uniform
        # degree in a regular graph in our usage.
        degree = G.degree()[0]

        # Compute the normalized expansion (second eigenvalue / degree)
        expansion = LPSGraph.compute_expansion(G)

        # Ramanujan threshold (normalized): 2*sqrt(d-1)/d
        ramanujan_threshold = (2 * math.sqrt(degree - 1)) / degree

        is_ramanujan = expansion <= ramanujan_threshold
        return (is_ramanujan, expansion)


    @staticmethod
    def find_prime(lower_bound: float, upper_bound: float):
        """Find the first prime congruent to 1 mod 4 in the range [lower_bound, upper_bound].
        
        Args:
            lower_bound (float): Lower bound of search range.
            upper_bound (float): Upper bound of search range.
        
        Returns:
            int: First prime p such that p ≡ 1 (mod 4) and lower_bound <= p <= upper_bound.
        
        Raises:
            ValueError: If no such prime exists in range.
        """
        # Search for first prime ≡ 1 (mod 4) in the range
        for k in range(math.ceil(lower_bound), math.floor(upper_bound)+1):
            if is_prime(k) and k % 4 == 1:
                return k
        
        raise ValueError("No prime congruent to 1 mod 4 found within the specified parameters.")


    def LPS_generators(self):
        """Compute LPS generator matrices from quaternion solutions.
        
        Finds all solutions (a₀, a₁, a₂, a₃) to the quaternion equation:
            a₀² + a₁² + a₂² + a₃² = p
        where a₀ is odd and a₁, a₂, a₃ are even. For each solution, creates
        a 2×2 matrix over GF(q):
            M = [a₀+ia₁ , a₂+ia₃] 
                [-a₂+ia₃, a₀-ia₁]
        Then normalizes to get an element in PGL(2, q).
        
        Returns:
            list: Normalized matrix generators over GL(q).
        
        Raises:
            ValueError: If quaternion solution count doesn't match p+1.
        """
        solutions = []

        # The maximum possible value for any a_i is sqrt(p)
        limit = int(math.sqrt(self.p))
        
        search_range = range(-limit, limit + 1)

        # a0 must be odd
        possible_a0 = [a for a in search_range if a % 2 != 0]
        # a1, a2, a3 must be even
        possible_others = [a for a in search_range if a % 2 == 0]

        for a0, a1, a2, a3 in iterproduct(possible_a0, possible_others, possible_others, possible_others):
            if (a0**2 + a1**2 + a2**2 + a3**2) == self.p:
                solutions.append((a0, a1, a2, a3))
        
        # Create finite field and convert solutions to matrix form
        F = FiniteField(self.q)
        generators = set()
        for solution in solutions:
            a0, a1, a2, a3 = solution
            i = sqrt(F(-1))
            # Elements in form [a, b, c, d] represent matrix [[a, b], [c, d]]
            matrix = [F(a0)+i*F(a1), F(a2)+i*F(a3), F(-a2)+i*F(a3), F(a0)-i*F(a1)] 
            generators.add(LPSGraph.normalized_PGL_matrix(matrix, self.q))

        # Verify correct number of generators
        if len(generators) != self.p + 1:
            raise ValueError("Generator count mismatch: expected {}, got {}".format(self.p + 1, len(solutions)))

        return generators


    @staticmethod
    def normalized_PGL_matrix(matrix: list, q:int):
        """Normalize a matrix in GL(2, q) to obtain a canonical coset representative or element of PGL(2, q).
        
        Takes a list of 4 field elements [a, b, c, d] representing the matrix [[a, b], [c, d]]
        and scales all entries by the inverse of the first non-zero element to obtain a
        representative in PGL(2, q) (equivalence classes of matrices up to scalar multiplication).
        
        Args:
            matrix (list): Four-element list [a, b, c, d] representing non-zero matrix in GL(2, q).
            q (int): Prime field size (must match field of matrix elements).
        
        Returns:
            Matrix: GL(2, q) matrix object with normalized entries.
        
        Raises:
            ValueError: If all matrix elements are zero (cannot normalize zero matrix).
            TypeError: If matrix is not a list of length 4 of elements from FiniteField(q)
        """
        F = FiniteField(q)
        general_linear_group = GL(2, q)

        if not isinstance(matrix, list):
            raise TypeError("matrix must be a four-element list [a, b, c, d].")
        for element in matrix:
            if F is not element.parent():
                raise TypeError("Elements in matrix must be members of FiniteField(q).")
        if (matrix[0]*matrix[3] - matrix[1]*matrix[2]) == F(0):
            raise ValueError("Element must in be in GL(2, q)")
        
        # Find first non-zero element and use its inverse as scaling factor
        inv = None
        for element in matrix:
            if element != 0:
                inv = element.inverse()
                break

        # Sanity check: zero matrix cannot be normalized
        if inv is None:
            raise ValueError("Cannot normalize zero matrix.")
        
        # Scale all entries by the inverse to normalize
        normalized_matrix = [inv * element for element in matrix]
        return general_linear_group(normalized_matrix)


    def build_graph(self):
        """Build the LPS Cayley graph from generators.
        
        Constructs a Cayley graph where vertices are elements of GL(2, q) (normalized to PGL(2, q)),
        and edges connect vertices g and g*gen for each generator gen. This creates the
        (p+1)-regular Ramanujan expander X(p, q).
        
        Returns:
            Graph: SageMath Graph object with vertices as normalized GL(2, q) matrices
                   and edges representing multiplication by generators.
        """
        
        G = Graph()
        F = FiniteField(self.q)
        general_linear_group = GL(2, self.q)
        
        # Generate all matrices in PGL(2, q) utilizing condition that determinant != 0
        # Case 1: First entry (a) is 1. Condition: d != b * c
        case1_matrices = [
            [F(1), b, c, d] 
            for b, c, d in iterproduct(F, F, F) 
            if d != b * c
        ]
        # Case 2: First entry (a) is 0, second entry (b) is 1. Condition: c != 0
        case2_matrices = [
            [F(0), F(1), c, d] 
            for c, d in iterproduct(F, F) 
            if c != 0
        ]

        # Combine the two lists and add to the graph
        all_matrices = case1_matrices + case2_matrices
        G.add_vertices([general_linear_group(m) for m in all_matrices])

        edge_list = []
        for v, gen in iterproduct(G.vertices(), self.generators):
            neighbor = v * gen
            neighbor = neighbor.list()
            neighbor = [neighbor[0][0], neighbor[0][1], neighbor[1][0], neighbor[1][1]]
            neighbor = self.normalized_PGL_matrix(neighbor, self.q)
            edge_list.append((v, neighbor))
        G.add_edges(edge_list)

        # Check if p is a quadratic residue modulo q using the Legendre symbol
        # If legendre_p_q == -1 (not a QR), graph is connected; if 1 (is QR), take identity component
        legendre_p_q = legendre_symbol(self.p, self.q)
        if legendre_p_q == 1:
            connected_component = G.connected_components_subgraphs()
            for component in connected_component:
                if general_linear_group.one() in component:
                    G = component
                    break
        
        # Sanity checks
        if not G.is_regular():
            raise ValueError("Constructed graph is not regular.")
        degree = G.degree()[0]
        if degree != self.p + 1:
            raise ValueError("Graph degree mismatch: expected {}, got {}".format(self.p + 1, degree))
        if not G.is_connected():
            raise ValueError("Constructed graph is not connected.")
        if not G.order() == (self.q - 1)*self.q*(self.q + 1)/(2 if legendre_p_q == 1 else 1):
            raise ValueError("Graph order mismatch.")

        return G


    @staticmethod
    def compute_expansion(G: Graph):
        """Compute the spectral expansion of the LPS graph.
        
        Args:
            G (Graph): SageMath Graph object representing the LPS graph.
        
        Returns:
            float: Expansion ratio (second eigenvalue / degree).
        
        Raises:
            ValueError: If G is not a Graph object or G is not regular.
        """
        if not isinstance(G, Graph):
            raise ValueError("G needs to be a SageMath Graph object")
        if not G.is_regular():
            raise ValueError("Graph needs to be regular.")
        
        degree = G.degree()[0]

        # Convert the adjacency matrix to a numpy matrix
        A = G.adjacency_matrix().numpy()

        # Convert from numpy to a scipy matrix to get top eigenvalues using scipy method.
        A_csr = scipy.sparse.csr_matrix(A)
        vals = eigsh(A_csr, k=3, which='LM', return_eigenvectors=False)

        # Take the absolute value of the top 3 eigenvalues. 
        abs_vals = [abs(val) for val in vals]
        abs_vals.sort()

        # If G is bipartite, two trivial eigenvalues instead of one. 
        if G.is_bipartite():
            second_largest_eigenvalue = abs_vals[0]
        else:
            second_largest_eigenvalue = abs_vals[1]
        
        return second_largest_eigenvalue / degree


    def get_graph(self):
        """Get the constructed LPS graph.
        
        Returns:
            Graph: The SageMath Graph object representing the LPS Cayley graph.
        """
        return self.graph


    def plot_graph(self):
        """Visualize the LPS graph using matplotlib.
        
        Creates a 2D visualization of the graph structure using SageMath's
        graph layout algorithms and matplotlib for rendering.
        """
        self._log("Plotting graph " + "...")
        P = self.graph.plot(vertex_size=0.5, edge_thickness=0.1, vertex_labels=False)
        P.save("lps_graph_p{}_q{}.png".format(self.p, self.q))
        self._log("Graph plot saved as lps_graph_p{}_q{}.png".format(self.p, self.q))


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_lps_graph_construction():
    """Test basic LPS graph construction with small primes.
    
    Verifies that an LPS graph can be constructed and satisfies basic
    properties: regularity, connectivity, and correct vertex/edge counts.
    
    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print('-'*70)
    print("Running TEST: LPS Graph.")
    print("WARNING: This test takes some amount of time -- possibly upto 10-15 minutes (due to checking expansion)")

    start_time = time.time()
    try:
        # Construct an LPS graph where (p|q) = 1
        lps1 = LPSGraph(p=5, q=13, silent=True)
        G1 = lps1.get_graph()
        
        # Check basic properties
        assert G1.is_regular(), "Graph should be regular"
        assert G1.is_connected(), "Graph should be connected"
        assert G1.degree()[0] == lps1.p + 1, f"Degree should be {lps1.p + 1}"
        
        # Construct another LPS graph where (p|q) = -1
        lps2 = LPSGraph(p=13, q=29, silent=True)
        G2 = lps2.get_graph()
        
        # Check basic properties
        assert G2.is_regular(), "Graph should be regular"
        assert G2.is_connected(), "Graph should be connected"
        assert G2.degree()[1] == lps2.p + 1, f"Degree should be {lps2.p + 1}"
        end_time = time.time()
        print(f"✓ test_lps_graph_construction passed in {(end_time-start_time)} seconds")
        return True
    except Exception as e:
        print(f"✗ test_lps_graph_construction failed: {e}")
        return False


def test_parametric_initialization():
    """Test parametric initialization with beta, n, l parameters.
    
    Verifies that the parametric initialization mode correctly finds
    suitable primes and constructs a valid graph.
    
    Returns:
        bool: True if initialization succeeds, False otherwise.
    """
    print('-'*70)
    print("Running TEST: parametric_initalization()")
    print('-'*70)
    start_time = time.time()
    try:
        lps = LPSGraph(beta=0.7, n=1100, l=0.8, silent=True)
        G = lps.get_graph()
        
        assert G.is_regular(), "Parametric graph should be regular"
        assert lps.p % 4 == 1, "p should be ≡ 1 (mod 4)"
        assert lps.q % 4 == 1, "q should be ≡ 1 (mod 4)"
        assert lps.p != lps.q, "p and q should be distinct"
        
        end_time = time.time()
        print(f"✓ test_parametric_initialization passed (p={lps.p}, q={lps.q}) in {(end_time-start_time)} seconds ")
        return True
    except Exception as e:
        print(f"✗ test_parametric_initialization failed: {e}")
        return False


def test_normalized_PGL_matrix():
    """Test matrix normalization to PGL(2, q) representative.

    Verifies that the normalized_PGL_matrix method correctly scales
    a matrix to obtain a PGL representative (canonical coset element).

    Returns:
        bool: True if normalization is correct, False otherwise.
    """
    print('-'*70)
    print("Running TEST: normalized_PGL_matrix()")

    start_time = time.time()
    try:

        F = FiniteField(5)
        G = GL(2, 5)
        representative_set = set()
        for element in G:
            items = element.matrix()
            matrix = [items[0][0], items[0][1], items[1][0], items[1][1]]
            normalized = LPSGraph.normalized_PGL_matrix(matrix, 5)
            # Check that normalized matrix is in GL(2, 5)
            assert normalized in GL(2, 5), "Normalized matrix should be in GL(2, 5)"
            representative_set.add(normalized)

        assert len(representative_set) == 120, "Number of elements in PGL(2, 5) is 120"

        end_time = time.time()
        print(f"✓ test_normalized_PGL_matrix passed in {(end_time-start_time)} seconds")
        return True
    except Exception as e:
        print(f"✗ test_normalized_PGL_matrix failed: {e}")
        return False


def test_compute_expansion():
    """Test that spectral expansion is correctly computed. 
    
    The expansion of the Petersen graph is 0.666..
    
    Returns:
        bool: True if expansion is correctly computed, False otherwise
    """
    print('-'*70)
    print("Running TEST: compute_expansion()")

    start_time = time.time()
    try:

        petersen = graphs.PetersenGraph()
        expansion = LPSGraph.compute_expansion(petersen)
        
        assert expansion > 0, "Expansion should be positive"
        petersen_degree = petersen.degree()[0]
        assert expansion < petersen_degree, f"Expansion {expansion:.4f} should be less than degree {petersen_degree}"
        # Petersen graph expansion is exactly 2/3
        assert abs(expansion - float(2/3)) < 1e-6, f"Expected 2/3, got {expansion:.4f}"
        
        end_time = time.time()
        print(f"✓ test_compute_expansion passed (expansion={expansion:.4f}) in {(end_time-start_time)} seconds")
        return True
    except Exception as e:
        print(f"✗ test_compute_expansion failed: {e}")
        return False


def test_graph_is_ramanujan():
    """Test the static method graph_is_ramanujan().

    The Petersen graph is Ramanujan. The degree 10 hypercube graph is not Ramanujan.

    Returns:
        bool: True if Ramanujan graphs are correctly identified, False otherwise.
    """
    print('-'*70)
    print("Running TEST: graph_is_ramanujan()")

    start_time = time.time()
    try:
        petersen = graphs.PetersenGraph()
        cycle_5 = graphs.CubeGraph(10)

        # Test Petersen graph (should be Ramanujan)
        petersen_is_ram, petersen_exp = LPSGraph.graph_is_ramanujan(petersen)
        assert petersen_is_ram, "Petersen graph should be Ramanujan."

        # Test degree 10 hypercube graph (should not be Ramanujan)
        cycle_is_ram, cycle_exp = LPSGraph.graph_is_ramanujan(cycle_5)
        assert not cycle_is_ram, "Degree 10 hypercube graph should not be Ramanujan."

        end_time = time.time()
        print(f"✓ test_graph_is_ramanujan passed in {(end_time-start_time)} seconds")
        return True
    except Exception as e:
        print(f"✗ test_graph_is_ramanujan failed: {e}")
        return False


def run_all_tests():
    """Run all testing functions and report results.
    
    Returns:
        bool: True if all tests pass, False if any test fails.
    """
    print("\n" + "=" * 70)
    print("Running LPS Graph and helpers tests")
    print("=" * 70)
    
    tests = [
        test_lps_graph_construction,
        test_parametric_initialization,
        test_normalized_PGL_matrix,
        test_compute_expansion,
        test_graph_is_ramanujan,
    ]
    
    results = [test() for test in tests]
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70 + "\n")
    
    return all(results)


# ============================================================================
# DEMO / MAIN CODE
# ============================================================================

if __name__ == "__main__":
    #Run tests
    #run_all_tests()
    
    #Sample usage
    lps1 = LPSGraph(p=5, q=13)
    print(lps1.expansion)
    #lps2 = LPSGraph(beta=0.7, n=1100, l=0.8)
    #lps1.plot_graph()