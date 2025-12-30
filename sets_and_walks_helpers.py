"""Helper functions to generate and sample biased sets and random walks. 

Key functionality:
1. Generate/sample from epsilon-balanced subsets of F_2^n.
2. Sample random walks and subsets of random walks on graphs.
"""

import math
import itertools
import random as pyrandom
import numpy as np
from sage.all import graphs, Graph, Subsets

#Global variable for maximum size of sets generated anywhere to prevent memory issues. 
MAXIMUM_SET_SIZE = 10**7 # Adjust based on system capabilities

def generate_biased_set(eps: float, n: int, mode: str = 'sample',
                        sample_size: int = 10000000):
    """Generate a subset S ⊆ F_2^n with Hamming weights in a specified range.

    The set contains all (or a sample of) vectors in F_2^n whose Hamming weight w
    satisfies: ceil(n * (1 - eps) / 2) <= w <= floor(n * (1 + eps) / 2).

    Args:
        eps (float): Bias parameter in (0, 1).
        n (int): Dimension of F_2^n.
        mode (str): 'all' to generate all vectors, 'sample' to randomly sample.
        sample_size (int): Number of samples to generate (only used if mode='sample').

    Yields:
        tuple or list: Vectors as tuples (if mode='all') or lists (if mode='sample').

    Raises:
        ValueError: If eps not in (0,1), n <= 0, or mode invalid.
        MemoryError: If requested size exceeds MAXIMUM_SET_SIZE.

    Notes:
        - Tested with mode='all' up to n=25 with eps=0.05; memory issues may arise beyond that.
        - For large sets, use mode='sample' to avoid memory issues.
    """
    # Validate epsilon parameter
    try:
        if not (0 < eps < 1):
            raise ValueError("epsilon must be in (0,1)")
    except TypeError:
        raise ValueError("epsilon must be a number")

    # Validate dimension parameter
    try:
        if n <= 0:
            raise ValueError("n must be a positive integer")
    except TypeError:
        raise ValueError("n must be a positive integer")

    # Validate mode parameter
    if mode not in ("all", "sample"):
        raise ValueError("mode must be 'all' or 'sample'")

    # Compute weight bounds: vectors with weight w in [low, high]
    low = math.floor(n * ((1 - eps) / 2))
    high = math.ceil(n * ((1 + eps) / 2))
    if low > high:
        raise ValueError(
            "No valid weights for the given epsilon and n. "
            "Make sure that epsilon is not too small."
        )

    # Precompute count of vectors at each allowed weight
    allowed_counts = [math.comb(n, w) for w in range(low, high + 1)]
    total_allowed = sum(allowed_counts)
    #print(f"Total number of vectors in biased set: {total_allowed}")

    if mode == 'sample':
        # Sample mode: yield random vectors with weights in [low, high]
        if sample_size > MAXIMUM_SET_SIZE:
            raise MemoryError(
                f"Requested sample size={sample_size} exceeds max_all={MAXIMUM_SET_SIZE}; "
                f"use smaller sample_size"
            )
        for _ in range(sample_size):
            vec = [0] * n
            # Randomly choose a weight in [low, high]
            weight = pyrandom.randint(low, high)
            # Randomly choose which positions have weight 1
            ones_positions = pyrandom.sample(range(n), weight)
            for pos in ones_positions:
                vec[pos] = 1
            yield vec

    elif mode == 'all':
        # All mode: enumerate all vectors with weights in [low, high]
        if total_allowed > MAXIMUM_SET_SIZE:
            raise MemoryError(
                f"Requested full set size={total_allowed} exceeds max_all={max_all}; "
                f"use mode='sample'"
            )

        for w in range(low, high + 1):
            # Generate all n-bit vectors of weight w using combinations
            for ones in itertools.combinations(range(n), w):
                vec = [0] * n
                for i in ones:
                    vec[i] = 1
                yield tuple(vec)


def sample_one_random_walk(G: Graph, k: int):
    """Sample one random walk of length k in graph G.

    Starts from a random vertex and performs k steps by uniformly choosing
    a random neighbor at each step.

    Args:
        G: SageMath Graph object.
        k (int): Length of the walk (number of steps).

    Returns:
        list: Walk as a tuple of vertex indices, length k+1.

    Raises:
        ValueError: If G is not a SageMath Graph object.
        StopIteration: If a vertex with no neighbors is encountered (can happen if G is directed).
    """

    if not isinstance(G, Graph):
        raise ValueError("G must be a SageMath Graph object.")

    vertices = list(G.vertices())
    current = pyrandom.choice(vertices)
    walk = [current]

    # Perform k random steps
    for _ in range(k):
        neighbors = list(G.neighbors(current))
        if not neighbors:
            raise StopIteration("Vertices with no neighbors encountered during walk.")
        current = pyrandom.choice(neighbors)
        walk.append(current)

    return tuple(walk)


def get_random_subset_of_walks(G, k: int, set_size: int):
    """Get a random subset of size set_size of random walks of length k in graph G.

    Args:
        G: SageMath Graph object.
        k (int): Length of each walk.
        size (int): Number of walks to generate.

    Returns:
        set: Set of random walks, each as a tuple of vertex indices.

    Raises:
        ValueError: If G is not a SageMath Graph object
        MemoryError: If set_size exceeds MAXIMUM_SET_SIZE.
    """

    # Note: This function is not used a lot but is nice to have for testing purposes. 

    if not isinstance(G, Graph):
        raise ValueError("G must be a SageMath Graph object.")

    if set_size > MAXIMUM_SET_SIZE:
        raise MemoryError(
            f"Requested set size={set_size} exceeds max_all={MAXIMUM_SET_SIZE}; reduce size of set"
        )

    walk_set = set()
    count = 0
    # Keep sampling until we have 'set_size' unique walks
    while count < set_size:
        walk = sample_one_random_walk(G, k)
        if walk not in walk_set:
            count += 1
            walk_set.add(walk)

    return walk_set


def get_all_walks(G, k: int):
    """Enumerate all walks of length k in graph G starting from each vertex.

    Args:
        G: A Sage Graph object.
        k (int): Length of walk (number of edges).

    Returns:
        set: Set of all walks, each as a tuple of vertex indices.

    Raises:
        ValueError: If G is not a SageMath Graph object.
        MemoryError: If size of the set of all walks exceeds MAXIMUM_SET_SIZE.
    """

    if not isinstance(G, Graph):
        raise ValueError("G must be a SageMath Graph object.")

    # This computation is an overestimate if G is not regular, but it is fine for our purposes. 
    total_num_walks = G.order()*(max(G.degree())**k)
    
    if total_num_walks > MAXIMUM_SET_SIZE:
        raise MemoryError(
            f"Requested set of walks of size {total_num_walks} exceeds MAXIMUM_SET_SIZE={MAXIMUM_SET_SIZE}; reduce walk length"
        )

    # Initialize walks: start at every vertex
    walks = [[v] for v in G.vertices()]

    # Extend each walk by one step, k times
    for _ in range(k):
        new_walks = []
        for w in walks:
            last = w[-1]
            # For each neighbor of the last vertex, extend the walk
            for nbr in G.neighbors(last):
                new_walks.append(w + [nbr])
        walks = new_walks

    # Convert to set of tuples for uniqueness checking
    walk_set = set()
    for walk in walks:
        walk_set.add(tuple(walk))

    return walk_set


def bias_amp_func_using_walk(str, walks):
    """Compute bias amplification using a set of walks.

    For a bit vector s and set of walks W, computes:
        (1/|W|) * sum_{w in W} (-1)^(sum_{vertex in w} str[vertex])
    That is, for each walk, compute the sum of the walks corresponding 

    Args:
        str: A binary string or list
        walks: Set or iterable of walks (each walk is a sequence of vertex indices).

    Returns:
        float: Amplified bias value in [0, 1].

    Raises:
        IndexError: If a vertex index in a walk exceeds length of s.
    """

    #Note: This function is not used a lot but is nice to have for testing purposes. 

    amplified_bias = 0
    count = 0

    for walk in walks:
        # Sum the bits of s at positions in the walk
        substr = 0
        for vertex in walk:
            try:
                substr += str[vertex]
            except (IndexError, TypeError):
                raise TypeError(
                    f"Input string str needs to be a binary string/list of length >= {vertex}"
                )
        # Compute parity (mod 2)
        substr = substr % 2
        # Add ±1 based on parity
        amplified_bias += (-1) ** substr
        count += 1

    # Return normalized average
    return abs(amplified_bias / count) if count > 0 else 0.0


def vectorized_bias_amp_func_using_walk(str, walks):
    """
    Vectorized computation of bias amplification.
    Assumption: All walks in 'walks' have the same length k.
    Note that as a result this function does not handle any errors. 
    Adding a check for this would take too much time and defeat the purpose of being vectorized.

    Args:
        str: Numpy array
        walks: Numpy array of shape (num_walks, k)

    Returns:
        float: Amplified bias value in [0, 1].
    """

    # Replace vertex indices with their bit values from s
    # Resulting shape: (num_walks, k)
    walk_bits = str[walks]

    # Sum along the walk (axis 1) and take mod 2
    # Resulting shape: (num_walks,)
    parities = np.sum(walk_bits, axis=1) % 2

    # Map parity {0, 1} to sign {1, -1} using the identity (-1)^p = 1 - 2p
    signs = 1 - 2 * parities

    # Compute the absolute mean of the signs
    return np.abs(np.mean(signs))


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================


def test_biased_set_generation():
    """Test biased set generation with both 'all' and 'sample' modes.

    Verifies that generated vectors have correct Hamming weights within
    the specified range for both enumeration and sampling modes.

    Returns:
        bool: True if all generated vectors are valid, False otherwise.
    """
    try:
        # Pick a random float between 0.1 to 1
        eps = pyrandom.uniform(0.1, 1)
        # Pick a random integer from [10, 11, 12, 13]
        n = pyrandom.randint(10, 13)

        # Test 'all' mode
        low = math.floor(n * ((1 - eps) / 2))
        high = math.ceil(n * ((1 + eps) / 2))

        count = 0
        all_vecs = []
        for vec in generate_biased_set(eps, n, mode='all'):
            weight = sum(vec)
            assert low <= weight <= high, f"Weight {weight} outside range [{low}, {high}]"
            all_vecs.append(vec)
            count += 1
        
        # Gives subsets of [1, 2, ..., 10] of size 5
        subsets = Subsets(10, 5)
        for subset in subsets:
            vec = [0] * n
            for ind in subset:
                vec[ind-1] = 1
            vec = tuple(vec)
            assert tuple(vec) in all_vecs, f"Vector {vec} of weight 0.5 in range [{low}, {high}] not generated."

        print(f"✓ test_biased_set_generation (all mode): {count} vectors generated")

        # Test 'sample' mode
        count = 0
        for vec in generate_biased_set(eps, n, mode='sample', sample_size=100):
            weight = sum(vec)
            assert low <= weight <= high, f"Weight {weight} outside range [{low}, {high}]"
            count += 1

        print(f"✓ test_biased_set_generation (sample mode): {count} vectors sampled")
        return True
    except Exception as e:
        print(f"✗ test_biased_set_generation failed: {e}")
        return False


def test_random_walk_generation():
    """Test random walk generation on a small graph.

    Verifies that walks are generated with correct length and that the
    walk tracing is valid (each step connects adjacent vertices).

    Returns:
        bool: True if walks are valid, False otherwise.
    """
    try:

        # Create a small random regular test graph
        G = graphs.RandomRegular(5, 10)

        # Test single random walk
        k = pyrandom.randint(1, 15)
        walk = sample_one_random_walk(G, k)
        assert len(walk) == k + 1, f"Walk length should be {k + 1}, got {len(walk)}"
        assert (G.has_edge((walk[0], walk[1])) and G.has_edge((walk[1], walk[2])) 
                and G.has_edge((walk[2], walk[3]))), f"Walk produced {walk} is not valid in graph with edges {G.edges()}"

        print(f"✓ test_random_walk_generation: Generated walk of length {len(walk)}")
        return True
    except Exception as e:
        print(f"✗ test_random_walk_generation failed: {e}")
        return False


def test_all_walks_enumeration():
    """Test exhaustive enumeration of all walks.

    Verifies that all walks of a given length can be enumerated correctly
    starting from all vertices in the graph.

    Returns:
        bool: True if walk enumeration succeeds, False otherwise.
    """
    try:

        # Create a small random regular test graph with degree=5, n=10
        G = graphs.RandomRegular(5, 10)

        # Test all walks enumeration
        k = 4
        all_walks = get_all_walks(G, k)

        #Check for all walks by exhaustive enumeration
        for v in G:
            for n1 in G.neighbors(v):
                for n2 in G.neighbors(n1):
                    for n3 in G.neighbors(n2):
                        for n4 in G.neighbors(n3):
                            assert (v, n1, n2, n3, n4) in all_walks, f'Walk {(v, n1, n2, n3, n4)} should be in all_walks'


        print(f"✓ test_all_walks_enumeration: Enumerated {len(all_walks)} walks (k={k+1})")
        return True
    except Exception as e:
        print(f"✗ test_all_walks_enumeration failed: {e}")
        return False


def test_vectorized_bias_amp():
    """Test exhaustive enumeration of all walks.

    Verifies that the vectorized bias amplification function works correctly,
    by checking it against the non-vectorized/manual function. 

    Returns:
        bool: True if the outputs of the two functions match, False otherwise.
    """

    try:

        # Create a small random regular test graph with degree=5, n=10
        G = graphs.RandomRegular(5, 10)

        # Get all walks of length k
        k = pyrandom.randint(1, 6)
        all_walks = get_all_walks(G, k)
        all_walks = np.array(list(all_walks))  # Convert to numpy array for vectorized function

        # Pick a random string
        str = pyrandom.choices([0, 1], k=10)
        str = np.array(str)  # Convert to numpy array for vectorized function
        
        # Compare outputs of vectorized_bias_amp...() and bias_amp...()
        assert (vectorized_bias_amp_func_using_walk(str, all_walks) 
                == bias_amp_func_using_walk(str, all_walks)), f'vectorized_bias_amp_func failed on {G.edges()} for string {str}.'
        

        print(f"✓ vectorized_bias_amp: Matches output of non-vectorized bias_amp_func")
        return True

    except Exception as e:
        print(f"✗ test_vectorized_bias_amp failed: {e}")
        return False


def run_all_tests():
    """Run all testing functions and report results.

    Returns:
        bool: True if all tests pass, False if any test fails.
    """
    print("\n" + "=" * 70)
    print("Running Bias Amplification Tests")
    print("=" * 70)

    tests = [
        test_biased_set_generation,
        test_random_walk_generation,
        test_all_walks_enumeration,
        test_vectorized_bias_amp
    ]

    results = [test() for test in tests]

    passed = sum(results)
    total = len(results)

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70 + "\n")

    return all(results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run all tests
    run_all_tests()

    #Sample usage
    '''biased_set = generate_biased_set(0.5, 3, mode='all')
    for elem in biased_set:
        print(elem)
    G = graphs.CycleGraph(5)
    print(sample_one_random_walk(G, 4))'''

