from lps_graph import LPSGraph
from sets_and_walks_helpers import *

# 1. Construct a (5+1)-regular LPS graph
lps = LPSGraph(p=5, q=13)
G = lps.get_graph()
expansion = lps.expansion

# 3. Relabel vertices from elements of PGL(2, q) to integers
# This is necessary to be able to index the string using vertices
G.relabel() 

# 2. Generate biased set
S = generate_biased_set(eps=expansion, n=G.order(), sample_size=10000)

# 3. Sample a subset of 100 walks of length 10
walk_subset = get_random_subset_of_walks(G, k=10, set_size=100)

# 4. Calculate bias
avg_amplified_bias = 0
for biased_vec in S:
    result = bias_amp_func_using_walk(biased_vec, walk_subset)
    avg_amplified_bias += result

avg_amplified_bias = avg_amplified_bias / 10000
print(f"Amplified Bias: {avg_amplified_bias}")