from lps_graph import LPSGraph
from sets_and_walks_helpers import generate_biased_set, get_all_walks, vectorized_bias_amp_func_using_walk
import random as pyrandom
import math
import time
from pathlib import Path
import numpy as np
from sage.all import graphs


def _write_output(text: str, output_file: str = None):
    """Helper function to write output to console and/or file.

    Args:
        text: Text to output
        output_file: Optional filename to write to
    """
    print(text, end='')
    output_folder = Path('results')
    output_folder.mkdir(parents=True, exist_ok=True)
    if output_file:
        with open('results/'+output_file, 'a') as f:
            f.write(text)


def run_bias_sampling_experiment(graph, expansion_ratio: float, walk_length: int,
                               biased_set_size: int, subsample_ratios: list[float],
                               output_file: str = None):
    """Run bias amplification experiment comparing full vs subsampled walk sets.

    This function investigates how subsampling the set of all walks affects bias
    amplification for vectors in a biased set. It compares the bias amplification
    using all walks vs using subsampled walk sets of different sizes.

    Args:
        graph: SageMath Graph object to run experiments on
        expansion_ratio: Spectral expansion ratio of the graph (λ₂/d)
        walk_length: Length of random walks to generate (k)
        biased_set_size: Number of biased vectors to sample
        subsample_ratios: List of ratios for subsampling (e.g., [0.5] for sqrt(|W|) walks)
        output_file: Optional filename to write results to (if None, prints to console)

    Returns:
        dict: Results containing max/average bias amplification and differences
    """

    # Prepare graph by relabeling vertices
    num_vertices = graph.order()
    graph.relabel()

    # Generate all walks of specified length
    all_walks = get_all_walks(graph, walk_length)
    total_walks = len(all_walks)

    # Create subsampled walk sets for each ratio
    subsampled_walk_sets = []
    for ratio in subsample_ratios:
        subsample_size = int(math.pow(total_walks, ratio))
        subsample = pyrandom.sample(list(all_walks), subsample_size)
        subsampled_walk_sets.append(subsample)

    # Generate biased set of vectors
    biased_set = generate_biased_set(0.9*expansion_ratio, num_vertices,
                                    sample_size=biased_set_size)

    # Initialize result tracking
    num_subsamples = len(subsample_ratios)
    results = {
        'max_bias_full': 0.0,
        'max_bias_subsamples': [0.0] * num_subsamples,
        'max_bias_differences': [0.0] * num_subsamples,
        'avg_bias_full': 0.0,
        'avg_bias_subsamples': [0.0] * num_subsamples,
        'avg_bias_differences': [0.0] * num_subsamples
    }

    # Convert all walks to numpy array for vectorized processing
    all_walks = np.array(list(all_walks))
    subsampled_walk_sets = [np.array(list(subsample)) for subsample in subsampled_walk_sets]
    start_time = time.time()
    # Process each biased vector
    for biased_vector in biased_set:
        np_biased_vector = np.array(biased_vector)

        # Compute bias amplification using all walks
        full_bias = vectorized_bias_amp_func_using_walk(np_biased_vector, all_walks)
        results['avg_bias_full'] += full_bias
        results['max_bias_full'] = max(results['max_bias_full'], full_bias)

        # Compute bias amplification for each subsampled set
        for i, subsample_walks in enumerate(subsampled_walk_sets):
            subsample_bias = vectorized_bias_amp_func_using_walk(np_biased_vector, subsample_walks)
            bias_difference = abs(subsample_bias - full_bias)

            # Update maximums
            results['max_bias_subsamples'][i] = max(results['max_bias_subsamples'][i], subsample_bias)
            results['max_bias_differences'][i] = max(results['max_bias_differences'][i], bias_difference)

            # Accumulate for averages
            results['avg_bias_subsamples'][i] += subsample_bias
            results['avg_bias_differences'][i] += bias_difference

    # Compute averages
    for i in range(num_subsamples):
        results['avg_bias_subsamples'][i] /= biased_set_size
        results['avg_bias_differences'][i] /= biased_set_size
    results['avg_bias_full'] /= biased_set_size

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print formatted results
    _write_output(f"\n{'='*60}\n", output_file)
    _write_output(f"Bias Amplification Experiment Results\n", output_file)
    _write_output(f"Experiment completed in {elapsed_time:.2f} seconds.\n", output_file)
    _write_output(f"{'='*60}\n", output_file)
    _write_output(f"Graph: {num_vertices} vertices, expansion ratio = {expansion_ratio:.4f}\n", output_file)
    _write_output(f"Walk length: {walk_length}, Total walks: {total_walks}\n", output_file)
    _write_output(f"Biased set size: {biased_set_size}\n", output_file)
    _write_output(f"Subsample ratios: {subsample_ratios}\n", output_file)
    _write_output(f"{'='*60}\n", output_file)

    _write_output(f"\nMaximum Bias Amplification:\n", output_file)
    _write_output(f"  Full walk set:     {results['max_bias_full']:.6f}\n", output_file)
    for i, ratio in enumerate(subsample_ratios):
        subsample_size = int(math.pow(total_walks, ratio))
        _write_output(f"  Subsample {ratio:.2f} ({subsample_size:>6} walks): {results['max_bias_subsamples'][i]:.6f}\n", output_file)

    _write_output(f"\nAverage Bias Amplification:\n", output_file)
    _write_output(f"  Full walk set:     {results['avg_bias_full']:.6f}\n", output_file)
    for i, ratio in enumerate(subsample_ratios):
        subsample_size = int(math.pow(total_walks, ratio))
        _write_output(f"  Subsample {ratio:.2f} ({subsample_size:>6} walks): {results['avg_bias_subsamples'][i]:.6f}\n", output_file)

    _write_output(f"\nAverage Bias Differences (Subsample - Full):\n", output_file)
    for i, ratio in enumerate(subsample_ratios):
        subsample_size = int(math.pow(total_walks, ratio))
        _write_output(f"  Subsample {ratio:.2f} ({subsample_size:>6} walks): {results['avg_bias_differences'][i]:.6f}\n", output_file)

    _write_output(f"\nTheoretical bound (λ^{walk_length/2}): {expansion_ratio**(walk_length/2):.6f}\n", output_file)
    _write_output(f"{'='*60}\n", output_file)
    
    return results


# ============================================================================
# EXPERIMENTAL RUNS
# ============================================================================

if __name__ == "__main__":
    
    # Petersen graph experiments
    petersen_graph = graphs.PetersenGraph()
    petersen_expansion = 2.0/3.0  # Known expansion ratio for Petersen graph
    run_bias_sampling_experiment(petersen_graph, petersen_expansion, 1, 100000, [0.75, 0.5, 0.25], output_file="petersen_results.txt")
    run_bias_sampling_experiment(petersen_graph, petersen_expansion, 3, 10000, [0.75, 0.5, 0.25], output_file="petersen_results.txt")

    # LPS graph experiments
    lps_13_5_obj = LPSGraph(p=13, q=5, silent=True)
    lps_13_5 = lps_13_5_obj.get_graph()
    run_bias_sampling_experiment(lps_13_5, lps_13_5_obj.expansion, 1, 100000, [0.5, 0.25], output_file="lps_13_5_results.txt")
    run_bias_sampling_experiment(lps_13_5, lps_13_5_obj.expansion, 2, 10000, [0.5], output_file="lps_13_5_results.txt")
    run_bias_sampling_experiment(lps_13_5, lps_13_5_obj.expansion, 3, 10000, [0.5], output_file="lps_13_5_results.txt")

    lps_5_13_obj = LPSGraph(p=5, q=13, silent=True)
    lps_5_13 = lps_5_13_obj.get_graph()
    run_bias_sampling_experiment(lps_5_13, lps_5_13_obj.expansion, 1, 100000, [0.5], output_file="lps_5_13_results.txt")
    #Takes around 20 minutes
    #run_bias_sampling_experiment(lps_5_13, lps_5_13_obj.expansion, 4, 35000, [0.5], output_file="lps_5_13_results.txt")
