#!/bin/bash
dir="/home/chad/codebase/nsfg/example/simulated_dataset/SingleRobotRangeOnly"

for d in $dir/*/ ; do
    echo "$d"
    efg="${d}factor_graph.fg"
    echo "$efg"
    output_dir="${d}gtsam"
    echo "$output_dir"
    # the last three entries in the following command (please see src/external/gtsam/README.md for details)
    # incremental_step, artificial_prior_sigma, groud_truth_initialization
    /home/chad/codebase/nsfg/src/external/gtsam/build/gtsam_solution "$efg" "$output_dir" 1 -1 0
done

