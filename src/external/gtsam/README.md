# 1.write your setting file YOUR_FILE
factor_graph_path [your blank, str]

output_dir [your blank, str]

incremental_step [your blank, integer]

artificial_prior_sigma [your blank, float, negative means no artificial prior.]

gt_init [your blank, 1 or 0, 0 means the solver is initialized with groudtruth values.]
# 2.execute with your setting file
./gtsam_solution YOUR_FILE
