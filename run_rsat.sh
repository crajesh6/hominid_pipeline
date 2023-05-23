#!/bin/bash

# rsat_ex="/packages/rsat/bin/rsat"
j_file="/home/chandana/ray_results/tune_hominid_pipeline-test/tune_hominid_5c0ee_00193_193_conv1_activation=relu,conv1_attention_pool_size=10,conv1_batchnorm=False,conv1_channel_weight=se,conv_2023-05-05_06-53-12/filters/filters_2_hits.jaspar"
mkdir /home/chandana/ray_results/tune_hominid_pipeline-test/tune_hominid_5c0ee_00193_193_conv1_activation=relu,conv1_attention_pool_size=10,conv1_batchnorm=False,conv1_channel_weight=se,conv_2023-05-05_06-53-12/filters/clustered_filters
results_dir="/home/chandana/ray_results/tune_hominid_pipeline-test/tune_hominid_5c0ee_00193_193_conv1_activation=relu,conv1_attention_pool_size=10,conv1_batchnorm=False,conv1_channel_weight=se,conv_2023-05-05_06-53-12/filters/clustered_filters"

rsat matrix-clustering -matrix rsat $j_file jaspar -o $results_dir/clusters -v 2

# rsat_command="${rsat_ex} matrix-clustering -matrix rsat ${j_file} -o ${results_dir} -v 2"

# singularity exec rsat.sif $rsat_ex matrix-clustering -matrix rsat $j_file -o $results_dir -v 2
# mkdir test
# singularity exec rsat.sif $rsat_ex matrix-clustering -matrix test test.transfac transfac -o test/ -v 2
# singularity exec rsat.sif $rsat_ex matrix-clustering -h
# singularity exec rsat.sif which rsat


# matrix-clustering  -v 1 -max_matrices 300 -matrix oct4_peak_motifs $RSAT/public_html/tmp/www-data/2023/05/15/matrix-clustering_2023-05-15.211608_AMPsjK/matrix-clustering_query_matrices.transfac transfac -hclust_method average -calc sum -title oct4_motifs_found_in_chen_2008_peak_sets -metric_build_tree Ncor -lth w 5 -lth cor 0.6 -lth Ncor 0.4 -quick -label_in_tree name -return json,heatmap -o $RSAT/public_html/tmp/www-data/2023/05/15/matrix-clustering_2023-05-15.211608_AMPsjK/matrix-clustering
# rsat matrix-clustering -matrix test test.transfac transfac -o temp -v 2

# singularity exec rsat.sif /packages/rsat/bin/rsat matrix-clustering -matrix test test.transfac transfac -o test/test -v 2

# which rsat
# which matrix-clustering


# singularity exec rsat.sif /packages/rsat/bin/rsat /packages/rsat/perl-scripts/matrix-clustering -matrix test test.transfac transfac -o test/test -v 2
# /bin/sh script.sh
