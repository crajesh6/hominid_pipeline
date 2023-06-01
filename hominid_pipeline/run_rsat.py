import sh


rsat_ex = "/packages/rsat/bin/rsat"
j_file="/home/chandana/projects/hominid_pipeline/temp/tune_hominid_5c0ee_00011_11_conv1_activation=relu,conv1_attention_pool_size=32,conv1_batchnorm=False,conv1_channel_weight=softconv_2023-05-04_16-39-12/filters/filters_2_hits.jaspar"
results_dir="/home/chandana/projects/hominid_pipeline/temp/tune_hominid_5c0ee_00011_11_conv1_activation=relu,conv1_attention_pool_size=32,conv1_batchnorm=False,conv1_channel_weight=softconv_2023-05-04_16-39-12/filters"
# rsat_command="${rsat_ex} matrix-clustering -matrix rsat ${j_file} -o ${results_dir} -v 2"

# singularity exec rsat.sif $rsat_ex matrix-clustering -matrix rsat $j_file -o $results_dir -v 2
# mkdir test
# singularity exec rsat.sif $rsat_ex matrix-clustering -matrix test test.transfac transfac -o test/ -v 2
# singularity exec rsat.sif $rsat_ex matrix-clustering -h
# singularity exec rsat.sif which rsat
# output_dir = "test/test"
# sh.singularity("exec", \
#     "rsat.sif", \
#     f"{rsat_ex}", \
#     "matrix-clustering", \
#     "-matrix", \
#     "test", \
#     "test.transfac transfac", \
#     f"-o {output_dir}", \
#     "-v 2"
#     )

# print(sh.singularity("exec", "rsat.sif", f"{rsat_ex}", "matrix-clustering", "-matrix", "test", "test.transfac", "transfac", "-o", "test/testp", "-v", "2", "-max_matrices", "300", "-hclust_method", "average", "-calc", "sum"))


print(sh.rsat("matrix-clustering", \
"-matrix test", \
"test.transfac", "transfac", \
"-o", "test/test", \
"-v", "2"))


# print(sh.echo("exec", \
#     "rsat.sif", \
#     f"{rsat_ex}", \
#     "matrix-clustering", \
#     "-matrix", \
#     "test", \
#     "test.transfac transfac", \
#     f"-o {output_dir}", \
#     "-v 2"
#     ))

print("hello!")


# print(sh.singularity("exec", \
# "rsat.sif", \
# f"{rsat_ex}", \
# "matrix-clustering", \
#  "-h"
# ))

# print(sh.echo("exec", \
# "rsat.sif", \
# f"{rsat_ex}", \
# "matrix-clustering", \
#  "-h"
# ))
