chunk_size=( 30 60 120 180 240 600)

for i in "${chunk_size[@]}"
do
    echo "Running pipeline with chunk_size: " $i
	python run_isax_experiments.py -input_file ../../notebooks/clusterbusters/aphelion_2019-11-15.txt \
    -cadence 1 -chunk_size $i -detrend_window 1800 -smooth_window 2 -min_cardinality 8 -max_cardinality 128 \
    -node_level_depth 16
done