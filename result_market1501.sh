python2 eval/metric_learning_market1501.py './features/cuhk03'
python2 eval/JointBayesian/get_distance.py './features/cuhk03'
cp ${result_dir}/D_Euclidean.npy ./eval/Market-1501-baseline
cp ${result_dir}/D_JointBayesian.npy eval/Market-1501-baseline
matlab -nodisplay -r "run eval/Market-1501-baseline/baseline_evaluation;exit()"
