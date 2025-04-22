python main.py \
    --name 'test_llama' \
    --llm_model 'meta-llama/Llama-2-7b-chat-hf' \
    --data_set_type 'JailBreakBench Data Set' \
    --data_set_path '/home/snt/projects_lujun/jail/jailbreaktester/dataset/jailBreakbench_failed_and_successfull.json' \
    --classifier 'Pert2Detect' \
    --classifier_options '--threshold 0.15 \
    --smoothllm_num_copies 10 \
    --smoothllm_pert_types ["RandomInsertPerturbation"] \
    --smoothllm_pert_pct_min 1 \
    --smoothllm_pert_pct_max 1'


python main.py \
    --name 'test_llama' \
    --llm_model 'meta-llama/Llama-2-7b-chat-hf' \
    --data_set_type 'JailBreakBench Data Set' \
    --data_set_path '/home/snt/projects_lujun/jail/jailbreaktester/dataset/jailBreakbench_failed_and_successfull.json' \
    --classifier 'Pert2Detect' \
    --classifier_options '--threshold 0.15 \
    --smoothllm_num_copies 10 \
    --smoothllm_pert_types ["RandomInsertPerturbation"] \
    --smoothllm_pert_pct_min 3 \
    --smoothllm_pert_pct_max 3'


python main.py \
    --name 'test_llama' \
    --llm_model 'meta-llama/Llama-2-7b-chat-hf' \
    --data_set_type 'JailBreakBench Data Set' \
    --data_set_path '/home/snt/projects_lujun/jail/jailbreaktester/dataset/jailBreakbench_failed_and_successfull.json' \
    --classifier 'Pert2Detect' \
    --classifier_options '--threshold 0.15 \
    --smoothllm_num_copies 10 \
    --smoothllm_pert_types ["RandomPatchPerturbation"] \
    --smoothllm_pert_pct_min 1 \
    --smoothllm_pert_pct_max 1'

python main.py \
    --name 'test_llama' \
    --llm_model 'meta-llama/Llama-2-7b-chat-hf' \
    --data_set_type 'JailBreakBench Data Set' \
    --data_set_path '/home/snt/projects_lujun/jail/jailbreaktester/dataset/jailBreakbench_failed_and_successfull.json' \
    --classifier 'Pert2Detect' \
    --classifier_options '--threshold 0.15 \
    --smoothllm_num_copies 10 \
    --smoothllm_pert_types ["RandomPatchPerturbation"] \
    --smoothllm_pert_pct_min 3 \
    --smoothllm_pert_pct_max 3'



python main.py \
    --name 'test_llama' \
    --llm_model 'meta-llama/Llama-2-7b-chat-hf' \
    --data_set_type 'JailBreakBench Data Set' \
    --data_set_path '/home/snt/projects_lujun/jail/jailbreaktester/dataset/jailBreakbench_failed_and_successfull.json' \
    --classifier 'Pert2Detect' \
    --classifier_options '--threshold 0.15 \
    --smoothllm_num_copies 10 \
    --smoothllm_pert_types ["RandomPatchPerturbation"] \
    --smoothllm_pert_pct_min 25 \
    --smoothllm_pert_pct_max 25'