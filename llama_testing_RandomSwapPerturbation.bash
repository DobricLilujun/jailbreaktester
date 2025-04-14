python main.py \
    --name 'test_llama' \
    --llm_model 'meta-llama/Llama-2-7b-chat-hf' \
    --data_set_type 'JailBreakBench Data Set' \
    --data_set_path '/home/snt/projects_lujun/jail/jailbreaktester/dataset/jailBreakbench_failed_and_successfull.json' \
    --classifier 'Pert2Detect' \
    --classifier_options '--threshold 0.15 \
    --smoothllm_num_copies 10 \
    --smoothllm_pert_types ["RandomSwapPerturbation"] \
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
    --smoothllm_pert_types ["RandomSwapPerturbation"] \
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
    --smoothllm_pert_types ["RandomSwapPerturbation"] \
    --smoothllm_pert_pct_min 5 \
    --smoothllm_pert_pct_max 5'


python main.py \
    --name 'test_llama' \
    --llm_model 'meta-llama/Llama-2-7b-chat-hf' \
    --data_set_type 'JailBreakBench Data Set' \
    --data_set_path '/home/snt/projects_lujun/jail/jailbreaktester/dataset/jailBreakbench_failed_and_successfull.json' \
    --classifier 'Pert2Detect' \
    --classifier_options '--threshold 0.15 \
    --smoothllm_num_copies 10 \
    --smoothllm_pert_types ["RandomSwapPerturbation"] \
    --smoothllm_pert_pct_min 10 \
    --smoothllm_pert_pct_max 10'

python main.py \
    --name 'test_llama' \
    --llm_model 'meta-llama/Llama-2-7b-chat-hf' \
    --data_set_type 'JailBreakBench Data Set' \
    --data_set_path '/home/snt/projects_lujun/jail/jailbreaktester/dataset/jailBreakbench_failed_and_successfull.json' \
    --classifier 'Pert2Detect' \
    --classifier_options '--threshold 0.15 \
    --smoothllm_num_copies 10 \
    --smoothllm_pert_types ["RandomSwapPerturbation"] \
    --smoothllm_pert_pct_min 15 \
    --smoothllm_pert_pct_max 15'


python main.py \
    --name 'test_llama' \
    --llm_model 'meta-llama/Llama-2-7b-chat-hf' \
    --data_set_type 'JailBreakBench Data Set' \
    --data_set_path '/home/snt/projects_lujun/jail/jailbreaktester/dataset/jailBreakbench_failed_and_successfull.json' \
    --classifier 'Pert2Detect' \
    --classifier_options '--threshold 0.15 \
    --smoothllm_num_copies 10 \
    --smoothllm_pert_types ["RandomSwapPerturbation"] \
    --smoothllm_pert_pct_min 25 \
    --smoothllm_pert_pct_max 25'





