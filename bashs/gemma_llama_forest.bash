python negbleurtForestClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_gemma_merged_RandomInsertPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --model_name google/gemma-2-9b-it \
  --num 3 \
  --server_url http://0.0.0.0:8002/v1/chat/completions


python negbleurtForestClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_gemma_merged_RandomPatchPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --model_name google/gemma-2-9b-it \
  --num 3 \
  --server_url http://0.0.0.0:8002/v1/chat/completions


python negbleurtForestClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_gemma_merged_RandomSwapPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --model_name google/gemma-2-9b-it \
  --num 3 \
  --server_url http://0.0.0.0:8002/v1/chat/completions


python negbleurtForestClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_merged_with_original_prompt_gt_pert2detect_formalized_gemma.jsonl \
  --model_name google/gemma-2-9b-it \
  --num 3 \
  --server_url http://0.0.0.0:8002/v1/chat/completions


python negbleurtForestClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_llama_merged_RandomInsertPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num 3 \
  --server_url http://0.0.0.0:8000/v1/chat/completions


python negbleurtForestClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_llama_merged_RandomPatchPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num 3 \
  --server_url http://0.0.0.0:8000/v1/chat/completions


python negbleurtForestClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_llama_merged_RandomSwapPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num 3 \
  --server_url http://0.0.0.0:8000/v1/chat/completions


python negbleurtForestClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_merged_with_original_prompt_gt_pert2detect_formalized_llama.jsonl \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num 3 \
  --server_url http://0.0.0.0:8000/v1/chat/completions


