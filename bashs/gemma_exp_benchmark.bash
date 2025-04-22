python jailGuardClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_gemma_merged_RandomInsertPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --llm_model google/gemma-2-9b-it \
  --cls_name JailGuardUpdated

python jailGuardClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_gemma_merged_RandomPatchPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --llm_model google/gemma-2-9b-it \
  --cls_name JailGuardUpdated

python jailGuardClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_gemma_merged_RandomSwapPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --llm_model google/gemma-2-9b-it \
  --cls_name JailGuardUpdated

python jailGuardClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_merged_with_original_prompt_gt_pert2detect_formalized_gemma.jsonl \
  --llm_model google/gemma-2-9b-it \
  --cls_name JailGuardUpdated


python smoothLLMClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_gemma_merged_RandomInsertPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --llm_model google/gemma-2-9b-it \
  --cls_name SmoothLLM

python smoothLLMClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_gemma_merged_RandomPatchPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --llm_model google/gemma-2-9b-it \
  --cls_name SmoothLLM

python smoothLLMClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_gemma_merged_RandomSwapPerturbation_25_with_only_pert2detect_formalized.jsonl \
  --llm_model google/gemma-2-9b-it \
  --cls_name SmoothLLM

python smoothLLMClassifier.py \
  --input_file_path output/benchmark_dataset/manual_checked_gt/benchmark_merged_with_original_prompt_gt_pert2detect_formalized_gemma.jsonl \
  --llm_model google/gemma-2-9b-it \
  --cls_name SmoothLLM
