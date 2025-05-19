import pandas as pd
import os
import platform
import time
import pandas as pd
import os
import readchar
import json

folder_path = "/home/snt/projects_lujun/jail/jailbreaktester/output/benchmark_dataset/benchmark_exp_results/negbleurtForest_processed_corrected"

file_name = "benchmark_llama_merged_RandomSwapPerturbation_25_with_only_pert2detect_formalized_23-0825_processed.jsonl"

# outputname = file_name.split(".jsonl")[0] + "_formalized_corrected_submission.jsonl"
outputname = file_name
joined_path = os.path.join(folder_path, file_name)
df = pd.read_json(joined_path, lines=True)


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


# if "manual_checking_label" not in df.columns:
#     df["manual_checking_label"] = None
# df["manual_checking_label"] = None

for i, row in df.iterrows():
    # if pd.notna(row["manual_checking_label"]):
    #     continue
    clear_screen()
    if "original_prompt" in file_name:
        print(f"\n[{i + 1}/{len(df)}] Response:\n{json.loads(row['responses'])[0]}")
        manual_checking_label = row["manual_checking_label"]
        manual_checking_label_name = (
            "Non Jail Break" if manual_checking_label == 0 else "Jail Break"
        )
        print(f"Original Manual Checking Label:\n{manual_checking_label_name}")
        print("Is it harmful? (y/n or q quit): ", end="", flush=True)
    else:
        print(f"\n[{i + 1}/{len(df)}] Response:\n{json.loads(row['responses'])[1]}")
        manual_checking_label = row["manual_checking_label"]
        manual_checking_label_name = (
            "Non Jail Break" if manual_checking_label == 0 else "Jail Break"
        )
        print(f"Original Manual Checking Label:\n{manual_checking_label_name}")
        print("Is it harmful? (y/n or q quit): ", end="", flush=True)

    while True:
        key = readchar.readchar().lower()
        if key in ["y", "n"]:
            df.at[i, "manual_checking_label"] = 1 if key == "y" else 0
            break
        elif key == "q":
            print("\nmanual checking quit")
            break
        else:
            print("\ninvalid chekc y/n/q: ", end="", flush=True)

    if key == "q":
        break

output_path = os.path.join(folder_path, outputname)
df.to_json(output_path, lines=True, orient="records", force_ascii=False)
print(f"\nThe labels file is saved: {output_path}")
