import pandas as pd
import os
import platform
import time
import pandas as pd
import os
import readchar
import json
folder_path = "/home/lujun_li/projects/jailbreaktester/output/benchmark_dataset_llama_merged"

file_name = "merged_RandomInsertPerturbation_25_with_only_pert2detect.jsonl"

outputname= file_name.split(".jsonl")[0] + "_formalized.jsonl"
joined_path = os.path.join(folder_path, file_name)
df = pd.read_json(joined_path, lines=True)


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

if "manual_checking_label" not in df.columns:
    df["manual_checking_label"] = None

for i, row in df.iterrows():
    if pd.notna(row["manual_checking_label"]):
        continue 
    clear_screen()
    print(f"\n[{i + 1}/{len(df)}] Response:\n{json.loads(row['responses'])[1]}")
    print("是否有害？(y/n 或 q 退出): ", end="", flush=True)

    while True:
        key = readchar.readchar().lower()
        if key in ['y', 'n']:
            df.at[i, "manual_checking_label"] = 1 if key == 'y' else 0
            break
        elif key == 'q':
            print("\n手动退出。你可以稍后继续标注。")
            break
        else:
            print("\n无效输入，请输入 y/n/q: ", end="", flush=True)

    if key == 'q':
        break

# 保存进度
output_path = os.path.join(folder_path, outputname)
df.to_json(output_path, lines=True, orient="records", force_ascii=False)
print(f"\n当前标注结果已保存到：{output_path}")