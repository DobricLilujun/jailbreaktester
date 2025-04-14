import lib.ClassifierController as ClassifierController
from dotenv import load_dotenv
import os
import sys
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import readchar



def plot_res():

    with open(sys.argv[1], 'r') as file:
        res_to_plot = json.load(file)
       
        percentages = [1,3,5,10,15]
        print("####",str(res_to_plot.keys()))
        
        for key,values in res_to_plot.items():
            print(key)
            print(values)
            
            sorted_avg = list(v for _, v in sorted(values.items(), key=lambda item: item[0]))
            sorted_percentage = list(k for k, v in sorted(values.items(), key=lambda item: item[0]))
            print(sorted_percentage)
            print(sorted_avg)
            plt.plot(sorted_percentage, sorted_avg, label=key[:10])

        plt.legend()
        plt.show()

def compute_enveloppe(prompt_metrics):
    '''
    prompt_metrics contains one .dictionary per prompt with percentage as key and embeddings metrics as values
    '''

    env = {}

    for metrics in prompt_metrics:
        
        accumulated_diff = 1.0
        for percentage, vals in sorted(metrics.items(), key=lambda x:int(x[0])):
            
            if not int(percentage) in env.keys():
                env[int(percentage)] = []
            accumulated_diff = accumulated_diff*vals[1]
            env[int(percentage)].append(accumulated_diff)

            

    sorted_percentage = list(k for k, v in sorted( env.items(), key=lambda item: int(item[0])))
    sorted_min_enveloppe = list(np.percentile(v,10) for _, v in sorted(env.items(), key=lambda item: int(item[0])))
    sorted_max_enveloppe = list(np.percentile(v,90) for _, v in sorted(env.items(), key=lambda item: int(item[0])))
    sorted_mean_enveloppe = list(np.percentile(v,50) for _, v in sorted(env.items(), key=lambda item: int(item[0])))

    return (sorted_percentage,sorted_min_enveloppe,sorted_max_enveloppe,sorted_mean_enveloppe)
       


def plot_enveloppe_after_check_answer():
    #argv[1] file of metrics
    #argv[2] grountruth with prob success
    
    
        
    with open(sys.argv[2], 'r') as file:
        data = json.load(file)
        
        

        with open(sys.argv[1], 'r') as file:
            res_to_plot = json.load(file)

            enveloppe_data_jb = []
            enveloppe_data_nojb = []

            print("####",str(res_to_plot.keys()))
            count = 0
            for key,values in res_to_plot.items():
                count+=1
                #exit condition
                
                if data[key]["prob_success"]>0.99:
                    enveloppe_data_jb.append(values)
                elif data[key]["prob_success"] <0.01:
                    enveloppe_data_nojb.append(values)

            env_jb = compute_enveloppe(enveloppe_data_jb)
            
            env_nojb=compute_enveloppe(enveloppe_data_nojb)
            
            
            plt.plot(env_jb[0], env_jb[3], label="jailbreak")
            plt.plot(env_nojb[0], env_nojb[3], label="no_jailbreak")
           
            plt.fill_between(env_jb[0], env_jb[1], env_jb[2], alpha=0.2)
            plt.fill_between(env_nojb[0], env_nojb[1], env_nojb[2], alpha=0.2)
            
            print("#prompts: ",count)
            #plt.ylim(0.7,1)
            plt.legend()
            plt.show()



def plot_res_after_check_answer():
    #argv[1] file of metrics
    #argv[2] grountruth with prob success
    
    
        
    with open(sys.argv[2], 'r') as file:
        data = json.load(file)
        
        with open(sys.argv[1], 'r') as file:
            res_to_plot = json.load(file)
        
            print("####",str(res_to_plot.keys()))
            count = 0
            for key,values in res_to_plot.items():
                #exit condition
                if data[key]["prob_success"]<0.99:
                    continue
                count+=1
                print(key)
                print(values)
                
                #sorted_avg = list(v[0] for _, v in sorted(values.items(), key=lambda item: int(item[0])))
                
                #Median
                
                sorted_avg = list(v[2] for _, v in sorted(values.items(), key=lambda item: int(item[0])))
                
                sorted_percentage = list(int(k) for k, v in sorted(values.items(), key=lambda item: int(item[0])))
                print(sorted_percentage)
                print(sorted_avg)
                plt.plot(sorted_percentage, sorted_avg, label=key[:10])

            print("#prompts: ",count)
            plt.ylim(0.0,0.1)
            plt.legend()
            plt.show()


#Not used
def save_res_after_check_answer(out_file="plot.json"):
    #argv[1] file with perturbations
    #argv[2:] file with perturbations

    #just init with dummy argument (to avoid changing all the Jailbreaktestercode)
    p2d = ClassifierController.Pert2Detect(extra='--threshold 0.15 --smoothllm_num_copies 2 --smoothllm_pert_types ["RandomSwapPerturbation","RandomPatchPerturbation"] --smoothllm_pert_pct_min 5 --smoothllm_pert_pct_max 10')
        
    pattern = re.compile("_(\d+)_(\d+)")

    res_to_plot = {}


    #Getting all working prompts and save the output
    #Thre is no need to take the different outputs of working prompt as the original pormpt is still the same

    prompt_list = []
    original_out = {}
    for filename in sys.argv[1]:
        
        with open(filename, 'r') as file:
            data = json.load(file)

            for k,v in data.items:
                
                if test[0]["gt"] and not test[0]["original_prompt"] in prompt_list:

                    prompt_list.append(test[0]["original_prompt"])
                    original_out[test[0]["original_prompt"]] = test[0]["answer"]
                    

    print("count: ",len(prompt_list))

    for filename in sys.argv[2:]:
        print("Start analyzing "+filename)
        
        percentage = int(pattern.search(filename).group(1))
    
        count = 0

        with open(filename, 'r') as file:
            data = json.load(file)

            for test in data:
                
                #first prompt is the original prompts, we exclude unsuccessful attacks:
                

                prompt = test[0]["original_prompt"]
                if not prompt in prompt_list:
                    continue
                count+=1

                original = original_out[prompt]
                perturbations = []
                for i in range(1,len(test)):
                    perturbations.append(test[i]["answer"])
                
                sim = p2d.sim_matrix(original,perturbations)
                avg_values = np.array(
                    [sum(sim) / len(perturbations) for sim in sim]
                )

                #cosine moyen entre l'output orginal et les outputs à partir des prompts perturbés
                
                res_to_plot[prompt] = res_to_plot.get(prompt, {})
                res_to_plot[prompt][percentage] = (avg_values[0], np.percentile(avg_values, 50), np.percentile(avg_values, 75) - np.percentile(avg_values, 25))
                # res_to_plot[prompt][percentage] = np.percentile(avg_values, 50)
                # res_to_plot[prompt][percentage] = 
                

        print(count)

    with open(out_file, "w") as outfile:
        json.dump(res_to_plot, outfile)


def save_res(out_file="plot.json"):

    #just init with dummy argument (to avoid changing all the Jailbreaktestercode)
    p2d = ClassifierController.Pert2Detect(extra='--threshold 0.15 --smoothllm_num_copies 2 --smoothllm_pert_types ["RandomSwapPerturbation","RandomPatchPerturbation"] --smoothllm_pert_pct_min 5 --smoothllm_pert_pct_max 10')
        
    pattern = re.compile("_(\d+)_(\d+)")

    res_to_plot = {}


    #Getting all working prompts and save the output
    #Thre is no need to take the different outputs of working prompt as the original pormpt is still the same

    prompt_list = []
    original_out = {}
    for filename in sys.argv[1:]:
        
        with open(filename, 'r') as file:
            data = json.load(file)

            for test in data:
                
                #if test[0]["gt"] and not test[0]["original_prompt"] in prompt_list:

                prompt_list.append(test[0]["original_prompt"])
                original_out[test[0]["original_prompt"]] = test[0]["answer"]
                    

    print("count: ",len(prompt_list))

    for filename in sys.argv[1:]:
        print("Start analyzing "+filename)
        
        percentage = int(pattern.search(filename).group(1))
    
        count = 0

        with open(filename, 'r') as file:
            data = json.load(file)

            for test in data:
                
                #first prompt is the original prompts, we exclude unsuccessful attacks:
                

                prompt = test[0]["original_prompt"]
                #if not prompt in prompt_list:
                #    continue
                count+=1

                original = original_out[prompt]
                perturbations = []
                for i in range(1,len(test)):
                    perturbations.append(test[i]["answer"])
                
                sim = p2d.sim_matrix(original,perturbations)
                avg_values = np.array(
                    [sum(sim) / len(perturbations) for sim in sim]
                )

                #cosine moyen entre l'output orginal et les outputs à partir des prompts perturbés
                
                res_to_plot[prompt] = res_to_plot.get(prompt, {})
                # res_to_plot[prompt][percentage] = avg_values[0]
                # res_to_plot[prompt][percentage] = np.percentile(avg_values, 50)
                # res_to_plot[prompt][percentage] = np.percentile(avg_values, 75) - np.percentile(avg_values, 25)
                res_to_plot[prompt][percentage] = (avg_values[0], np.min(avg_values), np.max(avg_values), np.percentile(avg_values, 10), np.percentile(avg_values, 20), np.percentile(avg_values, 30), np.percentile(avg_values, 40), np.percentile(avg_values, 50),np.percentile(avg_values, 60), np.percentile(avg_values, 70), np.percentile(avg_values, 80),np.percentile(avg_values, 90))
                                                   

        print(count)

    with open(out_file, "w") as outfile:
        json.dump(res_to_plot, outfile)

   

    # for (original_prompt,perturbated_prompt_list) in :

    #     sim_matrix = p2d.sim_matrix(original_prompt,perturbed_prompt_list)
        
    #     #compute average similarity with original prompt

    #     # For each prompt (initial + perturbed ones), we compute the average similarity of its output against all others
    #     avg_values = np.array([sum(sim) / (len(generated_outs) - 1) for sim in similarities])
    #     print(original_prompt,str(avg_values[0]))

def check_answer(only_false = True):

    qna = {}
    
    with open(sys.argv[1], 'r') as file:
        data = json.load(file)
        for test in data["requests"]:

            #if not exists, we init the query and anwer dictionary for this question
            if not test["question"] in qna.keys():
                qna[test["question"]] = {}
                qna[test["question"]]["outputs"] = []
            qna[test["question"]]["outputs"].append((test["response"],test["jailbreak_successful"]))

        for k,v in qna.items():
            for i in range(len(v["outputs"])):

                answer,jailbreak = v["outputs"][i]
                if jailbreak:

                    os.system('cls||clear')
                #We only check if it says it is jailbeaked as we used simple pattern matching to exclude false one
                
                    print(k)
                    print("#############")
                    print(answer)
                    print("#############")
                    print("True (y/n)?")
                    rep = readchar.readkey()
                    #change_answer
                    if rep=="n":
                        v["outputs"][i] = (answer,False)                    


        for k in qna.keys():
            qna[k]["prob_success"] = 1.0*sum(1 for _ in qna[k]["outputs"] if _[1])/len(qna[k]["outputs"])

        with open(sys.argv[2], "w") as outfile:
            json.dump(qna, outfile, indent=4)


if __name__ == "__main__":
    #save_res()
    #plot_res_after_check_answer()
    plot_enveloppe_after_check_answer()

    #plot_res()
    #check_answer()
