import json
from random import sample

from downstream_tasks import task_dataset







# # ====== IC-Hubert100 - Baseline ======
prompt_task = "IC"
model_date = "20220727_verbalizer"  # This means the number(code) of the model
sample_date = "20220727_verbalizer"  # This means the day you sampled
unit_model = "cpc100"




# # ====== IC-Hubert100 - Baseline ======
# prompt_task = "IC"
# model_date = "20220726_baseline"  # This means the number(code) of the model
# sample_date = "20220726_baseline"  # This means the day you sampled
# unit_model = "cpc100"











# # ====== KS-CPC100 - Verbalizer ======
# prompt_task = "KS"
# model_date = "20220720_verbalizer"  # This means the number(code) of the model
# sample_date = "20220720_verbalizer"  # This means the day you sampled
# unit_model = "cpc100"





# # # ====== KS-Hubert100 - Baseline ======

# prompt_task = "KS"
# model_date = "20220726_baseline"  # This means the number(code) of the model
# sample_date = "20220726_baseline"  # This means the day you sampled
# unit_model = "hubert100"


# prompt_task = "KS"
# model_date = "20220721_verbalizer_p01_t2"  # This means the number(code) of the model
# sample_date = "20220721_verbalizer_p01_t2"  # This means the day you sampled
# unit_model = "cpc100"



# # ====== KS-Hubert100 - Verbalizer ======
# prompt_task = "KS"
# model_date = "20220718_verbalizer"  # This means the number(code) of the model
# sample_date = "20220718_verbalizer"  # This means the day you sampled
# unit_model = "hubert100"





# # ====== IC-Hubert100 - Verbalizer ======
# prompt_task = "IC"
# model_date = "20220718_verbalizer"  # This means the number(code) of the model
# sample_date = "20220718_verbalizer"  # This means the day you sampled
# unit_model = "hubert100"








# # ====== KS-CPC100 - Baseline ======
# prompt_task = "KS"
# model_date = "20220718_baseline"  # This means the number(code) of the model
# sample_date = "20220718_baseline"  # This means the day you sampled
# unit_model = "cpc100"





# # ====== KS-CPC100 - Verbalizer ======
# prompt_task = "KS"
# model_date = "20220720_baseline"  # This means the number(code) of the model
# sample_date = "20220720_baseline"  # This means the day you sampled
# unit_model = "cpc100"








# # ====== IC-CPC100 - Baseline ======
# prompt_task = "IC"
# model_date = "20220718_baseline"  # This means the number(code) of the model
# sample_date = "20220718_baseline"  # This means the day you sampled
# unit_model = "cpc100"


# # ====== IC-CPC100 - Verbalizer ======
# prompt_task = "IC"
# model_date = "20220718_verbalizer"  # This means the number(code) of the model
# sample_date = "20220718_verbalizer"  # This means the day you sampled
# unit_model = "cpc100"





print("==>>>sample_date", sample_date)

n_correct = 0
with open(f"./samples/samples_{sample_date}/samples_{prompt_task}_{unit_model}_{model_date}.json") as f:
    # === read === #
    data = json.load(f)
    ids = data.keys()
    # === cal === #
    for id in ids:
        label = data[id]["label"]
        predict = data[id]["predict"]
        if label == predict:
            n_correct += 1
print(f"correct prediction number: {n_correct}")
print(f"total data number:{len(data)}")
print(f"acc: {n_correct / len(data)}")
