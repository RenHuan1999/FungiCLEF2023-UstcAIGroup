import pickle
import os
import glob
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import entropy

root = "fungi_pkl_meta_2_bs18_epoch64_poison_trainval/"
dirs = os.listdir(root)
dirs_temp = []
for d in dirs:
    if '.' not in d:
        dirs_temp.append(d)
dirs = dirs_temp

# average results from different model
img_nums = 118676
print(dirs)
result_final = dict()
result_final_count = dict()
print(len(dirs))
for d in dirs:
    print(d)
    results = []
    result_dir = root + d
    result_paths = glob.glob(result_dir + "/*.pkl")
    for p in result_paths:
        with open(p, "rb") as f:
            results.append(pickle.load(f))
    result = results[0]
    for i in range(1, len(results)):
        result.update(results[i])
    del results
    # to numpy
    for k, v in result.items():
        v = np.array(v)
        if k not in result_final:
            result_final[k] = v
        else:
            result_final[k] += v
            # result_final[k] += v[:1604]
    del result
    print(len(list(result_final.keys())))
for k, v in result_final.items():
    result_final[k] = v / len(dirs)

submit_result = defaultdict(list)
for k, v in result_final.items():
    k = k.split("-")[1].split(".")[0]
    submit_result[k].append(v.tolist())

with open(root + "avg.pkl", "wb") as f:
    pickle.dump(submit_result, f)

# post process on ensamble results
less_cls = [18, 21, 47, 110, 135, 248, 260, 393, 485, 500, 521, 530, 624, 656, 668, 690, 760, 825, 918, 1006, 1010, 1041, 1055, 1065, 1087, 1187, 1304, 1372, 1376, 1429, 1489, 1510, 1545, 1580, 1581, 1599, 1601]
with open(root + "avg.pkl", "rb") as f:
    result_temp = pickle.load(f)

max_result = dict()
entropy_dict = dict()
for k, v in result_temp.items():
    scores = v
    scores = np.vstack(scores)
    score = np.mean(scores, 0)
    torch_score = torch.tensor(score)
    max_score = np.max(score)
    max_idx = np.argmax(score)

    max_result[k] = (max_score, max_idx)

    probs = torch.softmax(torch_score, dim=-1)
    entropy_dict[k] = entropy(probs)

eta = 0.7
other_num = 0
with open(os.path.join(root, "submission.csv"), "w") as f:
    print("build submission")
    # f.write("ObservationId,ClassId\n")
    f.write("observation_id,class_id\n")
    for k, v in max_result.items():
        max_score, max_idx = v
        entropy_k = entropy_dict[k]
        if entropy_k < 4 or (entropy_k < 6 and max_idx in less_cls):
            f.write(f"{k},{max_idx}\n")
        else:
            other_num += 1
            f.write(f"{k},{-1}\n")
print(f"other number is {other_num}")
