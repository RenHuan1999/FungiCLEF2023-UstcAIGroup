import pickle
import os
import glob
import numpy as np
import torch
from collections import defaultdict

print(os.listdir("/ossfs/workspace/datasets/fungi_tta"))
root = "/ossfs/workspace/datasets/fungi_tta/"
dirs = os.listdir(root)

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
    del result
    print(len(list(result_final.keys())))
for k, v in result_final.items():
    result_final[k] = v / len(dirs)

submit_result = defaultdict(list)
for k, v in result_final.items():
    k = k.split("-")[1].split(".")[0]
    submit_result[k].append(v.tolist())

with open("/ossfs/workspace/datasets/" + "avg.pkl", "wb") as f:
    pickle.dump(submit_result, f)

# post process on ensamble results
less_cls = [18, 21, 47, 110, 135, 248, 260, 393, 485, 500, 521, 530, 624, 656, 668, 690, 760, 825, 918, 1006, 1010, 1041, 1055, 1065, 1087, 1187, 1304, 1372, 1376, 1429, 1489, 1510, 1545, 1580, 1581, 1599, 1601]
with open("/ossfs/workspace/datasets/" + "avg.pkl", "rb") as f:
    result_temp = pickle.load(f)
thresh = 9.8
max_result = dict()
filter_num = 0
diff_num = 0
less_count = 1
for k, v in result_temp.items():
    scores = v
    scores = np.vstack(scores)
    score = np.mean(scores, 0)
    torch_score = torch.tensor(score)
    max_score_torch, max_idx_torch = torch_score.topk(3)
    max_score_top2 = max_score_torch[1]
    max_idx_top2 = max_idx_torch[1]
    max_score_top3 = max_score_torch[2]
    max_idx_top3 = max_idx_torch[2]
    score_v2 = np.max(scores, 0)
    max_score = np.max(score)
    max_idx = np.argmax(score)
    max_score_v2 = np.max(score_v2)
    max_idx_v2 = np.argmax(score_v2)
    if max_idx_v2 in less_cls and max_idx not in less_cls and max_score_v2 > thresh-0.7:
        max_score = thresh + 1
        max_idx = max_idx_v2
        less_count += 1
    elif max_idx in less_cls and max_score > thresh - 0.7:
        max_score = thresh + 1.2
        less_count += 1
    elif max_idx_top2 in less_cls and max_idx not in less_cls and max_score_top2 > thresh - 0.7 and (max_score - max_score_top2) < 1.2:
        print(f"less cls={max_idx_top2}, score={max_score_top2}, max_score={max_score}")
        less_count += 1
        max_score = thresh + 1
        max_idx = max_idx_top2
    elif max_idx_top3 in less_cls and max_idx not in less_cls and max_score_top3 > thresh - 0.7 and (max_score - max_score_top3) < 1.2:
        print(f"so less cls={max_idx_top3}, score={max_score_top3}, max_score={max_score}")
        less_count += 1
        max_score = thresh + 1
        max_idx = max_idx_top3
    # if max_score <= thresh and max_idx != max_idx_v2 and max_score_v2 > 14:
    #     print(k, max_idx, max_idx_v2, max_score, max_score_v2)
    #     print(score[max_idx_v2], max(scores[:, max_idx]))
    #     diff_num += 1
    if max_score <= thresh and len(scores) > 1 and max_score_v2 > 15:
        print(k, max_idx_v2, max_score_v2)
        filter_num += 1
        # max_score = max(scores[:, max_idx])
        max_score = max_score_v2
        max_idx = max_idx_v2
    max_result[k] = (max_score, max_idx)
print(f"filter number {filter_num}")
print(f"less count top2 = {less_count}")
print(f"max mean diff num {diff_num}")

test_scores = []
for res in max_result.values():
    test_scores.append(res[0])
print(len(test_scores))
print(max(test_scores))
print(min(test_scores))
print(len(dirs))

other_num = 0
other_obid = set()
cls2other_count = defaultdict(int)
with open(os.path.join("/ossfs/workspace/datasets/", "submission.csv"), "w") as f:
    print("build submission")
    f.write("ObservationId,ClassId\n")
    for k, v in max_result.items():
        max_score, max_idx = v
        if max_score > thresh or (max_score > thresh - eta and max_idx in less_cls):
            f.write(f"{k},{max_idx}\n")
        # if (max_idx in less_cls and max_score > thresh):
        #     f.write(f"{k},{max_idx}\n")
        #     print("less {max_idx}, score={max_score}")
        else:
            other_num += 1
            cls2other_count[max_idx] += 1
            f.write(f"{k},{-1}\n")
            other_obid.add(k)
print(f"other number is {other_num}")
