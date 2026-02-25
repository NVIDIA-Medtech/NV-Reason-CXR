import re
from dataclasses import dataclass


## 14 classes based on CheXpert
@dataclass(frozen=True)
class CX:
    Atelectasis: str = "Atelectasis"
    Cardiomegaly: str = "Cardiomegaly"
    Consolidation: str = "Consolidation"
    Edema: str = "Edema"
    Enlarged_Cardiomediastinum: str = "Enlarged Cardiomediastinum"
    Fracture: str = "Fracture"
    Lung_Lesion: str = "Lung Lesion"
    Lung_Opacity: str = "Lung Opacity"
    No_Finding: str = "No Finding"
    Pleural_Effusion: str = "Pleural Effusion"
    Pleural_Other: str = "Pleural Other"
    Pneumonia: str = "Pneumonia"
    Pneumothorax: str = "Pneumothorax"
    Support_Devices: str = "Support Devices"

    def get_list():
        return [
            CX.Atelectasis,
            CX.Cardiomegaly,
            CX.Consolidation,
            CX.Edema,
            CX.Enlarged_Cardiomediastinum,
            CX.Fracture,
            CX.Lung_Lesion,
            CX.Lung_Opacity,
            CX.No_Finding,
            CX.Pleural_Effusion,
            CX.Pleural_Other,
            CX.Pneumonia,
            CX.Pneumothorax,
            CX.Support_Devices,
        ]


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # print(f"accuracy_reward kwargs: {kwargs}  completions: {completions} solution: {solution}")

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    CX_list = set([e.lower() for e in CX.get_list()])

    for content, sol in zip(contents, solution):
        # print(f"content: {content}  sol: {sol}\n\n")
        gold_parsed = set([e.strip().lower() for e in sol.strip().split(",")])
        gold_parsed = gold_parsed & CX_list  # filter

        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            answer_parsed = match.group(1)
            answer_parsed = set([e.strip().lower() for e in answer_parsed.strip().split(",")])
            answer_parsed = answer_parsed & CX_list  # filter

            intersect = gold_parsed & answer_parsed
            union = gold_parsed | answer_parsed

            reward = float(len(intersect)) / len(union) if len(union) > 0 else 1.0  # if both gold and answer are empty, reward 1.0 (correct)

        else:
            reward = 0.0

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    # print(f"format_reward kwargs: {kwargs}  completions: {completions}")

    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]

    # debug
    i = 0
    for r, c in zip(rewards, completion_contents):
        if r == 0.0:
            print(f"format_reward wrong {i} of {rewards} completion_contents {len(c)}: {c}")
            print("\n\n")
        i = i + 1

    return rewards


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`."""

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def accuracy_reward_hard(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # print(f"accuracy_reward kwargs: {kwargs}  completions: {completions} solution: {solution}")

    f_val = format_reward(completions)
    t_val = tag_count_reward(completions)
    t_val = [0.0 if v < 1.0 else 1.0 for v in t_val]

    c_val = [f * t for f, t in zip(f_val, t_val)]
    # print(f"c_val: {c_val} for f_val: {f_val} and t_val: {t_val} completions: {completions}")
    if not any(c_val):
        return c_val  # return 0.0 for all completions if no format or tag count is correct

    r = accuracy_reward(completions, solution)

    rewards = [r * c for r, c in zip(r, c_val)]
    # print(f"rewards: {rewards} for r before: {r} and c_val: {c_val}")

    return rewards


def get_soft_overshort_punishment(completion_ids: list[list[int]], **kwargs) -> list[float]:
    """Reward function that penalizes short completions."""
    min_completion_len = 400
    soft_punish_cache = 400

    rewards = []
    for ids in completion_ids:
        completion_length = len(ids)
        if completion_length >= min_completion_len:
            rewards.append(0.0)
        else:
            rewards.append(max(-1, (completion_length - min_completion_len) / soft_punish_cache))

    return rewards
