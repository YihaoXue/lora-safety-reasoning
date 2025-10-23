import json
import os
import re
import json
import os
import os
import json
import argparse
import argparse

import re

parser = argparse.ArgumentParser()
parser.add_argument('--directory_path', type=str, required=True, help='The directory path to be checked')
parser.add_argument('--task', type=str, required=True)

args = parser.parse_args()
directory_path = args.directory_path
task = args.task


def remove_boxed(s):
    if "\\fbox{" in s:  
        left = "\\fbox{"
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]

    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def unwrap_letter(s):
    """Extract the actual letter (A-Z) from wrapped strings like '(A)' or ' A '."""
    s = s.strip().upper()
    match = re.search(r"[A-Z]", s)
    return match.group(0) if match else None

def extract_answer(text: str) -> str:
    """Extract a multiple-choice letter (A–D), case-sensitive, from model output."""

    valid_choices = {"A", "B", "C", "D"}

    # 1. Boxed answer
    boxed_expr = last_boxed_only_string(text)
    if boxed_expr:
        raw = remove_boxed(boxed_expr)
        ans = unwrap_letter(raw)
        if ans in valid_choices:
            return ans

    # 2. Answer: X or Final Answer: (X)
    match = re.findall(r"(?:Final Answer|Answer(?: is)?):?\s*\(?([A-Z])\)?", text, re.IGNORECASE)
    if match and match[-1] in valid_choices:
        return match[-1]

    # 3. The answer is X
    match = re.findall(r"The answer is\s+\(?([A-Z])\)?", text, re.IGNORECASE)
    if match and match[-1] in valid_choices:
        return match[-1]

    # 4. The correct answer is X
    match = re.findall(r"The correct answer is\s+\(?([A-Z])\)?", text, re.IGNORECASE)
    if match and match[-1] in valid_choices:
        return match[-1]

    # 5. The final answer is X
    match = re.findall(r"The final answer is\s+\(?([A-Z])\)?", text, re.IGNORECASE)
    if match and match[-1] in valid_choices:
        return match[-1]

    return None


output_data = []

index = 0
problematic_batch = []

for filename in os.listdir(directory_path):
    if filename.startswith(f"samples_{task}"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                problem = data['doc']['Question']
                answer = unwrap_letter(data['target'])
                resp = data['resps'][0][0]
                
                resp_answer = extract_answer(resp)
                
                if resp_answer is not None: 
                    if resp_answer == answer:
                        exact_match = 1
                    else:
                        exact_match = 0
                        problematic_batch.append((index, problem, resp_answer, answer))
                else:
                    exact_match = 0
                
                output_data.append({
                    'index': index,
                    'problem': problem,
                    'answer': answer,
                    'resp': resp,
                    'resp_answer': resp_answer,
                    'exact_match': exact_match,
                })
                index += 1
            
                # if index == 2:
                #     raise NotImplementedError

problematic_output_data = [
    {
        'index': item['index'],
        'problem': item['problem'],
        'resp': item['resp'],
        'resp_answer': item['resp_answer'],
        'answer': item['answer'],
        'exact_match': item['exact_match']
    }
    for item in output_data if item['exact_match'] == 0
]

problematic_file_path = directory_path + f'/problematic_instances_{task}.json'
with open(problematic_file_path, 'w', encoding='utf-8') as problematic_file:
    json.dump(problematic_output_data, problematic_file, ensure_ascii=False, indent=4)


exact_match_sum = sum(item['exact_match'] for item in output_data)
none_resp_answer_sum = sum(1 for item in output_data if item['resp_answer'] is None)

print("The number of none resp_answer is ", none_resp_answer_sum)
print("The accuracy of exact_match is ", exact_match_sum / len(output_data))

output_file_path = directory_path + f'/answer_check_result_{task}.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_data, output_file, ensure_ascii=False, indent=4)

summary_results = {
    "none_resp_answer_count": none_resp_answer_sum,
    "exact_match_accuracy": exact_match_sum / len(output_data)
}

summary_file_path = directory_path + f'/summary_results_{task}.json'
with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
    json.dump(summary_results, summary_file, ensure_ascii=False, indent=4)