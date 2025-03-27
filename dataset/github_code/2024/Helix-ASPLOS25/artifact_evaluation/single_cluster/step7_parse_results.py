from typing import Dict, Tuple, List, Optional

import sys


def parse_result(file_name: str, warm_up_time: Optional[float], finish_time: Optional[float]):
    # read the logs
    with open(file_name, "r") as log_file:
        lines = log_file.readlines()
    print(f"{file_name} (excluding first {warm_up_time}s as warm up)")

    # set warm up time and finish time
    if warm_up_time is None:
        warm_up_time = 0
    if finish_time is None:
        finish_time = 99999

    # parse the lines
    # query_uid -> prompt out time
    prompt_unmatched_dict: Dict[int, float] = {}
    # query_uid -> (prompt latency, arrive time)
    prompt_matched_dict: Dict[int, Tuple[float, float]] = {}
    # (query_uid, context_len) -> out time
    decode_unmatched_dict: Dict[Tuple[int, int], float] = {}
    # (query_uid, context_len) -> (decode latency, arrive time)
    decode_matched_dict: Dict[Tuple[int, int], Tuple[float, float]] = {}
    decode_arrival_list: List[float] = []  # list of time
    for line in lines:
        event_time, query_idx, in_out, iter_type, context_len, processed_len = eval(line)

        # latency analysis
        if iter_type == "prompt":
            if in_out == "out":
                assert query_idx not in prompt_unmatched_dict
                prompt_unmatched_dict[query_idx] = event_time
            elif in_out == "in":
                assert query_idx in prompt_unmatched_dict
                prompt_out_time = prompt_unmatched_dict[query_idx]
                del prompt_unmatched_dict[query_idx]
                assert query_idx not in prompt_matched_dict
                prompt_matched_dict[query_idx] = (event_time - prompt_out_time, event_time)
            else:
                assert False
        elif iter_type == "decode":
            if in_out == "out":
                decode_key = (query_idx, context_len)
                assert decode_key not in decode_unmatched_dict
                decode_unmatched_dict[decode_key] = event_time
            elif in_out == "in":
                decode_key = (query_idx, context_len)
                assert decode_key in decode_unmatched_dict
                decode_out_time = decode_unmatched_dict[decode_key]
                del decode_unmatched_dict[decode_key]
                assert decode_key not in decode_matched_dict
                decode_matched_dict[decode_key] = (event_time - decode_out_time, event_time)
            else:
                assert False
        else:
            assert False

        # throughput analysis
        if iter_type == "decode" and in_out == "in":
            decode_arrival_list.append(event_time)

    # save latency dicts for plotting
    prompt_latency_list, decode_latency_list = [], []

    # calculate avg prompt latency and avg decode latency
    decode_arrival_time = []
    sum_prompt_latency, sum_decode_latency = 0, 0
    valid_prompt_count, valid_decode_count = 0, 0
    for prompt_latency, arrival_time in prompt_matched_dict.values():
        if warm_up_time <= arrival_time <= finish_time:
            sum_prompt_latency += prompt_latency
            valid_prompt_count += 1
            prompt_latency_list.append(prompt_latency)
    for decode_latency, arrival_time in decode_matched_dict.values():
        if warm_up_time <= arrival_time <= finish_time:
            sum_decode_latency += decode_latency
            valid_decode_count += 1
            decode_latency_list.append(decode_latency)
            decode_arrival_time.append(arrival_time)

    # print latency
    prompt_latency_list.sort()
    print(f"Prompt latency:")
    print(f"Latency 5th percentile: {prompt_latency_list[int(len(prompt_latency_list) * 0.05)]:.2f} s")
    print(f"Latency 25th percentile: {prompt_latency_list[int(len(prompt_latency_list) * 0.25)]:.2f} s")
    print(f"Latency 50th percentile: {prompt_latency_list[int(len(prompt_latency_list) * 0.5)]:.2f} s")
    print(f"Latency 75th percentile: {prompt_latency_list[int(len(prompt_latency_list) * 0.75)]:.2f} s")
    print(f"Latency 95th percentile: {prompt_latency_list[int(len(prompt_latency_list) * 0.95)]:.2f} s")
    decode_latency_list.sort()
    print(f"Decode latency:")
    print(f"Latency 5th percentile: {decode_latency_list[int(len(decode_latency_list) * 0.05)]:.2f} s")
    print(f"Latency 25th percentile: {decode_latency_list[int(len(decode_latency_list) * 0.25)]:.2f} s")
    print(f"Latency 50th percentile: {decode_latency_list[int(len(decode_latency_list) * 0.5)]:.2f} s")
    print(f"Latency 75th percentile: {decode_latency_list[int(len(decode_latency_list) * 0.75)]:.2f} s")
    print(f"Latency 95th percentile: {decode_latency_list[int(len(decode_latency_list) * 0.95)]:.2f} s")

    # calculate throughput
    print(f"Summary:")
    valid_decode_arrival_list = [time for time in decode_arrival_list if warm_up_time <= time <= finish_time]
    valid_decode_throughput = len(valid_decode_arrival_list) / (valid_decode_arrival_list[-1] -
                                                                valid_decode_arrival_list[0] + 1e-6)

    print(f"Avg prompt latency: {sum_prompt_latency / valid_prompt_count:.3f}s")
    print(f"Avg decode latency: {sum_decode_latency / valid_decode_count:.3f}s")
    print(f"Throughput: {valid_decode_throughput:.1f} Tokens/s")


def get_latency_list(file_name: str, warm_up_time: Optional[float], finish_time: Optional[float]):
    # read the logs
    with open(file_name, "r") as log_file:
        lines = log_file.readlines()
    print(f"{file_name} (excluding first {warm_up_time}s as warm up)")

    # set warm up time and finish time
    if warm_up_time is None:
        warm_up_time = 0
    if finish_time is None:
        finish_time = 99999

    # parse the lines
    # query_uid -> prompt out time
    prompt_unmatched_dict: Dict[int, float] = {}
    # query_uid -> (prompt latency, arrive time)
    prompt_matched_dict: Dict[int, Tuple[float, float]] = {}
    # (query_uid, context_len) -> out time
    decode_unmatched_dict: Dict[Tuple[int, int], float] = {}
    # (query_uid, context_len) -> (decode latency, arrive time)
    decode_matched_dict: Dict[Tuple[int, int], Tuple[float, float]] = {}
    decode_arrival_list: List[float] = []  # list of time
    for line in lines:
        event_time, query_idx, in_out, iter_type, context_len, processed_len = eval(line)

        # latency analysis
        if iter_type == "prompt":
            if in_out == "out":
                assert query_idx not in prompt_unmatched_dict
                prompt_unmatched_dict[query_idx] = event_time
            elif in_out == "in":
                assert query_idx in prompt_unmatched_dict
                prompt_out_time = prompt_unmatched_dict[query_idx]
                del prompt_unmatched_dict[query_idx]
                assert query_idx not in prompt_matched_dict
                prompt_matched_dict[query_idx] = (event_time - prompt_out_time, event_time)
            else:
                assert False
        elif iter_type == "decode":
            if in_out == "out":
                decode_key = (query_idx, context_len)
                assert decode_key not in decode_unmatched_dict
                decode_unmatched_dict[decode_key] = event_time
            elif in_out == "in":
                decode_key = (query_idx, context_len)
                assert decode_key in decode_unmatched_dict
                decode_out_time = decode_unmatched_dict[decode_key]
                del decode_unmatched_dict[decode_key]
                assert decode_key not in decode_matched_dict
                decode_matched_dict[decode_key] = (event_time - decode_out_time, event_time)
            else:
                assert False
        else:
            assert False

        # throughput analysis
        if iter_type == "decode" and in_out == "in":
            decode_arrival_list.append(event_time)

    # save latency dicts for plotting
    prompt_latency_list, decode_latency_list = [], []

    # calculate avg prompt latency and avg decode latency
    decode_arrival_time = []
    sum_prompt_latency, sum_decode_latency = 0, 0
    valid_prompt_count, valid_decode_count = 0, 0
    for prompt_latency, arrival_time in prompt_matched_dict.values():
        if warm_up_time <= arrival_time <= finish_time:
            sum_prompt_latency += prompt_latency
            valid_prompt_count += 1
            prompt_latency_list.append(prompt_latency)
    for decode_latency, arrival_time in decode_matched_dict.values():
        if warm_up_time <= arrival_time <= finish_time:
            sum_decode_latency += decode_latency
            valid_decode_count += 1
            decode_latency_list.append(decode_latency)
            decode_arrival_time.append(arrival_time)
    return prompt_latency_list, decode_latency_list


def aggregate_latency(file_names: List[str], warm_up_time: Optional[float], finish_time: Optional[float]):
    all_prompt_latency_list, all_decode_latency_list = [], []
    for file_name in file_names:
        prompt_latency_list, decode_latency_list = get_latency_list(file_name, warm_up_time, finish_time)
        all_prompt_latency_list.extend(prompt_latency_list)
        all_decode_latency_list.extend(decode_latency_list)
    all_prompt_latency_list.sort()
    all_decode_latency_list.sort()
    print(f"Prompt latency:")
    print(f"Latency 5th percentile: {all_prompt_latency_list[int(len(all_prompt_latency_list) * 0.05)]:.2f} s")
    print(f"Latency 25th percentile: {all_prompt_latency_list[int(len(all_prompt_latency_list) * 0.25)]:.2f} s")
    print(f"Latency 50th percentile: {all_prompt_latency_list[int(len(all_prompt_latency_list) * 0.5)]:.2f} s")
    print(f"Latency 75th percentile: {all_prompt_latency_list[int(len(all_prompt_latency_list) * 0.75)]:.2f} s")
    print(f"Latency 95th percentile: {all_prompt_latency_list[int(len(all_prompt_latency_list) * 0.95)]:.2f} s")
    print(f"Decode latency:")
    print(f"Latency 5th percentile: {all_decode_latency_list[int(len(all_decode_latency_list) * 0.05)]:.2f} s")
    print(f"Latency 25th percentile: {all_decode_latency_list[int(len(all_decode_latency_list) * 0.25)]:.2f} s")
    print(f"Latency 50th percentile: {all_decode_latency_list[int(len(all_decode_latency_list) * 0.5)]:.2f} s")
    print(f"Latency 75th percentile: {all_decode_latency_list[int(len(all_decode_latency_list) * 0.75)]:.2f} s")
    print(f"Latency 95th percentile: {all_decode_latency_list[int(len(all_decode_latency_list) * 0.95)]:.2f} s")


def main():
    # parse arguments
    assert len(sys.argv) == 4, f"Usage: python {sys.argv[0]} <method> <llama30b/llama70b> <online/offline>"
    method, model_name, serving_mode = sys.argv[1], sys.argv[2], sys.argv[3]
    assert model_name in ["llama30b", "llama70b"], f"Invalid model name: {model_name}"
    assert serving_mode in ["online", "offline"], f"Invalid serving mode: {serving_mode}"

    if model_name == "llama30b" and serving_mode == "offline" and method == "helix_a100":
        parse_result("./real_llama30b/helix_offline/a100/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "offline" and method == "helix_l4":
        parse_result("./real_llama30b/helix_offline/l4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "offline" and method == "helix_t4":
        parse_result("./real_llama30b/helix_offline/t4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "online" and method == "helix_a100":
        parse_result("./real_llama30b/helix_online/a100/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "online" and method == "helix_l4":
        parse_result("./real_llama30b/helix_online/l4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "online" and method == "helix_t4":
        parse_result("./real_llama30b/helix_online/t4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "offline" and method == "swarm":
        parse_result("./real_llama30b/swarm_offline/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "online" and method == "swarm":
        parse_result("./real_llama30b/swarm_online/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "offline" and method == "separate_a100":
        parse_result("./real_llama30b/separate_offline/a100/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "offline" and method == "separate_l4":
        parse_result("./real_llama30b/separate_offline/l4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "offline" and method == "separate_t4":
        parse_result("./real_llama30b/separate_offline/t4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "online" and method == "separate_a100":
        parse_result("./real_llama30b/separate_online/a100/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "online" and method == "separate_l4":
        parse_result("./real_llama30b/separate_online/l4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "online" and method == "separate_t4":
        parse_result("./real_llama30b/separate_online/t4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama70b" and serving_mode == "offline" and method == "helix":
        parse_result("./real_llama70b/helix_offline/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama70b" and serving_mode == "online" and method == "helix":
        parse_result("./real_llama70b/helix_online/events.txt", warm_up_time=200, finish_time=300)
    if model_name == "llama70b" and serving_mode == "offline" and method == "swarm":
        parse_result("./real_llama70b/swarm_offline/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama70b" and serving_mode == "online" and method == "swarm":
        parse_result("./real_llama70b/swarm_online/events.txt", warm_up_time=200, finish_time=300)
    if model_name == "llama70b" and serving_mode == "offline" and method == "separate_a100":
        parse_result("./real_llama70b/separate_offline/a100/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama70b" and serving_mode == "offline" and method == "separate_l4":
        parse_result("./real_llama70b/separate_offline/l4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama70b" and serving_mode == "offline" and method == "separate_t4":
        parse_result("./real_llama70b/separate_offline/t4/events.txt", warm_up_time=60, finish_time=300)
    if model_name == "llama70b" and serving_mode == "online" and method == "separate_a100":
        parse_result("./real_llama70b/separate_online/a100/events.txt", warm_up_time=200, finish_time=300)
    if model_name == "llama70b" and serving_mode == "online" and method == "separate_l4":
        parse_result("./real_llama70b/separate_online/l4/events.txt", warm_up_time=200, finish_time=300)
    if model_name == "llama70b" and serving_mode == "online" and method == "separate_t4":
        parse_result("./real_llama70b/separate_online/t4/events.txt", warm_up_time=200, finish_time=300)

    # aggregate results
    if model_name == "llama30b" and serving_mode == "online" and method == "helix":
        aggregate_latency(["./real_llama30b/helix_online/a100/events.txt",
                           "./real_llama30b/helix_online/l4/events.txt",
                           "./real_llama30b/helix_online/t4/events.txt"], warm_up_time=60, finish_time=300)
    if model_name == "llama30b" and serving_mode == "online" and method == "separate":
        aggregate_latency(["./real_llama30b/separate_online/a100/events.txt",
                           "./real_llama30b/separate_online/l4/events.txt",
                           "./real_llama30b/separate_online/t4/events.txt"], warm_up_time=60, finish_time=300)
    if model_name == "llama70b" and serving_mode == "online" and method == "separate":
        aggregate_latency(["./real_llama70b/separate_online/a100/events.txt",
                           "./real_llama70b/separate_online/l4/events.txt",
                           "./real_llama70b/separate_online/t4/events.txt"], warm_up_time=200, finish_time=300)


if __name__ == '__main__':
    main()
