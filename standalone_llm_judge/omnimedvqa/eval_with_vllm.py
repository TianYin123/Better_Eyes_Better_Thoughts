import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from standalone_llm_judge.common import (
    add_common_args,
    aggregate_total,
    build_judge_messages,
    count_by_key,
    extract_judge_bool,
    load_json_list,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Standalone LLM-as-Judge for OmniMedVQA results.json')
    add_common_args(parser)
    args = parser.parse_args()

    output_file = args.output_file or args.input_file
    data = load_json_list(args.input_file)
    print(f'Loaded {len(data)} records from {args.input_file}')

    print(f'Step 1/4: Building judge messages with {args.max_workers} CPU workers...')
    messages_list = process_map(
        build_judge_messages,
        data,
        max_workers=args.max_workers,
        chunksize=args.chunksize,
        desc='Building Messages',
    )

    print('Step 2/4: Applying chat template...')
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in tqdm(messages_list, desc='Templating')
    ]
    del tokenizer

    print('Step 3/4: Initializing vLLM...')
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    print('Step 4/4: Running inference...')
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True)

    print('Parsing outputs and saving results...')
    for i, output in tqdm(list(enumerate(outputs)), total=len(outputs), desc='Saving'):
        judge_output = output.outputs[0].text if output.outputs else ''
        data[i]['judge_output'] = judge_output
        data[i]['correct'] = extract_judge_bool(judge_output)

    metrics = {
        'total metrics': aggregate_total(data),
        'question type metrics': count_by_key(data, 'question_type'),
        'modality type metrics': count_by_key(data, 'modality_type'),
    }
    save_json(output_file, data)
    save_json(args.metrics_file, metrics)
    print(f'Saved judged results to {output_file}')
    print(f'Saved metrics to {args.metrics_file}')
    print(f"Accuracy: {metrics['total metrics']['acc']:.2%}")


if __name__ == '__main__':
    main()
