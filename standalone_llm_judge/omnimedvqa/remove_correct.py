import argparse
from standalone_llm_judge.common import load_json_list, remove_correct_field, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Remove correct / judge_output fields from OmniMedVQA results json.')
    parser.add_argument('--input_file', type=str, default='results.json')
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    output_file = args.output_file or args.input_file
    data = load_json_list(args.input_file)
    removed = remove_correct_field(data)
    save_json(output_file, data)
    print(f'Removed judge fields from {removed} records. Saved to {output_file}')


if __name__ == '__main__':
    main()
