from utils import OmniMedVQA, PATH_VQA, PMC_VQA, SLAKE, VQA_RAD

SUPPORTED_DATASETS = [
    "PATH_VQA",
    "PMC_VQA",
    "SLAKE",
    "VQA_RAD",
    "OmniMedVQA",
]


def prepare_benchmark(model, eval_dataset, eval_dataset_path, eval_output_path):
    if eval_dataset == "PATH_VQA":
        return PATH_VQA(model, eval_dataset_path, eval_output_path)
    if eval_dataset == "PMC_VQA":
        return PMC_VQA(model, eval_dataset_path, eval_output_path)
    if eval_dataset == "SLAKE":
        return SLAKE(model, eval_dataset_path, eval_output_path)
    if eval_dataset == "VQA_RAD":
        return VQA_RAD(model, eval_dataset_path, eval_output_path)
    if eval_dataset == "OmniMedVQA":
        return OmniMedVQA(model, eval_dataset_path, eval_output_path)

    print(f"unknown eval dataset {eval_dataset}, we only support {SUPPORTED_DATASETS}")
    return None


if __name__ == "__main__":
    pass
