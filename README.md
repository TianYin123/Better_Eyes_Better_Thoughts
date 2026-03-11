# Better Eyes Better Thoughts

Supported datasets:
- Path-VQA
- SLAKE
- VQA-RAD
- PMC-VQA
- OmniMedVQA

## Main pipeline (local / vLLM models)

Edit and run:

```bash
bash scripts/eval.sh
```

## For closed model
Run the template:

```bash
bash api_eval/run_batch_exp7.sh
```


## LLM as Judge

The main pipeline supports `use_llm_judge`.

There are also standalone judge scripts for `PMC-VQA` and `OmniMedVQA`:

## Outputs

Common output files:
- `results.json`
- `metrics.json`
- `total_results.json`
