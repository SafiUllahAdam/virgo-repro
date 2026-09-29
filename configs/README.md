# configs/

Optional JSON run-configs. Instead of passing flags, point `main.py` at a file:

```bash
python experiments/run_task.py --config configs/cora_linkpred.json
```

The config's `task` and `dataset` override the CLI flags. Example: `cora_linkpred.json`.
