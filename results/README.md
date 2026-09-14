# Evaluation results

The available result artifact is [stage1_evaluation_summary.txt](stage1_evaluation_summary.txt), an existing text summary of the fixed-position baseline. It is not an episode-level annotation table.

The main README retains the recorded Stage 2, Stage 3, stress-test, and grid-retest summaries. Their aggregate success rates have not been reconstructed from committed episode-level annotations because those annotations are not yet present.

The following planned files have not been created; their contents remain pending:

- `stage1_results.csv`
- `stage2_grid_results.csv`
- `stage3_continuous_results.csv`
- `stage3_stress_results.csv`
- `grid_retest_results.csv`
- `figures/` result charts

No example rows, inferred episode outcomes, or zero-filled results are provided. These tables can be added after the manual experiment records are checked against the source episodes.

[videos/manifest.csv](../videos/manifest.csv) describes six selected demonstration clips, not all evaluation trials. Its unconfirmed fields remain blank. Analysis reports generated under local `logs/` contain motion and data-integrity statistics; those statistics do not establish task success.
