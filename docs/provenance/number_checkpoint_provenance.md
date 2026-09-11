# Relay Joint v22-s0 number and checkpoint provenance

## Evaluation identity

- Pipeline repository: /home/yangyuwei/babylm/2026/babylm-eval
- HEAD: 3d57ddc8c40ee795c0b5e41b3a20251a9457a593
- Repository state at audit: dirty
- Primary human-readable source: /home/yangyuwei/babylm/2026/AMLM/eval_results_summary_new_pipeline.md
- Aggregation: NLP Avg = mean(7 NLP metrics); Human Avg = mean(Reading, AoA); Overall = mean(9 top-level metrics)

## Exact checkpoints

| Paper name | Directory | Checkpoint |
|---|---|---|
| Standard MLM | models/bb26_claim_clean_s0 | chck_100M |
| Syntagmatic | models/bb26_claim_comp_entguard_s0 | chck_100M |
| Paradigmatic | models/bb26_claim_agg_synonly_s0 | chck_100M |
| Relay Joint | models/bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s0 | chck_100M |

The summary file calls the Relay model prefix Delayed Joint. This is an obsolete display alias. The directory and checkpoint are exactly the formal Relay Joint model requested for the paper.

## Main table derivation

The direct paper CSV is v22_analysis/data/main_results_v22_s0.csv.

For every model, BLiMP, BLiMP Supplement, EWoK, Entity Tracking, and COMPS are the s0 values in the model section of eval_results_summary_new_pipeline.md. The underlying files are under:

/home/yangyuwei/babylm/2026/babylm-eval/strict/results/MODEL/main/zero_shot/mlm/

GlobalPIQA is the mean of the parallel and nonparallel scores. Reading is the mean of the SPR and ET scores in reading/report.txt. AoA is read from AoA_word/aoa_score.json. (Super)GLUE is the macro-average of BoolQ, MultiRC, RTE, WSC, MRPC F1, QQP F1, and MNLI.

| Method | PIQA-P | PIQA-NP | GlobalPIQA | SPR | ET | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard MLM | 24.27 | 46.00 | 35.13 | 4.25 | 7.81 | 6.03 | 0.00 |
| Syntagmatic | 25.24 | 52.00 | 38.62 | 4.48 | 8.10 | 6.29 | 0.00 |
| Paradigmatic | 25.24 | 44.00 | 34.62 | 4.17 | 7.46 | 5.81 | 0.00 |
| Relay Joint | 27.18 | 48.00 | 37.59 | 4.21 | 7.46 | 5.83 | 0.00 |

| Method | BoolQ | MultiRC | RTE | WSC | MRPC F1 | QQP F1 | MNLI | (Super)GLUE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard MLM | 68.99 | 57.55 | 63.31 | 63.46 | 83.13 | 68.66 | 48.66 | 64.82 |
| Syntagmatic | 67.58 | 66.83 | 62.59 | 69.23 | 82.74 | 69.69 | 53.99 | 67.52 |
| Paradigmatic | 67.16 | 58.50 | 64.03 | 67.31 | 82.39 | 68.46 | 52.63 | 65.78 |
| Relay Joint | 66.79 | 60.52 | 61.15 | 65.38 | 82.49 | 69.22 | 51.51 | 65.29 |

Relay Joint final aggregates are NLP Avg 51.81, Human Avg 2.92, and Overall 40.94. Its Overall difference from seed-matched Standard MLM is +1.25.

## Figure provenance

Figure 2 reads all 19 checkpoint result directories for the four s0 model families. Its source CSVs are in v22_analysis/data/figure2/. Its copied working generator is v22_analysis/scripts/working/plot_figure2_reading_aoa_v22_s0.py.

Figure 3 uses the four exact 100M checkpoints. Its primary comparison CSV is v22_analysis/data/figure3/paradigmatic_alignment_method_comparisons_v22_s0.csv. Equal-slice values are in paradigmatic_equal_slice_delta_v22_s0.csv. The copied working analysis scripts and plotter are in v22_analysis/scripts/working/.

## Relay method provenance

Relay training details were audited from the model-local run_cmd.sh, claim_manifest.txt, train_code.py, and saved configuration. The relevant directory is:

/home/yangyuwei/babylm/2026/AMLM/models/bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s0/

The paper uses Syn onset 0.0920386562, syntax Para onset 0.1840773125, syntax-position Syn retention 0.10, content Para onset 0.99, entity-anchor interval 0.0920386562 through 0.60, Syn slice 0:256, and Para slice 256:384.
