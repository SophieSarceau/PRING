## Complex Pathway Prediction

### 1. Data Description

All required data files are located in the `./complex_pathway/data` folder:

```
./complex_pathway/data/complex_test_pairs.txt   # Protein pairs for model inference
./complex_pathway/data/complex_proteins.fasta   # Protein sequences for model input
./complex_pathway/data/complex_graphs.pkl       # Ground truth complex graph
```

The file `complex_test_pairs.txt` follows the format:

```
protein1    protein2    label
O14807      P36873      1
```

where `label` indicates whether an interaction exists between the two proteins (`1` = interaction, `0` = no interaction).

---

### 2. Evaluation

Use the PPI prediction model trained on the **HUMAN dataset** (e.g., BFS strategy) to predict interactions for all pairs in `complex_test_pairs.txt`.

Run the provided evaluation script:

```bash
python eval.py \
    --pred_path ./data/complex_test_pred_pairs.txt \
    --complex_graph_path ./data/complex_graphs.pkl
```
The evaluation metrics include **Pathway Recall**, **Pathway Precision**, and **Pathway Connectivity**.

The prediction file `complex_test_pred_pairs.txt` should follow the format:

```
protein1    protein2    pred_label
```

where `pred_label` is the predicted interaction label (either `1` for interaction or `0` for no interaction).
