## GO Enrichment Analysis

This task evaluates the **functional quality** of the reconstructed PPI networks.
We demonstrate the HUMAN test PPI network as an example, using the BFS, DFS, and Random Walk sampled networks.

---

### GO Term Categories (Quick Reference)

| Code   | Category           | Description                                                      |
| ------ | ------------------ | ---------------------------------------------------------------- |
| GO\:BP | Biological Process | Biological objectives or processes (e.g., cell cycle, apoptosis) |
| GO\:MF | Molecular Function | Molecular-level activities (e.g., binding, catalysis)            |
| GO\:CC | Cellular Component | Locations within the cell (e.g., nucleus, membrane)              |

---

### 1. GO Term Retrieval

We provide a script to retrieve GO terms for proteins in a PPI network.
For example, to extract GO terms for the **BFS-sampled HUMAN test PPI network**:

```bash
python go_retrieve.py \
    --graph_path ../data_process/pring_dataset/human/BFS/human_test_graph.pkl \
    --output_path ./test_BFS_go_terms.pkl
```

---

### 2. GO Enrichment Analysis

First, reconstruct the HUMAN test PPI network using predicted all-against-all pairs (see **Topology-Oriented Task** for details).
For the **BFS-sampled HUMAN test PPI network**, you will need:

```
./pring_dataset/human/BFS/all_test_pred_ppi.txt   # Predicted all-against-all pairs
./pring_dataset/human/BFS/human_test_graph.pkl    # Ground truth test graph
```

Run the evaluation with:

```bash
python eval.py \
    --pred_file ../data_process/pring_dataset/human/BFS/all_test_pred_ppi.txt \
    --gt_file ../data_process/pring_dataset/human/BFS/human_test_graph.pkl \
    --uniprot_to_goterms ./test_BFS_go_terms.pkl \
    --source GO:BP
```

Again, the `all_test_pred_ppi.txt` file should contain predicted protein pairs in the format:

```
protein1    protein2    pred_label
```
where `pred_label` belongs to 0 or 1, indicating non-interaction or interaction, respectively.

---

### 3. Evaluation Metrics

The evaluation includes two key metrics: **Functional Alignment** and **Consistency Ratio**.
