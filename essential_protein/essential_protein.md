## Essential Protein Justification

This task evaluates whether the reconstructed **HUMAN test PPI network** (e.g., BFS, DFS, Random Walk) can effectively distinguish **essential proteins** from **non-essential proteins**.

---

### 1. File Description

Two key input files are provided:

```
./essential_proteins.txt       # List of essential proteins (HUMAN)
./non_essential_proteins.txt   # List of non-essential proteins (HUMAN)
```

---

### 2. Plot Network Centrality Distribution

We first plot the centrality distribution of essential vs. non-essential proteins.
Example for the **BFS-sampled PPI network**:

```bash
python plot_dist.py \
    --ppi-graph ../data_process/pring_dataset/human/BFS/human_test_graph.pkl
```

This generates:

```
network_centrality_distribution.png
```

This plot provides insight into the separation between essential and non-essential proteins, and helps determine thresholds for selection in the next step.

---

### 3. Select Essential and Non-Essential Proteins

Next, select proteins from the reconstructed test graph using centrality thresholds.
Example for **BFS-sampled PPI network**:

```bash
python select_prots.py \
    --essential_proteins_file ./essential_proteins.txt \
    --non_essential_proteins_file ./non_essential_proteins.txt \
    --graph_file ../data_process/pring_dataset/human/BFS/human_test_graph.pkl \
    --out_essential ./selected_essential_proteins.txt \
    --out_non_essential ./selected_non_essential_proteins.txt \
    --essential_threshold 30 \
    --non_essential_threshold 20
```

* The **essential threshold** and **non-essential threshold** should be chosen based on the distribution plot.
* For other networks (DFS, Random Walk), adjust thresholds accordingly.

This script outputs:

```
./selected_essential_proteins.txt               # Selected essential proteins (e.g., 100 samples)
./selected_non_essential_proteins.txt           # Selected non-essential proteins (e.g., 100 samples)
./selected_network_centrality_distribution.png  # Distribution plot of selected proteins
```

---

### 4. Evaluation

Finally, evaluate the predicted PPI network quality with respect to essential protein discrimination:

```bash
python eval.py \
    --gt_graph ../data_process/pring_dataset/human/BFS/human_test_graph.pkl \
    --pred_ppi ../data_process/pring_dataset/human/BFS/all_test_pred_ppi.txt
```

The prediction file `all_test_pred_ppi.txt` should follow the format:

```
protein1    protein2    label
```

where `label = 1` if an interaction exists, else `0`.

The evaluation outputs two key metrics:

* **Precision\@K (K=100)** – accuracy of identifying essential proteins among the top-ranked predictions
* **Distribution Overlap** – degree of separation between essential and non-essential proteins
