## Intra-Species PPI Network Reconstruction

This task focuses on reconstructing the complete human PPI test network using a trained PPI prediction model.
We provide three sampling strategies to simulate different network topologies: **BFS**, **DFS**, and **Random Walk**.
The following example demonstrates the **BFS** sampling strategy.

---

### 1. Required Files

```
./pring_dataset/human/BFS/human_train_ppi.txt      # Training pairs
./pring_dataset/human/BFS/human_val_ppi.txt        # Validation pairs
./pring_dataset/human/BFS/human_test_ppi.txt       # Test pairs (binary classification)
./pring_dataset/human/BFS/all_test_ppi.txt         # Test pairs (graph reconstruction)
./pring_dataset/human/human_simple.fasta           # Protein sequences
./pring_dataset/human/BFS/human_test_graph.pkl     # Ground truth test graph
./pring_dataset/human/BFS/test_sampled_nodes.pkl   # BFS-sampled subgraphs
```

---

### 2. Model Training

Use the following files for model training:

* `human_train_ppi.txt` (training set)
* `human_val_ppi.txt` (validation set)

Protein sequences required for model input are provided in `human_simple.fasta`.

---

### 3. Model Inference

After training, use the model to predict PPI pairs listed in `all_test_ppi.txt`.
These predictions are then used to reconstruct the complete test graph.

---

### 4. Model Evaluation

To assess model performance, use the following command as an example:

```bash
python eval.py \
    --ppi_path ../data_process/pring_dataset/human/BFS/all_test_ppi_pred.txt \
    --out_path ../data_process/pring_dataset/human/BFS/ \
    --gt_graph_path ../data_process/pring_dataset/human/BFS/human_test_graph.pkl \
    --test_graph_node_path ../data_process/pring_dataset/human/BFS/test_sampled_nodes.pkl
```

Evaluation metrics include:

* **Graph Similarity**
* **Relative Density**
* **Degree Distribution (MMD)**
* **Clustering Coefficient (MMD)**
* **Spectral (MMD)**

`eval.py` requires an `all_test_ppi_pred.txt` file containing the predicted pairs in the following format:

```
uniprot_id1 uniprot_id2 label
```

where `label` = 1 for positive pairs and 0 for negative pairs.

---

### 5. Notes

For faster model iteration, you may also perform **binary classification evaluation**:

* Instead of using `all_test_ppi.txt`, run inference on `human_test_ppi.txt`.
* This approach evaluates prediction accuracy without requiring full graph reconstruction.
