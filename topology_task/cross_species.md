## Cross-Species PPI Network Reconstruction

This task focuses on reconstructing the PPI networks of non-human species (**ARATH**, **YEAST**, **ECOLI**) using a model trained on the human PPI network.
For each species, three sampling strategies are provided to simulate different network topologies: **BFS**, **DFS**, and **Random Walk**.
The following example demonstrates the workflow for the **ARATH** species.

---

### 1. Required Files

```
./pring_dataset/arath/arath_all_test_ppi.txt              # All-against-all test pairs
./pring_dataset/arath/arath_test_graph.pkl                # Ground truth test graph
./pring_dataset/arath/arath_BFS_sampled_nodes.pkl         # BFS-sampled subgraphs
./pring_dataset/arath/arath_DFS_sampled_nodes.pkl         # DFS-sampled subgraphs
./pring_dataset/arath/arath_RANDOM_WALK_sampled_nodes.pkl # Random Walk-sampled subgraphs
```

---

### 2. Model Training

Use the human-trained PPI prediction model as described in
`./topology_task/intra_species.md`.

You may directly reuse the model trained under a single sampling strategy (e.g., **BFS**) without retraining on the target species.

---

### 3. Model Inference

Run inference on `arath_all_test_ppi.txt` using the human-trained model.
The predicted interactions are then used to reconstruct the complete ARATH test graph.

---

### 4. Model Evaluation

Evaluate the reconstructed graph under different sampling strategies (**BFS**, **DFS**, and **Random Walk**) with:

```bash
python eval.py \
    --ppi_path ../data_process/pring_dataset/arath/arath_all_test_ppi_pred.txt \
    --out_path ../data_process/pring_dataset/arath/ \
    --gt_graph_path ../data_process/pring_dataset/arath/arath_test_graph.pkl \
    --test_graph_node_path ../data_process/pring_dataset/arath/arath_BFS_sampled_nodes.pkl # Or DFS / Random Walk strategies
```

The same set of metrics used in intra-species evaluation applies here, including **Graph Similarity**, **Relative Density**, **Degree Distribution (MMD)**, **Clustering Coefficient (MMD)**, and **Spectral (MMD)**.
