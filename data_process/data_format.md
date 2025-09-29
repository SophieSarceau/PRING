PRING Dataset Structure

```
PRING Dataset
├── arath
│   ├── arath_all_test_ppi.txt      # all-against-all pairs in the test graph
│   ├── arath_BFS_sampled_nodes.pkl # BFS sampled subgraphs
│   ├── arath_DFS_sampled_nodes.pkl # DFS sampled subgraphs
│   ├── arath_graph.pkl             # full PPI graph
│   ├── arath_ppi.txt               # PPI pairs
│   ├── arath_protein_id.csv        # protein id mapping
│   ├── arath_RANDOM_WALK_sampled_nodes.pkl # RANDOM_WALK sampled subgraphs
│   ├── arath_simple.fasta          # protein sequences with UniProt IDs
│   ├── arath_test_graph.pkl        # ground truth test graph
│   ├── arath_test_ppi.txt          # test PPI pairs for binary classification
│   └── arath.fasta                 # protein sequences with complete meta info
├── ecoli
│   ├── ecoli_all_test_ppi.txt
│   ├── ecoli_BFS_sampled_nodes.pkl
│   ├── ecoli_DFS_sampled_nodes.pkl
│   ├── ecoli_graph.pkl
│   ├── ecoli_ppi.txt
│   ├── ecoli_protein_id.csv
│   ├── ecoli_RANDOM_WALK_sampled_nodes.pkl
│   ├── ecoli_simple.fasta
│   ├── ecoli_test_graph.pkl
│   ├── ecoli_test_ppi.txt
│   └── ecoli.fasta
├── human
│   ├── BFS
│   │   ├── all_test_ppi.txt     # all-against-all pairs in the test graph
│   │   ├── human_BFS_split.pkl  # BFS sampled subgraphs
│   │   ├── human_test_graph.pkl # ground truth test graph
│   │   ├── human_test_ppi.txt   # test PPI pairs for binary classification
│   │   ├── human_train_graph.pkl # ground truth train graph
│   │   ├── human_train_ppi.txt  # train PPI pairs for PPI prediction models
│   │   ├── human_val_ppi.txt    # validation PPI pairs for early stopping
│   │   └── test_sampled_nodes.pkl # sampled nodes in the test graph
│   ├── DFS
│   │   ├── all_test_ppi.txt
│   │   ├── human_DFS_split.pkl
│   │   ├── human_test_graph.pkl
│   │   ├── human_test_ppi.txt
│   │   ├── human_train_graph.pkl
│   │   ├── human_train_ppi.txt
│   │   ├── human_val_ppi.txt
│   │   └── test_sampled_nodes.pkl
│   ├── RANDOM_WALK
│   │   ├── all_test_ppi.txt
│   │   ├── human_RANDOM_WALK_split.pkl
│   │   ├── human_test_graph.pkl
│   │   ├── human_test_ppi.txt
│   │   ├── human_train_graph.pkl
│   │   ├── human_train_ppi.txt
│   │   ├── human_val_ppi.txt
│   │   └── test_sampled_nodes.pkl
│   ├── human_graph.pkl          # full PPI graph
│   ├── human_ppi.txt            # PPI pairs
│   ├── human_protein_id.csv     # protein id mapping
│   ├── human_simple.fasta       # protein sequences with UniProt IDs
│   └── human.fasta              # protein sequences with complete meta info
└── yeast
    ├── yeast_all_test_ppi.txt
    ├── yeast_BFS_sampled_nodes.pkl
    ├── yeast_DFS_sampled_nodes.pkl
    ├── yeast_graph.pkl
    ├── yeast_ppi.txt
    ├── yeast_protein_id.csv
    ├── yeast_RANDOM_WALK_sampled_nodes.pkl
    ├── yeast_simple.fasta
    ├── yeast_test_graph.pkl
    ├── yeast_test_ppi.txt
    └── yeast.fasta
```
