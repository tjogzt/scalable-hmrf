# Data manifest

Raw data are **not** versioned. This file documents what each experiment expects and how to
obtain it.

## 1. DLPFC 12 sections (10x Visium, human dorsolateral prefrontal cortex)

* Location: `data/dlpfc12/<section>/`
* Sections: 151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674,
  151675, 151676 (4 donors: Br5292, Br5595, Br8100; 7 cortical layers + white matter)
* Per-section files:
  * `counts.mtx.gz` — gene × spot count matrix (Matrix Market)
  * `genes.txt` — gene symbols (row order)
  * `spots.csv` — spot coordinates (`barcode`, `x`, `y` or array coordinates)
* Provenance: public DLPFC data from the spatialLIBD project (Lieber Institute; Maynard et al.,
  2021). Rebuild with the spatialLIBD Bioconductor package or the 10x public downloads.
* Used by: experiments 11, 12, 14, 16, 17.

## 2. Single-section constant (legacy)

* Location: `data/dlpfc/`
* Files: `151673_raw_feature_bc_matrix.h5`, `cluster_labels.csv`, `tissue_positions_list.txt`
* Used by: experiment 11 (single-section pipeline, legacy path kept for reproducibility).

## 3. Xenium breast IDC (574,852 cells)

* Location: external; set `HMRF_XENIUM_DIR` to the folder containing `outs/`.
  Example: `/Volumes/<disk>/data/Xenium_V1_FFPE_Human_Breast_IDC_outs`
* Files consumed: `outs/cell_feature_matrix.h5`, `outs/cells.csv.gz`, `outs/transcripts.parquet`
  (or the corresponding 10x export for the release used here).
* Used by: experiment 18.

## 4. Derived artefacts shipped with the repository

`experiments/*/results/` contains the CSV/JSON/NPZ outputs quoted in the manuscript (labels,
metrics, timings, equivalence reports). They are small (≤ 34 MB in total for the largest
experiment) and are kept under version control so that figures can be regenerated without
re-running every baseline.

## 5. External software data

* GraphST installs its own pretrained weights if used; the sparse patch only touches
  `GraphST.py`, `model.py` and `preprocess.py` and keeps `.bak-hmrf` backups.
* BayesSpace / SC.MEB use the exported SingleCellExperiment objects (`15_export_sce.R`) or the
  matrices above; no additional data download is required.
