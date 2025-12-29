# AILS vs MCPP Comparison - Using Exact Lee & Lee (2025) Benchmark Data

## Quick Start

### Step 1: Download Moving AI Lab Benchmark Maps

The Lee & Lee paper uses maps from the Moving AI Lab MAPF benchmarks.

**Download individually:**
- [den312d.map.zip](https://www.movingai.com/benchmarks/mapf/den312d.map.zip) (65×81, 2,445 vertices)
- [ht_chantry.map.zip](https://www.movingai.com/benchmarks/mapf/ht_chantry.map.zip) (162×141, 7,461 vertices)
- [random-64-64-20.map.zip](https://www.movingai.com/benchmarks/mapf/random-64-64-20.map.zip) (64×64, 3,270 vertices)

**Or download all maps at once:**
- [mapf-map.zip (73KB)](https://www.movingai.com/benchmarks/mapf/mapf-map.zip)

### Step 2: Extract Maps

```bash
mkdir -p benchmark_data
cd benchmark_data
unzip den312d.map.zip
unzip ht_chantry.map.zip
unzip random-64-64-20.map.zip
```

### Step 3: Run the Notebook

```bash
jupyter notebook ails_mcpp_exact_benchmark.ipynb
```

Or run as Python script:
```bash
python ails_mcpp_exact_benchmark.py
```

---

## Data Sources (From Lee & Lee Paper)

### 1. Grid Maps (Moving AI Lab)

| Map | Dimensions | Vertices | Edges | Source |
|-----|------------|----------|-------|--------|
| den312d | 65×81 | 2,445 | 4,391 | Dragon Age game |
| ht_chantry | 162×141 | 7,461 | 13,963 | Game map |
| random-64-64-20 | 64×64 | 3,270 | 5,149 | 20% random |

Download: https://www.movingai.com/benchmarks/mapf/index.html

### 2. Road Network Digital Twins (Non-Grid)

| Network | Vertices | Edges | Location |
|---------|----------|-------|----------|
| pnu | 92 | 112 | Pusan National University, Busan |
| jangjeon | 3,997 | 4,251 | Jangjeon-dong, Busan |

**Options to obtain:**
1. **Contact authors:** myungho.lee@pusan.ac.kr
2. **Extract from OpenStreetMap:**
   ```bash
   pip install osmnx
   ```
   The notebook includes code to extract road networks from OSM.

---

## Experimental Setup (From Lee & Lee Paper)

### Parameters
- `MAX_ITER = 2000` (maximum clustering iterations)
- `THRESHOLD = 0.005` (convergence threshold, 0.5%)
- Agent counts: k = 2, 5, 10, 20, 30, 40

### Agent Position Settings
1. **Arbitrary:** Agents randomly distributed
2. **Clutter:** Agents densely clustered together

### Metrics
- Maximum path length among all agents
- Standard deviation of path lengths
- Execution time (seconds)

---

## File Structure

```
benchmark_data/
├── den312d.map          # Download from Moving AI Lab
├── ht_chantry.map       # Download from Moving AI Lab
└── random-64-64-20.map  # Download from Moving AI Lab

outputs/
├── ails_results_summary.csv
├── ails_results_full.csv
├── mcpp_results_summary.csv
├── mcpp_results_full.csv
├── benchmark_maps.png
├── ails_time_improvement.png
└── mcpp_results.png
```

---

## Expected Results (From Lee & Lee Paper - Table 2)

### Maximum Path Lengths (arbitrary setting)

| Map | Method | k=2 | k=5 | k=10 | k=20 | k=30 | k=40 |
|-----|--------|-----|-----|------|------|------|------|
| den312d | MFC | 3092 | 1282 | 780 | 501 | 372 | 278 |
| den312d | MSTC*-NB | 2580 | 1114 | 677 | 442 | 367 | 331 |
| den312d | **Proposed** | **2654** | **1110** | **628** | **387** | **330** | **275** |
| ht_chantry | MFC | 8285 | 3636 | 2259 | 1394 | 1019 | 818 |
| ht_chantry | MSTC*-NB | 7627 | 3239 | 1782 | 1056 | 820 | 725 |
| ht_chantry | **Proposed** | **7744** | **3230** | **1692** | **1013** | **766** | **664** |

---

## References

1. **Lee & Lee (2025):** "Multi-Agent Coverage Path Planning Using Graph-Adapted K-Means in Road Network Digital Twin", Electronics, 14, 3921
   - DOI: https://doi.org/10.3390/electronics14193921

2. **Moving AI Lab Benchmarks:** https://www.movingai.com/benchmarks/

3. **Map Format Documentation:** https://www.movingai.com/benchmarks/formats.html
