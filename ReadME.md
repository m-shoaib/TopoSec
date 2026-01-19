# TopoSec: Experimental Data and Analysis

This repository contains experimental data and analysis scripts for the TopoSec paper: **"TopoSec: A Multi-Module Defense Framework for SDN Topology Attacks"**.

## 📊 Dataset Description

The `data/` directory contains experimental data collected from our SDN testbed over 10 experimental runs:

| File | Description | Records |
|------|-------------|---------|
| `lldp_events.csv` | Raw LLDP packet events captured by controller | 50,000 |
| `detections.csv` | Detection records from TopoSec defense modules | ~9,500 |
| `link_state.csv` | Link state transitions during experiments | 148 |
| `performance_metrics.csv` | System performance measurements | 30 |
| `decoy_link_analysis.csv` | Analysis of decoy link effectiveness | 15 |
| `experiment_artifacts.csv` | Anomalies and artifacts observed | 10 |
| `table2_detection_performance.csv` | Paper Table 2: Detection performance | 11 |
| `table3_comparative_detection.csv` | Paper Table 3: Comparative detection | 11 |
| `table4_latency_analysis.csv` | Paper Table 4: Latency analysis | 7 |
| `table5_cpu_overhead.csv` | Paper Table 5: CPU overhead | 5 |
| `table6_memory_consumption.csv` | Paper Table 6: Memory consumption | 5 |
| `table7_toposec_modules.csv` | Paper Table 7: Module resources | 11 |
| `table8_network_overhead.csv` | Paper Table 8: Network overhead | 6 |

## 🔬 Experimental Setup

Our experiments were conducted on a testbed with:

- **Controller**: Ryu SDN controller with TopoSec modules
- **Switches**: 20 OpenFlow 1.3 switches
- **Network**: Mininet emulation environment
- **Attacks**: 10 topology attack types (100 instances each)
- **Runs**: 10 experimental runs with different seeds
- **Metrics**: Detection rate, latency, false positives, resource usage

Detailed setup information is available in [`docs/EXPERIMENT_SETUP.md`](docs/EXPERIMENT_SETUP.md).

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt