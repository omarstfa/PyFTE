# Fault Tree Analysis Tool (example05.py)

This script performs comprehensive fault tree analysis based on fault logs. It extracts minimal cut sets, estimates failure rates and repair times, computes analytical reliability, importance factors (Birnbaum, Criticality, Fussell‑Vesely), runs Monte Carlo simulation for repairable systems, and performs sensitivity analysis.

## Features

- Extracts minimal cut sets from time‑ordered fault logs.
- Estimates failure rates (λ) and mean repair times (MTTR) from log data.
- Computes exact analytical reliability R(t) using minimal cut sets.
- Calculates equivalent system failure rate λ_eq via small‑time slope and MTTF.
- Computes importance factors: Birnbaum, Criticality, Fussell‑Vesely.
- Monte Carlo simulation of a repairable system (exponential failures, fixed repair times) with per‑component availability time series.
- Sensitivity analysis: multiply each basic event’s failure rate by factors (1–5) and observe system unavailability.
- Generates plots and CSV outputs.

## Requirements

- Python 3.7 or higher (uses dataclasses, f‑strings, and type hints)
- Required packages: `numpy`, `pandas`, `matplotlib`

Install dependencies using:

```bash
pip install -r requirements.txt