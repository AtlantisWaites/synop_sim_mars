# synop_sim_mars
Ptolemaic Synastry as a Crew Selection Algorithm - A Projection Model for Long-Duration Spaceflight
# Synastry-Optimized Mars Crew Simulation

**A projection model for reducing interpersonal conflict in long-duration spaceflight crews**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains a simulation that projects a **14.8% reduction** in daily crew conflict using synastry-based pattern matching for crew selection and role assignment.

- **1,000 STEM natal charts** (1980–2000 birth years, realistic orbital constraints)  
- **250 four-person crews** simulated over a **1,825-day Mars analog**  
- **Control**: Random assignment  
- **Optimized**: Top crews selected from 20,000 synastry-scored combinations  
- **Calibration**: Expedition 13 ISS data (predicted 0.128 vs reported ~0.15 incidents/day)  
- **Statistical significance**: p = 0.00015 (p < 0.001, 10,000 Monte Carlo runs)

The model is **uncalibrated** beyond Expedition 13 and invites real crew data for refinement.

**White Paper**: [synastry-mars-crew-white-paper.pdf](synastry-mars-crew-white-paper.pdf) (full methodology, results, validation)

## Quick Start

# Clone the repo
git clone https://github.com/atlantiswaites/synop-sim-mars.git
cd synop-sim-mars

# Install dependencies
pip install pandas matplotlib

# Generate charts (if not present)
python generate_charts.py

# Run simulation + validation
python synop_sim_mars.py

Output includes:

Console results (14.8% reduction, p-value)
Figures: conflict histogram, Mars sign distribution
Expedition 13 and CHAPEA proxy validation

Validation:

Expedition 13: Predicted 0.128 vs real ~0.15 incidents/day (Excellent)
CHAPEA proxy: Predicted 0.144 vs reported ~0.05 (gap consistent with known under-reporting)

Repository Contents

synop_sim_mars.py — Main simulation + validation
generate_charts.py — Natal chart generator
mars_crew_1000.csv — Sample dataset (regeneratable)
fig1_conflict_distribution.png — Daily conflict histogram
fig2_cumulative_incidents.png — Cumulative incidents (optional)
synastry-mars-crew-white-paper.pdf — Full paper

<img width="960" height="720" alt="results_mars_distribution" src="https://github.com/user-attachments/assets/1fe0be16-2875-47dd-92ec-dbc47f5d762d" />
<img width="2000" height="1200" alt="fig2_cumulative_incidents" src="https://github.com/user-attachments/assets/c944ea9b-f8d8-4904-8159-344d8b11a4be" />
<img width="2000" height="1200" alt="fig1_conflict_distribution" src="https://github.com/user-attachments/assets/eaa03f55-b68c-483c-9681-bb4bd4cc7551" />


Calibration & Collaboration
The model is calibrated to publicly available Expedition 13 data.
Real crew birth records + conflict logs would enable precise tuning.

NASA / SpaceX / Analog Teams:
Interested in blind validation? Share 10 anonymized crew birth dates and conflict summaries.
We’ll predict daily incidents. You score the outcome.

Contact: atlantiswaites@gmail.com or open an issue.

License
MIT License — feel free to fork, test, extend.

Jessica Marie Pena (Atlantis Waites)
Independent Researcher
December 2025
