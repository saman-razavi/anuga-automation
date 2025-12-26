# anuga-automation

This repository contains automated workflows for setting up, running, and analysing **ANUGA** simulations.

The focus is on **reproducible, script-based modelling**, with a clear separation between:
- Jupyter notebooks for teaching and exploratory analysis
- reusable Python code for automated execution

The repository is intended for research development, student collaboration, and future extension to batch and HPC workflows.

---

## Repository Structure

```text
anuga-automation/
├─ notebooks/    # Teaching notebooks and exploratory workflows
├─ src/          # Reusable ANUGA-related functions and modules
├─ scripts/      # Runnable scripts for automated simulations
├─ configs/      # Configuration files (e.g. YAML/JSON)
├─ docs/         # Extended documentation
├─ DATASTORE/    # Large input data and model outputs (not tracked by Git)
```

---

## Design Philosophy

- **Automation-first**: minimise manual interaction and GUI dependence  
- **Reproducibility**: all model setup and execution steps are explicit and version-controlled  
- **Teaching-oriented**: notebooks are written to be readable and instructional  
- **Refactoring path**: working notebooks are later cleaned up and reorganised into standalone Python scripts, without changing functionality

---

## Platform Notes

ANUGA is natively supported on **Linux systems**.  
All development and execution for this project are therefore assumed to take place on **Ubuntu**.

---

## Status

This repository is under active development. Initial work focuses on a small, self-contained demonstration case that can serve as a template for more complex ANUGA applications.

---

## Contributions

Development is conducted via feature branches.  
Direct commits to the `main` branch are avoided to preserve stability and reproducibility.
