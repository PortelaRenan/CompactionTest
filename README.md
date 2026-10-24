<div align="center">

# 🧵 Compaction Test of Infiltrated UD-NCF

**Viscoelastic model fitting (Maxwell and fractional Zener) with Particle Swarm Optimization**

🎓 *Developed as part of my doctoral research in Mechanical Engineering at the University of Waterloo*

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
[![Paper](https://img.shields.io/badge/paper-Composites%20Part%20A-005EB8)](https://www.sciencedirect.com/science/article/pii/S1359835X26000904)
![License](https://img.shields.io/badge/license-MIT-green)
![Research](https://img.shields.io/badge/PhD%20research-University%20of%20Waterloo-FFC72C)

[About](#-about) · [Contents](#-contents) · [Method](#-method) · [How to run](#-how-to-run) · [Publication](#-publication) · [Contact](#-contact)

</div>

---

## 📌 About

This repository contains the **data analysis and curve fitting codes** for the **compaction tests** that I conducted during my **doctoral research in Mechanical Engineering at the University of Waterloo** (Canada).

The tests measure the **time-dependent compaction response of an infiltrated, binder-stabilized unidirectional non-crimp fabric (UD-NCF)**, a reinforcement used in **wet compression molding (WCM)** of composite parts. The experimental data were used to fit two viscoelastic models, the **Maxwell** model and the **fractional Zener** model, with **Particle Swarm Optimization (PSO)**.

The results of this work were used in the publication listed in the [Publication](#-publication) section.

## 📂 Contents

| File | Description |
|---|---|
| [`UD-NCF Compression Test.ipynb`](<UD-NCF Compression Test.ipynb>) | Notebook with the analysis of the UD-NCF compression test |
| [`compaction_analysis.py`](compaction_analysis.py) | Script for the analysis of the compaction test data |
| [`compaction_problem.py`](compaction_problem.py) | Definition of the curve fitting problem for the compaction data |
| [`maxwell_problem.py`](maxwell_problem.py) | Definition of the curve fitting problem for the Maxwell model |
| [`Functions.py`](Functions.py) | Auxiliary functions (script version) |
| [`Functions.ipynb`](Functions.ipynb) | Auxiliary functions (notebook version) |
| [`LICENSE`](LICENSE) | MIT license |

<!-- Add here any data folders, and confirm the description of each file -->

## 🧠 Method

1. **Compaction tests**: infiltrated UD-NCF samples are compressed and the force is recorded over time, showing the **stress relaxation** of the fabric after the load is applied.
2. **Viscoelastic models**: the time-dependent response is described by two models.
   - **Maxwell model**: a spring in series with a dashpot, which gives an exponential relaxation.
   - **Fractional Zener model**: uses fractional derivative elements (spring-pots), which can describe a **broader range of relaxation times** with few parameters.
3. **Parameter identification**: the model parameters are found with **Particle Swarm Optimization (PSO)**, a population-based optimization method that searches for the parameters that minimize the error between the model and the experimental data.

In the publication, the fractional Zener model gave a better fit to the experimental data than the Maxwell model.

<!--
## 🖼️ Results
Add the fitted curves (experimental data vs. Maxwell vs. fractional Zener) here. Save the images in an images/ folder, for example:

| Maxwell fit | Fractional Zener fit |
|:---:|:---:|
| ![Maxwell](images/maxwell_fit.png) | ![Zener](images/zener_fit.png) |
-->

## 🚀 How to run

**Requirements:** Python 3.8 or later.

**1. Clone the repository**

```bash
git clone https://github.com/PortelaRenan/REPO-NAME.git
cd REPO-NAME
```

**2. Install the dependencies**

```bash
pip install numpy scipy pandas matplotlib jupyter
```

**3. Run the analysis**

```bash
jupyter notebook "UD-NCF Compression Test.ipynb"
```

or run the scripts directly:

```bash
python compaction_analysis.py
```

<!-- Replace REPO-NAME with the name of this repository, adjust the dependencies (including the PSO library, if any) to the imports of the code, and describe the inputs and outputs -->

## 🛠️ Technologies

- **Python**: data processing and modeling
- **NumPy** and **SciPy**: numerical computation
- **pandas**: handling of the test data
- **Matplotlib**: plots of the results
- **Jupyter Notebook**: interactive analysis
- **Particle Swarm Optimization**: identification of the model parameters

## 📝 Publication

The results of this work were used in:

> **Characterizing and modelling the time-dependent compaction response of an infiltrated binder-stabilized unidirectional non-crimp fabric**
> *Composites Part A: Applied Science and Manufacturing*, 2026, Article 109643.
> 🔗 [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1359835X26000904)

If you use this code or data, please cite the paper:

```bibtex
@article{portela_compaction_2026,
  author  = {AUTHORS},
  title   = {Characterizing and modelling the time-dependent compaction response of an infiltrated binder-stabilized unidirectional non-crimp fabric},
  journal = {Composites Part A: Applied Science and Manufacturing},
  year    = {2026},
  pages   = {109643},
  url     = {https://www.sciencedirect.com/science/article/pii/S1359835X26000904}
}
```

## 🔗 Related projects

- [data-analysis-static-crush](https://github.com/PortelaRenan/data-analysis-static-crush): data analysis of static crush tests of composite hat channels, also from my doctoral research

## 📜 License

Distributed under the **MIT** license. See the [`LICENSE`](LICENSE) file for details.

## 📬 Contact

**Renan Portela**, PhD in Mechanical Engineering (University of Waterloo, Canada) · Master's in Mechanical Engineering (UNICAMP)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/YOUR-PROFILE)
[![Email](https://img.shields.io/badge/E--mail-D14836?logo=gmail&logoColor=white)](mailto:YOUR-EMAIL@example.com)

---

<div align="center">

⭐ If this repository helped you, leave a star!

</div>
