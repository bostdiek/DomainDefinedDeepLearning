# Physics-Informed Neural Networks (PINNs): A Hands-On Course

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive tutorial series on **Physics-Informed Neural Networks (PINNs)** — teaching how to embed physical laws directly into neural network training. This course progresses from simple ODEs to complex PDEs, building intuition through hands-on examples.

## 🎯 Learning Objectives

By completing this course, you will:

1. **Understand the PINN paradigm**: Learn why and when to embed physics into neural networks
2. **Master automatic differentiation**: Use PyTorch autograd to compute derivatives through networks
3. **Solve forward problems**: Predict system behavior given physical parameters and equations
4. **Solve inverse problems**: Estimate unknown physical parameters from noisy observations
5. **Handle increasing complexity**: Progress from 1D ODEs to 2D PDEs with shock formation
6. **Recognize PINN limitations**: Understand when PINNs struggle (e.g., sharp gradients, shocks)

## 📚 Course Structure

The course consists of three progressively challenging notebooks:

### Notebook 1: Simple Harmonic Oscillator
**File:** `notebooks/01_simple_harmonic_oscillator.ipynb`

| Topic | Description |
|-------|-------------|
| **Physics** | Undamped oscillator: $\ddot{x} + \omega^2 x = 0$ |
| **Concepts** | Loss functions, vanilla NN vs PINN, initial conditions |
| **Key Insight** | Physics constraints prevent overfitting and enable extrapolation |

**What you'll implement:**
- Neural network architecture with tanh activation
- Data loss (MSE) for fitting observations
- Physics loss encoding the ODE
- Comparison: vanilla NN fails to extrapolate, PINN succeeds

---

### Notebook 2: Damped Harmonic Oscillator
**File:** `notebooks/02_damped_harmonic_oscillator.ipynb`

| Topic | Description |
|-------|-------------|
| **Physics** | Damped oscillator: $\ddot{x} + \gamma\dot{x} + kx = 0$ |
| **Regimes** | Underdamped, critically damped, overdamped |
| **Advanced** | Inverse problem — learn damping coefficient from data |

**What you'll implement:**
- Forward problem for all three damping regimes
- Inverse problem: estimate unknown $\gamma$ from noisy observations
- Parameterized PINN: one network for any $(d, \omega_0)$ values

**Key equations:**

| Regime | Condition | Solution Behavior |
|--------|-----------|-------------------|
| Underdamped | $\gamma^2 < 4mk$ | Oscillates with decay |
| Critically damped | $\gamma^2 = 4mk$ | Fastest return to equilibrium |
| Overdamped | $\gamma^2 > 4mk$ | Slow exponential decay |

---

### Notebook 3: Burgers' Equation
**File:** `notebooks/03_burgers_equation.ipynb`

| Topic | Description |
|-------|-------------|
| **Physics** | Viscous Burgers: $u_t + u u_x = \nu u_{xx}$ |
| **Challenge** | Shock formation, nonlinear PDE, 2D input (t, x) |
| **Validation** | Finite-difference reference solver comparison |

**What you'll implement:**
- 2D PINN architecture: $(t, x) \rightarrow u$
- PDE residual with mixed partial derivatives
- Shock diagnostics: gradient magnitude heatmaps
- Viscosity sweep: see how PINN accuracy degrades with sharper shocks

**Key physics:**
- Shock formation time: $t^* = 1/\pi \approx 0.318$ for $u(0,x) = -\sin(\pi x)$
- Low viscosity → sharp shocks → higher PINN error
- Collocation density study: more physics points → better accuracy

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- Basic understanding of:
  - Neural networks and PyTorch
  - Differential equations (ODEs and PDEs)
  - Calculus (derivatives, gradients)

### Installation

#### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/DarshKodwani/DomainDefinedDeepLearning.git
cd DomainDefinedDeepLearning

# Install dependencies with uv
uv sync

# Launch Jupyter
uv run jupyter notebook
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/DarshKodwani/DomainDefinedDeepLearning.git
cd DomainDefinedDeepLearning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

#### Option 3: Using Dev Container (VS Code)

1. Install [Docker](https://www.docker.com/get-started) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the repository in VS Code
3. Click "Reopen in Container" when prompted (or use `Cmd/Ctrl + Shift + P` → "Dev Containers: Open Folder in Container")

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.9.1 | Neural networks, autograd |
| `numpy` | ≥2.4.0 | Numerical operations |
| `matplotlib` | ≥3.10.8 | Visualization |
| `notebook` | ≥7.5.1 | Jupyter notebooks |
| `ipykernel` | ≥7.1.0 | Jupyter kernel |

---

## 📖 Pedagogical Approach

### For Instructors

Each notebook follows a consistent structure:

1. **Learning Objectives** — Clear goals at the start
2. **Physics Background** — Mathematical derivation with LaTeX equations
3. **Implementation** — Code cells with `# TODO` comments for students
4. **Validation** — Assert statements to verify implementations
5. **Visualization** — Rich plots showing physics and PINN behavior
6. **Summary** — Key takeaways and references

### Code Structure

Student exercises are marked with:
```python
# TODO: Description of what to implement
# Hint: Helpful guidance

# SOLUTION START
# ... instructor solution ...
# SOLUTION END
```

### Key Concepts Covered

| Concept | Notebook 1 | Notebook 2 | Notebook 3 |
|---------|:----------:|:----------:|:----------:|
| Neural network basics | ✅ | ✅ | ✅ |
| Automatic differentiation | ✅ | ✅ | ✅ |
| Physics loss function | ✅ | ✅ | ✅ |
| Initial conditions | ✅ | ✅ | ✅ |
| Boundary conditions | — | — | ✅ |
| Forward problem | ✅ | ✅ | ✅ |
| Inverse problem | — | ✅ | — |
| Multiple regimes | — | ✅ | ✅ |
| Parameterized PINN | — | ✅ | — |
| Shock/sharp gradients | — | — | ✅ |
| Numerical validation | — | — | ✅ |

---

## 🔬 The PINN Framework

### What is a Physics-Informed Neural Network?

Traditional neural networks learn purely from data. **PINNs** augment this with physical constraints:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}
$$

Where:
- $\mathcal{L}_{\text{data}}$: Fit observations (e.g., initial/boundary conditions)
- $\mathcal{L}_{\text{physics}}$: Satisfy the governing differential equation

### Why PINNs?

| Traditional ML | Physics-Informed ML |
|----------------|---------------------|
| Requires lots of data | Works with sparse data |
| May violate physics | Respects physical laws |
| Interpolation only | Can extrapolate |
| Black box | Interpretable constraints |

### When to Use PINNs

✅ **Good for:**
- Sparse or expensive data
- Well-known governing equations
- Smooth solutions
- Inverse problems (parameter estimation)

⚠️ **Challenging for:**
- Sharp gradients / shocks
- Highly turbulent flows
- Discontinuous solutions
- Very high-dimensional problems

---

## 📁 Repository Structure

```
DomainDefinedDeepLearning/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── pyproject.toml           # Project configuration
├── requirements.txt         # Pip dependencies
├── notebooks/
│   ├── 01_simple_harmonic_oscillator.ipynb
│   ├── 02_damped_harmonic_oscillator.ipynb
│   └── 03_burgers_equation.ipynb
├── plots/                   # Generated figures
└── .devcontainer/          # VS Code dev container config
```

---

## 📚 References

### Foundational Papers

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. [DOI](https://doi.org/10.1016/j.jcp.2018.10.045)

2. **Lagaris, I. E., Likas, A., & Fotiadis, D. I.** (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Transactions on Neural Networks*, 9(5), 987-1000. [DOI](https://doi.org/10.1109/72.712178)

3. **Karniadakis, G. E., et al.** (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422-440. [DOI](https://doi.org/10.1038/s42254-021-00314-5)

### Additional Resources

- [DeepXDE Library](https://github.com/lululxvi/deepxde) — Popular PINN library
- [NVIDIA Modulus](https://developer.nvidia.com/modulus) — Industrial-scale physics-ML framework
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

---

## 🛠️ Development

### Code Quality

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
uv run ruff check notebooks/

# Auto-fix issues
uv run ruff check --fix notebooks/

# Format code
uv run ruff format notebooks/
```

Configuration in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 150

[tool.ruff.lint]
extend-select = ["I"]  # Import sorting
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `ruff check` and `ruff format`
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- Physics-Informed Neural Networks Course Development Team
- Microsoft AI for Good Research Lab

---

## 🙏 Acknowledgments

- The PINN community for foundational research
- PyTorch team for automatic differentiation
- Students and instructors who provided feedback

---

*Last updated: December 2025*

