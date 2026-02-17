# EnergyTrading

STRUKTURA PROJEKTU

EnergyTrading/sf_project/loader -> Načte původní data a vytvoří bid&ask, price soubory
EnergyTrading/sf_project/create_prediction -> Modely, vytváření predikcí
EnergyTrading/sf_project/strategy -> SingleEntry a MultiEntry strategie
EnergyTrading/sf_project/plots_statistics -> jen základní statistiky pro finální obchody
EnergyTrading/sf_project/settings -> Zvolené finální nastavení strategie multi-entry pro out-of-sample obchody


## Setup

Clone the repository:

git clone https://github.com/filtip/EnergyTrading.git
cd EnergyTrading

Create virtual environment:

python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows

Install dependencies:

pip install -r requirements.txt

Register Jupyter kernel:

python -m ipykernel install --user --name energytrading

---

## Running the Project

Start Jupyter:

jupyter notebook

Then select kernel:
Kernel → Change Kernel → energytrading