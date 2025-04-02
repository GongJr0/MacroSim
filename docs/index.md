# MacroSim Documentation

MacroSim is a python library aimed at creating symbloic models of economic variables thgough PySR's `PySRRegressor`. It utilizes a FRED series accessor and a cunstomizable equation search engine to find an accurate smybolic representation of the selected variavle using the features retrieved from FRED series. MacroSim contains a simulation engine that has the capability to extrapolate the given data points using fully symbolic, per-variable growth rate equations.


MacroSim's main focus is not on producing the most accurate output, but to ensure explorability of outputs for research purposes. Both the symbolic regression results and the fitted growth equations prioritize interpretability. Although kinks are often produced in the output extrapolation process, the equations themselves are configured to be differenciable in most cases.


## Installation

MacroSim can be intalled throuh `pip`; and the builds are available in the github repo if you prefer to install it manually. The pip command required to retireve MacroSim is:

```bash
python -m pip install macrosim
```
