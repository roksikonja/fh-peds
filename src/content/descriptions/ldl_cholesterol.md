# LDL-Cholesterol (LDL-C)

**Required field.** Enter the **untreated** LDL-cholesterol value — i.e. measured *in the absence of lipid-lowering therapy*.

LDL-C is the single most important predictor in the ML-FH-PeDS model.

## Units

Accepted units:

- `mmol/L` — model-native unit
- `mg/dL` — converted using $1\ \text{mmol/L} = 38.67\ \text{mg/dL}$

The model uses the value in `mmol/L` internally.
