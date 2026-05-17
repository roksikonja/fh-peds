# Body Mass Index (BMI)

BMI is calculated by dividing body weight (kg) by the square of height (m):

$$
\text{BMI} = \frac{\text{weight (kg)}}{\text{height (m)}^2}
$$

The model uses an **age- and sex-adjusted BMI Z-score (SDS)**, not the raw BMI value. You can enter either form using the unit selector.

## Index (kg/m²)

Select **Index** to enter the raw BMI value. It is automatically converted to a Z-score using the **British 1990 (UK90) LMS reference data**.

> Requires **age** and **sex** to be filled in.

## Z-score

Select **Z-score** to enter the age- and sex-adjusted Z-score directly.

## How the Z-score is computed

Using the LMS method of *Cole & Green (1992)*:

$$
Z = \frac{\left(\dfrac{\text{BMI}}{M}\right)^{L} - 1}{L \cdot S}
$$

where $L$, $M$, $S$ are sex-specific UK90 reference parameters interpolated at the patient's age. They are stored at `0.05`-year intervals from `0` to `18` years for each sex.

## References

- Cole TJ, Green PJ. *Statistics in Medicine.* 1992; **11**(10): 1305–19.
- Cole TJ, Freeman JV, Preece MA. *Statistics in Medicine.* 1998; **17**(4): 407–29.
