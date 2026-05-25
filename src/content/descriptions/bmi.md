# Body Mass Index (BMI)

BMI is calculated by dividing body weight (kg) by the square of height (m):

$$
\text{BMI} = \frac{\text{weight (kg)}}{\text{height (m)}^2}
$$

You can enter the value in either unit using the unit selector — as the raw **Index (kg/m²)** or as a pre-computed **Z-score (SDS)**.

Requires **age** and **sex** to be filled in.

## Index (kg/m²)

Select **Index** to enter the raw BMI value.

## Z-score

Select **Z-score** to enter the age- and sex-adjusted Z-score directly.

## How the Z-score is computed

Using the LMS method of [_Cole & Green (1992)_](https://onlinelibrary.wiley.com/doi/10.1002/sim.4780111005):

$$
Z = \frac{\left(\dfrac{\text{BMI}}{M}\right)^{L} - 1}{L \cdot S}
$$

where $L$, $M$, $S$ are sex-specific UK90 reference parameters interpolated at the patient's age.
