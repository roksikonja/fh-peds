Build a website according to this specification.

<layout>

- Modern static website.
- Narrow screen: 1 column.
- Wide-screen: 3 column design with content in the center column.
- Mobile-friendly.
- Background is white.
- Primary color is #36478D.
- Default font size is 11px. Font family is: 'Raleway', sans-serif.

</layout>

<header>

- Header with 2 logos on the right side of the center column.
- Link to two pages: Calculator (`index.html`) and About (about `about.html`).
- All font in the content of about.html to be of the same size.
- Both index.html and about.html should use exactly the same styling.
- Page links in the header to be in the center column, left aligned.
- Both pages share the same layout and style.

</header>

<index.html>

- One input form: <form>
- Two tabs for controlling the calculator selection. We navigate between one or the other by clicking on a tab. Tabs should span the full width of the center column. There are two calculators:
  - ML-FH-PeDS (Default, Left): Uses <calculator-ml-fh-peds>
  - FH-PeDS (Right): Uses <calculator-fh-peds>

</index.html>

<form>

- Fields are in the table below.


| Input Field                        | Field Identifier (Do not display) | Input Type and Values                                                                                                                                                                                                          | Unit  | Tooltip Description                                                                                  |
| ------------------------------------ | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------ |
| Age                                | `age`                             | 0 ≤ Integer ≤ 18                                                                                                                                                                                                             | Years | Enter age at examination in years (example 8.3 years). Only for children and adolescents ≤18 years. |
| Sex                                | `gender`                          | Dropdown Input:                                                                                                                                                                                                                |       |                                                                                                      |
| 1 Female                           |                                   |                                                                                                                                                                                                                                |       |                                                                                                      |
| 2 Male                             |                                   | Enter biological sex as appropriate.                                                                                                                                                                                           |       |                                                                                                      |
| Family History of High Cholesterol | `fh_high_cholesterol`             | Dropdown Input:                                                                                                                                                                                                                |       |                                                                                                      |
| 0 No                               |                                   |                                                                                                                                                                                                                                |       |                                                                                                      |
| 1 Only first degree relative       |                                   |                                                                                                                                                                                                                                |       |                                                                                                      |
| 2 Only second degree relative      |                                   |                                                                                                                                                                                                                                |       |                                                                                                      |
| 3 First and second degree relative |                                   | Select the relationship to a family member who has high cholesterol. Any mention of “high cholesterol” during the examination regardless of exact lipid concentrations or lipid-lowering therapy use is considered positive. |       |                                                                                                      |

- Definitions of relative's degrees: 1) First degree relatives include parents, siblings and children. 2) Second degree relatives include grandparents, grandchildren, aunts, uncles, nieces, and nephews.
- Any family history reported without an exact age is considered positive. |
  | Family History of Premature Coronary Artery Disease | `fh_premature_cad` | Dropdown Input:
  0 No
  1 Only first degree relative
  2 Only second degree relative
  3 First and second degree relative |  | Select the relationship to a family member who has premature coronary artery disease.
- Definitions of relative's degrees: 1) First degree relatives include parents, siblings and children. 2) Second degree relatives include grandparents, grandchildren, aunts, uncles, nieces, and nephews.
- Any family history reported without an exact age is considered positive.
- Premature cardiovascular disease (coronary artery disease or vascular disease) is defined as an even occurring before 55 years in men and 60 years in women. |
  | Family History of Premature Peripheral Artery Disease | `fh_pad_cvi` | Dropdown Input:
  0 No
  1 Only first degree relative
  2 Only second degree relative
  3 First and second degree relative |  | Select the relationship to a family member who has premature vascular artery disease. Vascular Disease is defined as any atherosclerotic disease other than coronary artery disease.
- Definitions of relative's degrees: 1) First degree relatives include parents, siblings and children. 2) Second degree relatives include grandparents, grandchildren, aunts, uncles, nieces, and nephews.
- Any family history reported without an exact age is considered positive.
- Premature cardiovascular disease (coronary artery disease or vascular disease) is defined as an even occurring before 55 years in men and 60 years in women. |
  | Family History of Tendinous Xanthoma or Xanthelasma | `fh_xant` | Dropdown Input:
  0 No
  1 Yes |  | Select the relationship to a family member who has tendinous xanthoma or xanthelasma.
- Definitions of relative's degrees: 1) First degree relatives include parents, siblings and children. 2) Second degree relatives include grandparents, grandchildren, aunts, uncles, nieces, and nephews.
- Any family history reported without an exact age is considered positive. |
  | Family History of Arcus Cornealis | `fh_acrus_senilis` | Dropdown Input:
  0 No
  1 Yes |  | Select the relationship to a family member who has arcus cornealis.
- Definitions of relative's degrees: 1) First degree relatives include parents, siblings and children. 2) Second degree relatives include grandparents, grandchildren, aunts, uncles, nieces, and nephews.
- Any family history reported without an exact age is considered positive. |
  | Total Cholesterol (TC) Level | `total_cholesterol` | Float ≥ 0 | - mmol/L
- mg/dL | Enter untreated lipid values in the absence of lipid lowering therapy. |
  | High-Density Lipoprotein Cholesterol (HDL-C) Level | `hdl_cholesterol` | Float ≥ 0 | - mmol/L
- mg/dL | Enter untreated lipid values in the absence of lipid lowering therapy. |
  | Low-Density Lipoprotein Cholesterol (LDL-C) Level | `ldl_cholesterol` | Float ≥ 0 | - mmol/L
- mg/dL | Enter untreated lipid values in the absence of lipid lowering therapy. |
  | Triglycerides (TAG) Level | `tag` | Float ≥ 0 | - mmol/L
- mg/dL | Enter untreated lipid values in the absence of lipid lowering therapy. |
  | Lipoprotein(a) (Lp(a)) Level | `lp_a` | Float ≥ 0 | - mmol/L
- mg/dL | Enter untreated lipid values in the absence of lipid lowering therapy. |
  | Body Mass Index | `bmi` | 0 ≤ Float ≤ 50 | kg/m2 | Enter BMI at examination. BMI is calculated by dividing a person’s body weight in kilograms by the square of their height in meters.

The model automatically converts the absolute value of BMI to BMI Z-score using the British 1990 reference data with the LMS Growth which is then used in the calculation. |

- Form has no title.
- Form has two column layout.
- For number input fields with multiple units, allow input of the numeric value and the input of the unit as a dropdown.
- Name of each field should fit in a single line.
- Each field to have a validation of the input.
- All values are optional and can be missing.
- Each value has a tooltip. Tooltip to be wide and left aligned. Tooltip text should not include any html tags or any special rendering, however please do retain the new line characters.
- The form should be submitted as the user is filling out the form.
- Add a reset button that unsets all of the form input fields above <results>.

<result>

- Result should be displayed within the same div as the form just with special highlights written: Likelihood of FH: <probability>.
- If a value is outside of the range, the result displays: “Invalid input <field-name>”

</result>

</form>

<calculator-ml-fh-peds>

- Write a JavaScript function `calculateMLFHPEDS`.
- Returns a randomly generated number between 0.0 and 1.0.

</calculator-ml-fh-peds>

<calculator-fh-peds>

- Write a JavaScript function `calculateFHPEDS`.
- Below are the criteria. Encode this logic into the function with if-else statements and returning the sum of points.


| Input Field         | Points | Criteria                                   |
| --------------------- | -------- | -------------------------------------------- |
| **LDL-C**           | 14     | 6.5 mmol/L < ldl_cholesterol               |
|                     | 12     | 4.8 mmol/L < ldl_cholesterol ≤ 6.5 mmol/L |
|                     | 8      | 3.8 mmol/L < ldl_cholesterol ≤ 4.8 mmol/L |
|                     | 4      | 3.0 mmol/L < ldl_cholesterol ≤ 3.8 mmol/L |
| **HDL-C**           | -2     | 1.4 mmol/L < hdl_cholesterol ≤ 2.2 mmol/L |
|                     | -4     | 2.2 mmol/L < hdl_cholesterol               |
| **TAG**             | -2     | 2.0 mmol/L < tag ≤ 3.5 mmol/L             |
|                     | -4     | 3.5 mmol/L < tag ≤ 4.5 mmol/L             |
|                     | -6     | 4.5 mmol/L < tag                           |
| **Body Mass Index** | -2     | bmi > 1.645                                |

</calculator-fh-peds>

<about.html>

<introduction>

<title>**New Diagnostic Scores – FH-PeDS and ML-FH-PeDS**</title>

We developed novel clinical diagnostic scores to aid physicians in identifying children and adolescents with familial hypercholesterolemia with the use of real-world clinical data and without the use of genetic testing. The first is a diagnostic tool based on **machine learning model (ML-FH-PeDS)** and the second **a semi-quantitative clinical scoring system (FH-PeDS)**, similar to traditional diagnostic scores.

</introduction>

<background>

<title>**Background**</title>

Familial hypercholesterolemia (FH) is an autosomal codominant disorder of lipoprotein metabolism, leading to accelerated atherosclerosis and increased risk for premature cardiovascular disease (CVD). Early identification of individuals with FH, combined with timely initiation of lipid-lowering therapy, significantly reduces the risk of CVD. Although genetic testing is the gold standard for establishing diagnosis, most countries primarily rely on clinical diagnostic tools, largely due to cost considerations and limited access to genetic testing.

To facilitate the identification of children at risk, we developed a novel clinical FH diagnostic scoring system **Familial Hypercholesterolemia Pediatric Diagnostic Score (FH-PeDS)** that has been validated in pediatric population using real-world clinical data.

The diagnostic scoring system was developed on two pediatric cohorts of children with hypercholesterolemia who  were identified through a nationwide universal screening program for FH in Slovenia and from the Portuguese FH Study, thus FH-PeDS should primarily be used only in individuals detected through a universal screening program or opportunistic testing, on the other hand, its use in the setting of cascade screening has yet to be established.

A key advantage of an efficient diagnostic algorithm is its ability to better identify individuals most likely to have true monogenic FH, guiding confirmatory genetic testing and early treatment initiation. Therefore, we developed two novel diagnostic tools:

- **a semi-quantitative clinical scoring system (FH-PeDS) and**
- **a Machine Learning model (ML-FH-PeDS).**

Since diagnostic scores are primarily used as confirmatory tests, the ML-FH-PeDS model’s specificity is set arbitrarily at 98 %.

<background>

<references>

<title>**References**</title>

- Kafol J, Miranda B, Sikonja R, Sikonja J, Wiegman A, Medeiros AM, Alves AC, Freiberger T, Hutten BA, Mlinaric M, Battelino T; FH-PeDS Collaborators; Humphries SE, Bourbon M, Groselj U. Proposal of a Familial Hypercholesterolemia Pediatric Diagnostic Score (FH-PeDS). Eur J Prev Cardiol. 2025 Jun 20:zwaf352. doi: 10.1093/eurjpc/zwaf352.

</references>

</about.html>
