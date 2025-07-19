
## Installation

- Inference

    ```bash
    uv venv --python 3.10 venv
    source venv/bin/activate
    uv pip install -r requirements.txt
    ```

- Development

    ```bash
    uv venv --python 3.10 venv
    source venv/bin/activate

    uv pip install --upgrade pip wheel setuptools
    uv pip install pandas numpy scikit-learn openpyxl jupyter matplotlib
    ```
