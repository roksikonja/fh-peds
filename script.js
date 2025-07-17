// Modern static site JS for collapsible, form validation, live results, tooltips

document.addEventListener('DOMContentLoaded', function() {
  // Tab switching logic
  const tabML = document.getElementById('tab-ml');
  const tabFH = document.getElementById('tab-fh');
  const formML = document.getElementById('form-ml');
  const formFH = document.getElementById('form-fh');
  tabML.addEventListener('click', function() {
    tabML.classList.add('active');
    tabFH.classList.remove('active');
    formML.classList.add('active');
    formFH.classList.remove('active');
  });
  tabFH.addEventListener('click', function() {
    tabFH.classList.add('active');
    tabML.classList.remove('active');
    formFH.classList.add('active');
    formML.classList.remove('active');
  });

  // Collapsible background
  const backgroundToggle = document.getElementById('backgroundToggle');
  const backgroundContent = document.getElementById('backgroundContent');
  backgroundToggle.classList.remove('active');
  backgroundContent.classList.remove('active');
  backgroundContent.style.maxHeight = '0';
  backgroundToggle.addEventListener('click', function() {
    this.classList.toggle('active');
    backgroundContent.classList.toggle('active');
    if (backgroundContent.classList.contains('active')) {
      backgroundContent.style.maxHeight = backgroundContent.scrollHeight + 'px';
    } else {
      backgroundContent.style.maxHeight = '0';
    }
  });

  // Tooltips: make icons focusable for keyboard users
  document.querySelectorAll('.tooltip-icon').forEach(icon => {
    icon.setAttribute('tabindex', '0');
    icon.addEventListener('focus', function() {
      this.classList.add('show-tooltip');
    });
    icon.addEventListener('blur', function() {
      this.classList.remove('show-tooltip');
    });
  });

  // ML-FH-PeDS form logic
  const formMLForm = document.getElementById('diagnosticFormML');
  const resultsTextML = document.getElementById('resultsTextML');
  const formInputsML = formMLForm.querySelectorAll('input, select');
  function validateInput(input) {
    if (input.value === '') {
      input.classList.remove('invalid');
      return true;
    }
    if (input.type === 'number') {
      if (!input.validity.valid) {
        input.classList.add('invalid');
        return false;
      } else {
        input.classList.remove('invalid');
        return true;
      }
    }
    input.classList.remove('invalid');
    return true;
  }
  function handleFormSubmissionML() {
    let hasAnyData = false;
    let allValid = true;
    const data = {};
    formInputsML.forEach(input => {
      if (input.value !== '') {
        hasAnyData = true;
        if (!validateInput(input)) {
          allValid = false;
        }
        data[input.name] = input.value;
      } else {
        input.classList.remove('invalid');
      }
    });
    [
      ['totalCholesterol', 'totalCholesterolUnit'],
      ['hdlCholesterol', 'hdlCholesterolUnit'],
      ['ldlCholesterol', 'ldlCholesterolUnit'],
      ['triglycerides', 'triglyceridesUnit'],
      ['lipoproteinA', 'lipoproteinAUnit']
    ].forEach(([field, unitField]) => {
      const unitSelect = document.getElementById(unitField+'-ml');
      if (unitSelect) {
        data[unitField] = unitSelect.value;
      }
    });
    if (hasAnyData && allValid) {
      const result = predictML(data);
      displayResultsML(result);
    } else {
      resultsTextML.style.display = 'none';
    }
  }
  formInputsML.forEach(input => {
    input.addEventListener('input', function() {
      clearTimeout(this.submitTimeout);
      this.submitTimeout = setTimeout(() => {
        handleFormSubmissionML();
      }, 400);
    });
  });
  function predictML(data) {
    return 1.0;
  }
  function displayResultsML(result) {
    resultsTextML.textContent = `Likelihood of FH: ${result}`;
    resultsTextML.style.display = 'block';
    resultsTextML.style.opacity = 0;
    setTimeout(() => {
      resultsTextML.style.transition = 'opacity 0.4s';
      resultsTextML.style.opacity = 1;
    }, 10);
  }
  handleFormSubmissionML();

  // FH-PeDS form logic
  const formFHForm = document.getElementById('diagnosticFormFH');
  const resultsTextFH = document.getElementById('resultsTextFH');
  const formInputsFH = formFHForm.querySelectorAll('input, select');
  function handleFormSubmissionFH() {
    let hasAnyData = false;
    let allValid = true;
    const data = {};
    formInputsFH.forEach(input => {
      if (input.value !== '') {
        hasAnyData = true;
        if (!validateInput(input)) {
          allValid = false;
        }
        data[input.name] = input.value;
      } else {
        input.classList.remove('invalid');
      }
    });
    if (hasAnyData && allValid) {
      const result = predictFH(data);
      displayResultsFH(result);
    } else {
      resultsTextFH.style.display = 'none';
    }
  }
  formInputsFH.forEach(input => {
    input.addEventListener('input', function() {
      clearTimeout(this.submitTimeout);
      this.submitTimeout = setTimeout(() => {
        handleFormSubmissionFH();
      }, 400);
    });
  });
  function predictFH(data) {
    return 1.0;
  }
  function displayResultsFH(result) {
    resultsTextFH.textContent = `Likelihood of FH: ${result}`;
    resultsTextFH.style.display = 'block';
    resultsTextFH.style.opacity = 0;
    setTimeout(() => {
      resultsTextFH.style.transition = 'opacity 0.4s';
      resultsTextFH.style.opacity = 1;
    }, 10);
  }
  handleFormSubmissionFH();
}); 