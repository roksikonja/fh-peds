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
      const value = parseFloat(input.value);
      const min = parseFloat(input.min);
      const max = parseFloat(input.max);
      
      if (isNaN(value) || value < min || (max && value > max)) {
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
    let invalidField = null;
    
    formInputsML.forEach(input => {
      if (input.value !== '') {
        hasAnyData = true;
        if (!validateInput(input)) {
          allValid = false;
          if (!invalidField) {
            invalidField = input.name;
          }
        }
        data[input.name] = input.value;
      } else {
        input.classList.remove('invalid');
      }
    });
    
    // Handle unit fields
    [
      ['total_cholesterol', 'total_cholesterol_unit'],
      ['hdl_cholesterol', 'hdl_cholesterol_unit'],
      ['ldl_cholesterol', 'ldl_cholesterol_unit'],
      ['tag', 'tag_unit'],
      ['lp_a', 'lp_a_unit']
    ].forEach(([field, unitField]) => {
      const unitSelect = document.getElementById(unitField+'-ml');
      if (unitSelect) {
        data[unitField] = unitSelect.value;
      }
    });
    
    if (hasAnyData && allValid) {
      const result = calculateMLFHPEDS(data);
      displayResultsML(result);
    } else if (invalidField) {
      displayResultsML(`Invalid input ${invalidField}`);
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
  
  function calculateMLFHPEDS(data) {
    // Returns a randomly generated number between 0.0 and 1.0
    return Math.random().toFixed(3);
  }
  
  function displayResultsML(result) {
    if (result.startsWith('Invalid input')) {
      resultsTextML.textContent = result;
      resultsTextML.style.background = '#ffe6e6';
      resultsTextML.style.borderLeftColor = '#dc3545';
      resultsTextML.style.color = '#dc3545';
    } else {
      resultsTextML.textContent = `Likelihood of FH: ${result}`;
      resultsTextML.style.background = '#e8f4fd';
      resultsTextML.style.borderLeftColor = '#36478D';
      resultsTextML.style.color = '#36478D';
    }
    resultsTextML.style.display = 'block';
    resultsTextML.style.opacity = 0;
    setTimeout(() => {
      resultsTextML.style.transition = 'opacity 0.4s';
      resultsTextML.style.opacity = 1;
    }, 10);
  }
  
  handleFormSubmissionML();

  // Reset button for ML-FH-PeDS form
  const resetMLBtn = document.getElementById('resetML');
  resetMLBtn.addEventListener('click', function() {
    formMLForm.reset();
    formInputsML.forEach(input => {
      input.classList.remove('invalid');
    });
    resultsTextML.style.display = 'none';
  });

  // FH-PeDS form logic
  const formFHForm = document.getElementById('diagnosticFormFH');
  const resultsTextFH = document.getElementById('resultsTextFH');
  const formInputsFH = formFHForm.querySelectorAll('input, select');
  
  function handleFormSubmissionFH() {
    let hasAnyData = false;
    let allValid = true;
    const data = {};
    let invalidField = null;
    
    formInputsFH.forEach(input => {
      if (input.value !== '') {
        hasAnyData = true;
        if (!validateInput(input)) {
          allValid = false;
          if (!invalidField) {
            invalidField = input.name;
          }
        }
        data[input.name] = input.value;
      } else {
        input.classList.remove('invalid');
      }
    });
    
    // Handle unit fields
    [
      ['total_cholesterol', 'total_cholesterol_unit'],
      ['hdl_cholesterol', 'hdl_cholesterol_unit'],
      ['ldl_cholesterol', 'ldl_cholesterol_unit'],
      ['tag', 'tag_unit'],
      ['lp_a', 'lp_a_unit']
    ].forEach(([field, unitField]) => {
      const unitSelect = document.getElementById(unitField+'-fh');
      if (unitSelect) {
        data[unitField] = unitSelect.value;
      }
    });
    
    if (hasAnyData && allValid) {
      const result = calculateFHPEDS(data);
      displayResultsFH(result);
    } else if (invalidField) {
      displayResultsFH(`Invalid input ${invalidField}`);
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
  
  function calculateFHPEDS(data) {
    let totalPoints = 0;
    
    // Convert units to mmol/L for calculations
    const ldlCholesterol = convertToMmolL(data.ldl_cholesterol, data.ldl_cholesterol_unit, 'ldl_cholesterol');
    const hdlCholesterol = convertToMmolL(data.hdl_cholesterol, data.hdl_cholesterol_unit, 'hdl_cholesterol');
    const tag = convertToMmolL(data.tag, data.tag_unit, 'tag');
    const bmi = parseFloat(data.bmi);
    
    // LDL-C scoring
    if (ldlCholesterol > 6.5) {
      totalPoints += 14;
    } else if (ldlCholesterol > 4.8) {
      totalPoints += 12;
    } else if (ldlCholesterol > 3.8) {
      totalPoints += 8;
    } else if (ldlCholesterol > 3.0) {
      totalPoints += 4;
    }
    
    // HDL-C scoring
    if (hdlCholesterol > 1.4 && hdlCholesterol <= 2.2) {
      totalPoints -= 2;
    } else if (hdlCholesterol > 2.2) {
      totalPoints -= 4;
    }
    
    // TAG scoring
    if (tag > 2.0 && tag <= 3.5) {
      totalPoints -= 2;
    } else if (tag > 3.5 && tag <= 4.5) {
      totalPoints -= 4;
    } else if (tag > 4.5) {
      totalPoints -= 6;
    }
    
    // BMI scoring (BMI Z-score > 1.645)
    if (bmi > 1.645) {
      totalPoints -= 2;
    }
    
    return totalPoints;
  }
  
  function convertToMmolL(value, unit, fieldName) {
    if (!value || !unit) return null;
    const numValue = parseFloat(value);
    if (unit === 'mg/dL') {
      // Convert mg/dL to mmol/L
      // For TC, HDL-C, LDL-C: 1 mmol/L = 38.67 mg/dL
      // For TAG: 1 mmol/L = 88.57 mg/dL
      // For Lp(a): 1 mmol/L = 38.67 mg/dL (assuming same as cholesterol)
      if (fieldName === 'tag') {
        return numValue / 88.57;
      } else {
        return numValue / 38.67;
      }
    }
    return numValue;
  }
  
  function displayResultsFH(result) {
    if (typeof result === 'string' && result.startsWith('Invalid input')) {
      resultsTextFH.textContent = result;
      resultsTextFH.style.background = '#ffe6e6';
      resultsTextFH.style.borderLeftColor = '#dc3545';
      resultsTextFH.style.color = '#dc3545';
    } else {
      resultsTextFH.textContent = `Likelihood of FH: ${result}`;
      resultsTextFH.style.background = '#e8f4fd';
      resultsTextFH.style.borderLeftColor = '#36478D';
      resultsTextFH.style.color = '#36478D';
    }
    resultsTextFH.style.display = 'block';
    resultsTextFH.style.opacity = 0;
    setTimeout(() => {
      resultsTextFH.style.transition = 'opacity 0.4s';
      resultsTextFH.style.opacity = 1;
    }, 10);
  }
  
  handleFormSubmissionFH();

  // Reset button for FH-PeDS form
  const resetFHBtn = document.getElementById('resetFH');
  resetFHBtn.addEventListener('click', function() {
    formFHForm.reset();
    formInputsFH.forEach(input => {
      input.classList.remove('invalid');
    });
    resultsTextFH.style.display = 'none';
  });
}); 