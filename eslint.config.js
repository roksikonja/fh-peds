'use strict';

const js = require('@eslint/js');
const globals = require('globals');

module.exports = [
  {
    ignores: [
      '**/node_modules/**',
      '**/venv/**',
      '**/.venv/**',
      '**/__pycache__/**',
      'ml-fh-peds/results/**',
      'models/**',
      '**/*.min.js',
    ],
  },
  js.configs.recommended,
  {
    // Browser-side inference code (shared with the Node test harness).
    files: ['public/js/**/*.js'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'script',
      globals: {
        ...globals.browser,
        // Cross-file globals exposed by other website scripts loaded in <script> tags.
        BMI_ZSCORE_TABLE: 'readonly',
        preprocess: 'readonly',
        predict_probability: 'readonly',
        loadModel: 'readonly',
        plot: 'readonly',
      },
    },
    rules: {
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
      eqeqeq: ['error', 'smart'],
    },
  },
  {
    // Node-based test harness (ESM — matches the root package.json `type: module`).
    files: ['ml-fh-peds/tests/**/*.js'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.node,
        // Globals injected into the test harness scope by vm.runInThisContext
        // when public/js/{bmi_zscore_table,preprocessing,model}.js are loaded.
        bmiToZScore: 'readonly',
        BMI_LMS_MALE: 'readonly',
        BMI_LMS_FEMALE: 'readonly',
        formSampleToRawSample: 'readonly',
        CHOL_MGDL_PER_MMOLL: 'readonly',
        TAG_MGDL_PER_MMOLL: 'readonly',
      },
    },
    rules: {
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    },
  },
];
