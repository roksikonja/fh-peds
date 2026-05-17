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
      },
    },
    rules: {
      // These classic scripts share a global namespace: every top-level
      // `const`/`function` is intentionally exposed for sibling files to
      // consume. Cross-file references therefore look "undefined" inside
      // one file but are valid at runtime. Disable both checks rather than
      // hand-maintain a globals list that must mirror every top-level decl.
      'no-undef': 'off',
      // Top-level declarations look "unused" within a single file but are
      // consumed by other scripts in the same <script> chain.
      'no-unused-vars': 'off',
      eqeqeq: ['error', 'smart'],
    },
  },
  {
    // Node-based test harness (ESM — matches the root package.json
    // `type: module`; the test files use `import`/`export` syntax).
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
