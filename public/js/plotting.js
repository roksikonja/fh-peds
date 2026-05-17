/* ============================================================
   ML-FH-PeDS  —  Charts (pure Canvas 2D)
   ============================================================ */

'use strict';

/* ── Embedded curve data — run 20260331_220132 ───────────────*/

const CURVE = {
  slo: {
    recall: [0.992063,0.984127,0.968254,0.960317,0.960317,0.952381,0.944444,0.928571,0.928571,0.920635,0.920635,0.912698,0.904762,0.904762,0.888889,0.888889,0.888889,0.888889,0.880952,0.865079,0.849206,0.849206,0.849206,0.833333,0.809524,0.801587,0.793651,0.785714,0.761905,0.753968,0.730159,0.722222,0.714286,0.714286,0.706349,0.706349,0.698413,0.690476,0.68254,0.674603,0.65873,0.65873,0.65873,0.642857,0.634921,0.626984,0.626984,0.626984,0.619048,0.619048,0.619048,0.603175,0.595238,0.595238,0.579365,0.571429,0.555556,0.555556,0.547619,0.547619,0.547619,0.531746,0.52381,0.52381,0.52381,0.52381,0.5,0.484127,0.47619,0.468254,0.460317,0.444444,0.428571,0.420635,0.420635,0.412698,0.388889,0.388889,0.388889,0.388889,0.380952,0.373016,0.349206,0.34127,0.34127,0.333333,0.325397,0.31746,0.301587,0.277778,0.246032,0.222222,0.214286,0.206349,0.190476,0.18254,0.150794,0.134921,0.126984,0.0],
    prec:   [0.264271,0.292453,0.323607,0.342776,0.374613,0.39604,0.419014,0.43985,0.466135,0.47541,0.493617,0.502183,0.525346,0.535211,0.541063,0.546341,0.551724,0.571429,0.587302,0.598901,0.601124,0.601124,0.601124,0.6,0.607143,0.612121,0.621118,0.626582,0.627451,0.641892,0.647887,0.65942,0.671642,0.692308,0.700787,0.706349,0.709677,0.713115,0.722689,0.726496,0.72807,0.747748,0.761468,0.757009,0.754717,0.752381,0.759615,0.759615,0.772277,0.78,0.787879,0.791667,0.797872,0.806452,0.820225,0.818182,0.813953,0.813953,0.821429,0.831325,0.831325,0.8375,0.835443,0.835443,0.846154,0.846154,0.84,0.835616,0.84507,0.855072,0.852941,0.861538,0.857143,0.854839,0.854839,0.881356,0.875,0.875,0.875,0.875,0.872727,0.886792,0.88,0.877551,0.877551,0.933333,0.931818,0.952381,0.95,0.945946,0.939394,0.933333,0.964286,0.962963,0.96,0.958333,0.95,0.944444,0.941176,null],
    spec98_idx: 75, // threshold=0.76 → specificity=0.9827
  },
  por: {
    recall: [1.0,1.0,1.0,1.0,1.0,1.0,0.994048,0.994048,0.994048,0.994048,0.988095,0.988095,0.988095,0.988095,0.988095,0.988095,0.988095,0.982143,0.97619,0.97619,0.97619,0.97619,0.970238,0.970238,0.970238,0.964286,0.964286,0.964286,0.958333,0.958333,0.958333,0.958333,0.958333,0.958333,0.958333,0.952381,0.952381,0.952381,0.952381,0.952381,0.952381,0.946429,0.934524,0.934524,0.934524,0.928571,0.928571,0.928571,0.922619,0.922619,0.922619,0.922619,0.922619,0.916667,0.910714,0.89881,0.89881,0.892857,0.886905,0.886905,0.880952,0.875,0.869048,0.869048,0.869048,0.863095,0.857143,0.85119,0.839286,0.833333,0.827381,0.809524,0.791667,0.791667,0.791667,0.779762,0.761905,0.761905,0.72619,0.714286,0.708333,0.708333,0.702381,0.690476,0.678571,0.678571,0.678571,0.672619,0.64881,0.625,0.583333,0.571429,0.547619,0.517857,0.5,0.470238,0.428571,0.363095,0.285714,0.0],
    prec:   [0.495575,0.497041,0.498516,0.501493,0.507553,0.510638,0.513846,0.517028,0.520249,0.521875,0.523659,0.525316,0.526984,0.532051,0.538961,0.542484,0.547855,0.548173,0.550336,0.554054,0.559727,0.559727,0.558219,0.562069,0.564014,0.5625,0.56446,0.566434,0.570922,0.572954,0.583333,0.587591,0.594096,0.600746,0.600746,0.606061,0.610687,0.610687,0.615385,0.615385,0.622568,0.623529,0.630522,0.635628,0.640816,0.641975,0.644628,0.647303,0.651261,0.654008,0.662393,0.670996,0.673913,0.675439,0.68,0.683258,0.689498,0.694444,0.693023,0.709524,0.718447,0.720588,0.722772,0.726368,0.726368,0.732323,0.734694,0.737113,0.742105,0.76087,0.763736,0.768362,0.764368,0.773256,0.796407,0.793939,0.8,0.810127,0.807947,0.816327,0.82069,0.832168,0.842857,0.846715,0.844444,0.857143,0.870229,0.875969,0.879032,0.875,0.882883,0.897196,0.901961,0.896907,0.913043,0.929412,0.935065,0.953125,0.96,null],
    spec98_idx: 97, // threshold=0.98 → specificity=0.9826
  },
};

const WEIGHTS_CHART = [
  { name: 'LDL-C',                 val:  1.355290 },
  { name: 'Total Cholesterol',     val:  1.147028 },
  { name: 'FH High Chol. (3°)',    val:  0.643903 },
  { name: 'FH High Chol. (1°)',    val:  0.348735 },
  { name: 'Sex',                   val:  0.321438 },
  { name: 'Arcus Cornealis',       val:  0.241316 },
  { name: 'FH Premature CAD (2°)', val:  0.128122 },
  { name: 'FH Premature CAD (1°)', val:  0.052265 },
  { name: 'FH Premature CAD (3°)', val:  0.034277 },
  { name: 'FH PAD/CVI (1°)',       val:  0.022371 },
  { name: 'FH High Chol. (2°)',    val:  0.020783 },
  { name: 'FH PAD/CVI (3°)',       val:  0.009784 },
  { name: 'Age',                   val:  0.006678 },
  { name: 'BMI z-score',           val: -0.057086 },
  { name: 'Xanthoma/Xanthelasma',  val: -0.210880 },
  { name: 'FH PAD/CVI (2°)',       val: -0.364994 },
  { name: 'Lp(a)',                 val: -0.402892 },
  { name: 'Triglycerides (TAG)',   val: -0.536108 },
  { name: 'HDL-C',                 val: -0.876071 },
];

function drawPRChart() {
  const canvas = document.getElementById('prChart');
  if (!canvas) return;

  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.offsetWidth || 700;
  const H   = Math.round(W * 0.52);
  canvas.width        = W * dpr;
  canvas.height       = H * dpr;
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const PAD = { top: 20, right: 20, bottom: 44, left: 48 };
  const pw  = W - PAD.left - PAD.right;
  const ph  = H - PAD.top  - PAD.bottom;

  const toX = r => PAD.left + r * pw;
  const toY = p => PAD.top  + (1 - p) * ph;

  // Background
  ctx.fillStyle = '#fafbff';
  ctx.beginPath();
  ctx.roundRect(PAD.left, PAD.top, pw, ph, 4);
  ctx.fill();

  // Grid
  ctx.strokeStyle = '#e8eaf0';
  ctx.lineWidth = 0.8;
  for (let i = 0; i <= 10; i++) {
    ctx.beginPath(); ctx.moveTo(PAD.left + i * pw / 10, PAD.top);  ctx.lineTo(PAD.left + i * pw / 10, PAD.top + ph);  ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD.left, PAD.top + i * ph / 10); ctx.lineTo(PAD.left + pw, PAD.top + i * ph / 10); ctx.stroke();
  }

  // Axis ticks
  ctx.font      = '9.5px Raleway, sans-serif';
  ctx.fillStyle = '#8892aa';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    ctx.fillText(v.toFixed(1), toX(v), PAD.top + ph + 13);
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(1), PAD.left - 5, toY(v) + 3.5);
    ctx.textAlign = 'center';
  }

  // Axis labels
  ctx.fillStyle = '#64708a';
  ctx.font      = '10.5px Raleway, sans-serif';
  ctx.fillText('Recall', PAD.left + pw / 2, H - 5);
  ctx.save();
  ctx.translate(11, PAD.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Precision', 0, 0);
  ctx.restore();

  function drawCurve(cohort, color, spec98idx) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.lineJoin    = 'round';
    let started = false;
    for (let i = 0; i < cohort.recall.length; i++) {
      if (cohort.prec[i] === null) continue;
      if (!started) { ctx.moveTo(toX(cohort.recall[i]), toY(cohort.prec[i])); started = true; }
      else          { ctx.lineTo(toX(cohort.recall[i]), toY(cohort.prec[i])); }
    }
    ctx.stroke();

    // 98 % specificity marker
    const r98 = cohort.recall[spec98idx];
    const p98 = cohort.prec[spec98idx];
    if (p98 !== null) {
      ctx.beginPath();
      ctx.arc(toX(r98), toY(p98), 4.5, 0, Math.PI * 2);
      ctx.fillStyle   = color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth   = 1.5;
      ctx.stroke();
    }
  }

  drawCurve(CURVE.slo, '#36478D', CURVE.slo.spec98_idx);
  drawCurve(CURVE.por, '#e07b39', CURVE.por.spec98_idx);

  // Legend
  const ly = PAD.top + 14;
  const items = [
    { label: 'Slovenia (internal)', color: '#36478D', line: true  },
    { label: 'Portugal (external)', color: '#e07b39', line: true  },
    { label: '@98% specificity',    color: '#555',    line: false },
  ];
  ctx.font = '10px Raleway, sans-serif';
  let lx = PAD.left + 6;
  items.forEach(item => {
    ctx.fillStyle = item.color;
    if (item.line) {
      ctx.fillRect(lx, ly - 4.5, 14, 2.5);
    } else {
      ctx.beginPath(); ctx.arc(lx + 7, ly - 3, 3.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = '#333';
    ctx.textAlign = 'left';
    ctx.fillText(item.label, lx + 19, ly + 0.5);
    lx += ctx.measureText(item.label).width + 38;
  });
}

function drawWeightsChart() {
  const canvas = document.getElementById('weightsChart');
  if (!canvas) return;

  const dpr   = window.devicePixelRatio || 1;
  const BAR_H = 17;
  const GAP   = 5;
  const PAD   = { top: 14, right: 20, bottom: 32, left: 168 };
  const W     = canvas.offsetWidth || 700;
  const H     = PAD.top + WEIGHTS_CHART.length * (BAR_H + GAP) - GAP + PAD.bottom;

  canvas.width        = W * dpr;
  canvas.height       = H * dpr;
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const pw    = W - PAD.left - PAD.right;
  const maxAbs = Math.max(...WEIGHTS_CHART.map(d => Math.abs(d.val)));
  const scale  = (pw / 2) / (maxAbs * 1.1);
  const zeroX  = PAD.left + pw / 2;
  const plotH  = H - PAD.top - PAD.bottom;

  // Background
  ctx.fillStyle = '#fafbff';
  ctx.beginPath();
  ctx.roundRect(PAD.left, PAD.top, pw, plotH, 4);
  ctx.fill();

  // Grid + ticks
  const ticks = [-1.0, -0.5, 0, 0.5, 1.0];
  ctx.font      = '9.5px Raleway, sans-serif';
  ctx.fillStyle = '#8892aa';
  ctx.textAlign = 'center';
  ticks.forEach(t => {
    const x = zeroX + t * scale;
    ctx.strokeStyle = '#e8eaf0'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + plotH); ctx.stroke();
    ctx.fillText(t.toFixed(1), x, H - PAD.bottom + 13);
  });

  // Zero line
  ctx.strokeStyle = '#c8ccdc'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(zeroX, PAD.top); ctx.lineTo(zeroX, PAD.top + plotH); ctx.stroke();

  // Axis label
  ctx.fillStyle = '#64708a'; ctx.font = '10.5px Raleway, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Coefficient', zeroX, H - 3);

  // Bars
  WEIGHTS_CHART.forEach((d, i) => {
    const y    = PAD.top + i * (BAR_H + GAP);
    const barW = Math.abs(d.val) * scale;
    const x    = d.val >= 0 ? zeroX : zeroX - barW;
    const col  = d.val >= 0 ? '#36478D' : '#c0392b';

    // Bar
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.roundRect(x, y, Math.max(barW, 1), BAR_H, 2);
    ctx.fill();

    // Label
    ctx.fillStyle  = '#333';
    ctx.font       = '10px Raleway, sans-serif';
    ctx.textAlign  = 'right';
    ctx.fillText(d.name, PAD.left - 7, y + BAR_H * 0.71);

    // Value
    const inside = barW > 32;
    ctx.fillStyle = inside ? '#fff' : col;
    ctx.textAlign = d.val >= 0 ? 'left' : 'right';
    ctx.fillText(d.val.toFixed(3), d.val >= 0 ? x + barW + 3 : x - 3, y + BAR_H * 0.71);
  });
}
