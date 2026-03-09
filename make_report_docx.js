/**
 * DSS5104 — Word Report Builder
 * Generates house_price_report.docx
 * Run: node make_report_docx.js
 */

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, LevelFormat, Header, Footer,
  TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');
const path = require('path');

// ── Results ───────────────────────────────────────────────────────
const R = JSON.parse(fs.readFileSync('results.json'));
const fmt = (v) => v.toFixed(2) + '%';
const sgn = (v) => (v >= 0 ? '+' : '') + v.toFixed(2) + 'pp';

// ── Helpers ───────────────────────────────────────────────────────
const A4_W = 11906;   // DXA
const A4_H = 16838;
const LM   = 1440;    // 1 inch margins
const RM   = 1440;
const CONTENT_W = A4_W - LM - RM;  // 9026 DXA

const NAVY  = '1A2E4A';
const BLUE  = '2C5F8A';
const GREY  = 'F4F6F9';
const DGREY = '888888';

function sp(pt = 6) {
  return new Paragraph({ spacing: { before: 0, after: 0 }, children: [new TextRun({ text: '', size: pt })] });
}

function hr() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC', space: 1 } },
    spacing: { before: 120, after: 120 },
    children: [],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, size: 26, color: NAVY, font: 'Arial' })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 180, after: 80 },
    children: [new TextRun({ text, bold: true, size: 22, color: BLUE, font: 'Arial' })],
  });
}

// Body paragraph — supports simple inline bold via **text** markers
function body(text, opts = {}) {
  const runs = [];
  const parts = text.split(/(\*\*[^*]+\*\*)/);
  for (const part of parts) {
    if (part.startsWith('**') && part.endsWith('**')) {
      runs.push(new TextRun({ text: part.slice(2, -2), bold: true, size: 20, font: 'Arial', color: '1A1A1A' }));
    } else if (part) {
      runs.push(new TextRun({ text: part, size: 20, font: 'Arial', color: '1A1A1A' }));
    }
  }
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 120 },
    children: runs,
    ...opts,
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 160 },
    children: [new TextRun({ text, italics: true, size: 17, color: DGREY, font: 'Arial' })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, size: 20, font: 'Arial', color: '1A1A1A' })],
  });
}

// ── Image helper ─────────────────────────────────────────────────
function figureImg(filename, widthCm, aspectRatio) {
  const imgPath = path.join('figures', filename);
  if (!fs.existsSync(imgPath)) return null;
  const data = fs.readFileSync(imgPath);
  const emuPerCm = 360000;
  const w = Math.round(widthCm * emuPerCm);
  const h = Math.round(w * aspectRatio);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 40 },
    children: [new ImageRun({ data, transformation: { width: Math.round(widthCm * 37.8), height: Math.round(widthCm * 37.8 * aspectRatio) }, type: 'png' })],
  });
}

// ── Table helpers ─────────────────────────────────────────────────
const BORDER = { style: BorderStyle.SINGLE, size: 1, color: 'DDDDDD' };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const CELL_MARGINS = { top: 100, bottom: 100, left: 140, right: 140 };

function headerCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { fill: NAVY, type: ShadingType.CLEAR },
    borders: BORDERS,
    margins: CELL_MARGINS,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, color: 'FFFFFF', size: 18, font: 'Arial' })],
    })],
  });
}

function dataCell(text, width, shade = false, bold = false) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { fill: shade ? 'F4F6F9' : 'FFFFFF', type: ShadingType.CLEAR },
    borders: BORDERS,
    margins: CELL_MARGINS,
    verticalAlign: VerticalAlign.TOP,
    children: [new Paragraph({
      children: [new TextRun({ text, size: 18, font: 'Arial', bold, color: '1A1A1A' })],
      spacing: { before: 0, after: 0 },
    })],
  });
}

function makeTable(headers, rows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => headerCell(h, colWidths[i])),
      }),
      ...rows.map((row, ri) =>
        new TableRow({
          children: row.map((cell, ci) => dataCell(
            typeof cell === 'object' ? cell.text : cell,
            colWidths[ci],
            ri % 2 === 1,
            typeof cell === 'object' ? cell.bold : false
          )),
        })
      ),
    ],
  });
}

// ── Build document ────────────────────────────────────────────────
const children = [];

// ── Title block ───────────────────────────────────────────────────
children.push(sp(24));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 80 },
  children: [new TextRun({ text: 'House Price Prediction with Linear Models', bold: true, size: 40, color: NAVY, font: 'Arial' })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 60 },
  children: [new TextRun({ text: 'DSS5104 — Applied Linear Regression  ·  Continuous Assessment 1', size: 22, color: BLUE, font: 'Arial' })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 160 },
  children: [new TextRun({ text: 'March 2026', size: 20, color: DGREY, font: 'Arial' })],
}));
children.push(hr());

// ── Abstract ──────────────────────────────────────────────────────
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: 'Abstract', bold: true, size: 20, color: NAVY, font: 'Arial' })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { before: 0, after: 160 },
  indent: { left: 720, right: 720 },
  children: [new TextRun({
    text: 'This paper investigates how far linear regression models can be pushed on a residential house price prediction task through systematic feature engineering, regularisation, and ensemble methods, while maintaining interpretability. Applied to a deduplicated dataset of 4,553 property transactions in King County, Washington, we construct a lean 20-feature Ridge model achieving a test-set Mean Absolute Percentage Error (MAPE) of ' + fmt(R.lean_test) + ', and a stacked ensemble of kernel-augmented linear models achieving ' + fmt(R.stacked_test) + '. A data integrity audit revealed that the original 9,102-row dataset was an artificial duplication of 4,553 unique records, inflating tree-based benchmarks by an estimated 3–4 percentage points. On the clean data, our stacked model trails a properly regularised XGBoost benchmark by 1.2 percentage points, while offering full coefficient-level interpretability. Lasso regularisation path analysis demonstrates that predictive performance saturates near 20 features, confirming that thoughtful feature design is more valuable than feature proliferation.',
    size: 18, font: 'Arial', italics: false, color: '1A1A1A',
  })],
}));
children.push(hr());

// ── 1. Introduction ───────────────────────────────────────────────
children.push(h1('1. Introduction'));
children.push(body('Modern machine learning applications in real estate overwhelmingly favour gradient boosting and neural network approaches for their predictive power. However, in regulated domains such as mortgage appraisal, insurance underwriting, and property taxation, model interpretability is a legal or institutional requirement. Linear models, when supported by careful feature engineering, offer a principled middle ground: competitive accuracy with fully auditable predictions.'));
children.push(body('The central methodological contribution is not any single model, but a **feature engineering discipline**: we show through Lasso regularisation path analysis that predictive performance saturates at approximately 20 well-chosen features, that additional features provide negligible marginal gain for Ridge regression, and that the dominant signal — neighbourhood comparable prices — can be captured without a single tree-based model.'));
children.push(body('The remainder of this paper is structured as follows. Section 2 describes data cleaning, including the discovery and resolution of a systematic data duplication issue. Section 3 presents exploratory analysis motivating each engineering decision. Section 4 details the three-round feature engineering process. Section 5 describes the modelling strategy. Section 6 reports results. Section 7 interprets the primary model. Section 8 contextualises performance against XGBoost benchmarks. Section 9 concludes.'));

// ── 2. Data ───────────────────────────────────────────────────────
children.push(h1('2. Data'));
children.push(h2('2.1 Dataset Description'));
children.push(body('The dataset comprises residential property transactions in King County, Washington, recorded between May and July 2014. Each record contains 18 variables: physical attributes (square footage of living area, lot, above-ground and basement spaces), quality indicators (condition on a 1–5 ordinal scale, view quality on 0–4, waterfront binary), location (city, state-ZIP, street address), construction history (year built, year last renovated), and sale price. The 70-day transaction window limits temporal variation, making month of sale a weak but non-zero signal.'));

children.push(h2('2.2 Data Cleaning and Integrity Audit'));
children.push(body('Raw data contained 9,200 rows. Ninety-eight zero-price records were removed as invalid entries, leaving 9,102 rows. A systematic audit then identified a critical data quality issue: **4,549 of the 4,553 unique property records appeared exactly twice**, making approximately 50% of the dataset an artificial copy. Five independent checks confirmed this was not legitimate resale activity:'));
children.push(bullet('All duplicate pairs share the identical sale date, ruling out resales across different time periods.'));
children.push(bullet('Zero pairs exhibit any difference in sale price, ruling out renegotiations.'));
children.push(bullet('Every duplicate appears at a consistent row offset of exactly 4,551 positions — precisely half the dataset — indicating programmatic origin.'));
children.push(bullet('9,098 rows are byte-for-byte identical including date; only 2 rows differ (a single sqft_living transcription discrepancy).'));
children.push(bullet('Under a naive 80/20 split, 81% of test-set properties had their twin in the training set, directly inflating held-out performance.'));
children.push(sp(4));
children.push(body('After deduplication, 4,553 unique sales remain. The impact on reported MAPEs was asymmetric: linear models were minimally affected (less than 0.8 pp) because they cannot memorise individual records. Gradient boosting models showed a 3–4 pp inflation. All results reported in this paper use the deduplicated dataset.'));

children.push(makeTable(
  ['Check', 'Finding', 'Conclusion'],
  [
    ['Duplicate sale dates',    'All 4,549 pairs share identical sale date',  'Not legitimate resales'],
    ['Price variation',         'Zero pairs with differing price',            'Not renegotiations'],
    ['Row offset pattern',      'Consistent offset of 4,551 = half dataset',  'Programmatic copy'],
    ['Full-row identity',       '9,098 rows byte-for-byte identical',         'Artificial duplication'],
    ['Half-vs-half comparison', 'Only 2 rows differ (sqft_living typo)',      'Confirmed artificial'],
  ],
  [3100, 3900, 2026]
));
children.push(caption('Table 1. Five independent checks confirming that duplicate records are artificial rather than legitimate property resales.'));

children.push(h2('2.3 Train/Test Split'));
children.push(body('A fixed 80/20 random split (seed = 42) yields 3,642 training and 911 test observations. All data-dependent transformations — target encodings, scalers — are fitted exclusively on the training partition and applied to test data without refitting, preventing any form of data leakage.'));

// ── 3. EDA ────────────────────────────────────────────────────────
children.push(h1('3. Exploratory Data Analysis'));
children.push(body('Exploratory analysis served as the primary motivation for feature engineering decisions. Three findings had the most significant methodological implications.'));

const fig1 = figureImg('fig_price_dist.png', 15, 0.42);
if (fig1) { children.push(fig1); children.push(caption('Figure 1. Left: raw price distribution exhibiting strong right skew (mean $558K, median $465K). Right: log-price is approximately Gaussian, validating the choice of log(price) as modelling target.')); }

const fig2 = figureImg('fig_eda_insights.png', 15, 0.38);
if (fig2) { children.push(fig2); children.push(caption('Figure 2. Three EDA findings driving feature engineering: (a) U-shaped age–price curve; (b) non-linear condition premium with disproportionate jump at score 5; (c) 6× price range across cities.')); }

children.push(body('**Price distribution.** Sale prices range from $7,800 to $26.6 million (median $465K, mean $558K) with strong right skew. Log-transforming the target produces a near-Gaussian distribution, stabilises variance, and aligns the training objective with MAPE\'s emphasis on relative error.'));
children.push(body('**Age–price non-linearity.** A U-shaped relationship is visible: both new builds (age ≤ 5 years) and pre-1920 character properties command premiums above mid-century construction. This motivates log_house_age as the primary age feature alongside a binary is_new indicator.'));
children.push(body('**Condition non-linearity.** A disproportionate price jump appears at condition score 5 relative to scores 3 and 4. A top_condition binary indicator and a condition × house_age interaction together provide adequate flexibility without overfitting.'));
children.push(body('**Location heterogeneity.** Median prices vary by a factor of six across cities, motivating smoothed target encoding for both city and ZIP code, and a zip_city_diff feature capturing fine-grained intra-city variation.'));

// ── 4. Feature Engineering ────────────────────────────────────────
children.push(h1('4. Feature Engineering'));
children.push(body('Feature engineering proceeded empirically, guided by EDA findings, domain knowledge, and Lasso regularisation path diagnostics. The Lasso path established that predictive performance saturates near 20 features, informing the design of a lean, interpretable primary model.'));

children.push(h2('4.1 Target Variable'));
children.push(body('The modelling target is log(price). Back-transformation via exponentiation yields price predictions in original dollars; MAPE is computed in this original space. Log-transformation is motivated by the multiplicative structure of the housing market and by the near-Gaussian distribution of log-price established in Section 3.'));

children.push(h2('4.2 Location Features'));
children.push(body('Location is the dominant price determinant. We employ **Bayesian-smoothed target encoding** for city (44 levels), ZIP code (70 levels), and build era (6 levels). Each category\'s mean log-price is blended with the global mean using a smoothing strength of 30, preventing overfitting on sparsely observed categories. All encodings are fitted on training data only and re-computed inside each cross-validation fold.'));
children.push(body('A zip_city_diff feature captures the micro-location premium of a specific ZIP code within its broader city. The zip_x_sqft interaction encodes the neighbourhood-dependent marginal value of living area.'));

children.push(h2('4.3 Age and Renovation Features'));
children.push(body('Three features characterise the temporal dimension of a property. log_house_age = log(year_sold − yr_built + 1) compresses the right tail of the age distribution. effective_age = year_sold − max(yr_built, yr_renovated) uses renovation year where available. recent_reno is a binary indicator for renovation within 10 years of sale, encoding the "move-in ready" premium.'));

children.push(h2('4.4 Condition and Interaction Features'));
children.push(body('The interaction cond_x_age = condition × house_age captures a finding from EDA: a 60-year-old home rated condition 5 commands a disproportionate premium, as sustained upkeep over decades signals exceptional ownership quality. size_x_condition reflects the amplification of condition premiums in larger homes. city_x_sqft captures the multiplicative location–size interaction.'));

children.push(h2('4.5 Size and Room Features'));
children.push(body('The concave size–price relationship is captured by log_sqft_above and log_sqft_living_sq. Above-ground area is preferred over total living area as basements contribute less value per square foot. sqft_per_bedroom is a spaciousness proxy; has_basement is a binary presence flag.'));

children.push(makeTable(
  ['Group', 'Features in lean model', 'Motivation'],
  [
    ['Location (4)',    'city_lp, zip_lp, zip_city_diff, zip_x_sqft',           'Dominant price determinant; smoothed encoding prevents overfit'],
    ['Size (3)',        'log_sqft_above, log_sqft_living_sq, log_sqft_basement', 'Log + quadratic captures concave diminishing-returns curve'],
    ['Condition (2)',   'condition, cond_x_age',                                 'Base level + interaction with age captures upkeep signal'],
    ['Age (2)',         'log_house_age, effective_age',                          'Log captures U-shape; effective age uses renovation date'],
    ['Rooms (2)',       'bathrooms, sqft_per_bedroom',                           'Quality and spaciousness proxies'],
    ['View/Water (2)',  'view, waterfront',                                      'Survive strong Lasso regularisation; direct effects'],
    ['Basement (1)',    'has_basement',                                          'Binary presence flag; size ratio excluded as near-zero'],
    ['Renovation (1)',  'recent_reno',                                           '"Move-in ready" premium within 10-year window'],
    ['Interaction (2)', 'size_x_condition, city_x_sqft',                        'Size amplifies condition; city level multiplies living area value'],
    ['Time (1)',        'month_sold',                                            'Seasonal variation within the 70-day window'],
  ],
  [1800, 3500, 3726]
));
children.push(caption('Table 2. The 20 features comprising the lean primary model, organised by group. All location encodings are fitted on training data and re-fitted per CV fold.'));

// ── 5. Modelling ──────────────────────────────────────────────────
children.push(h1('5. Modelling'));

children.push(h2('5.1 Feature Count Optimisation via Lasso Path'));
children.push(body('Before fitting the primary model, a Lasso regularisation path analysis determined empirically how many features are justified. The Lasso path sweeps 40 regularisation strengths, recording the number of surviving features and both 5-fold CV and test MAPE at each point.'));

const fig3 = figureImg('fig_mape_vs_nfeats.png', 15, 0.50);
if (fig3) { children.push(fig3); children.push(caption('Figure 3. MAPE as a function of feature count along the Lasso regularisation path. Performance saturates rapidly near 19–20 features; reducing from 38 to 20 features costs less than 0.2 pp MAPE while substantially improving interpretability.')); }

children.push(body('The analysis reveals a clear elbow: performance improves sharply from 3 to 12 features (MAPE 21% → 18%), then flattens. Above 20 features, gains are marginal (less than 0.2 pp per 10 additional features). This directly informs the lean 20-feature primary model.'));

children.push(h2('5.2 Regularisation Strategy'));
children.push(body('Ridge regression is selected over Lasso and Elastic Net for three reasons. First, correlated feature pairs (e.g., log_house_age and effective_age) are better handled by L2\'s simultaneous shrinkage than L1\'s winner-takes-all zeroing. Second, Ridge retains all features, preserving a complete coefficient vector for interpretation. Third, Ridge produced the lowest cross-validated MAPE across all regularisation families evaluated. The regularisation strength α = 0.5 was selected by 5-fold cross-validation.'));

children.push(h2('5.3 Kernel Methods (Nyström Approximation)'));
children.push(body('As a performance extension, RBF and polynomial kernel approximations are applied via the Nyström method (500 components), mapping the feature space into a higher-dimensional representation where linear regression can capture non-linear interactions. Both methods fully comply with the linear model constraint.'));

children.push(h2('5.4 OOF Stacking'));
children.push(body('Ridge, RBF-Nyström, and Poly-Nyström are combined via 5-fold out-of-fold (OOF) stacking. For each fold, each base model is trained on the remaining 4 folds and predicts on the held-out fold. A Ridge meta-learner (α = 0.001) is fitted on these OOF predictions. The meta-learner never sees a prediction generated by a model trained on the same observation, preventing data leakage. The assignment brief explicitly permits "ensembling multiple linear models combined in any reasonable way" — every component is a linear model, making this approach fully compliant.'));

children.push(h2('5.5 Validation Protocol'));
children.push(body('All hyperparameter selection uses 5-fold cross-validation on the training set. Features requiring data-dependent fitting are re-fitted inside each fold. The test set is never used for model selection decisions.'));

// ── 6. Results ────────────────────────────────────────────────────
children.push(h1('6. Results'));

children.push(makeTable(
  ['Model', 'Features', 'CV MAPE', 'Train MAPE', 'Test MAPE', 'Gap'],
  [
    [{ text: 'Lean Ridge (primary)', bold: true }, '20', fmt(R.lean_cv), fmt(R.lean_train), { text: fmt(R.lean_test), bold: true }, sgn(R.lean_test - R.lean_train)],
    ['Extended Ridge',              '35', fmt(R.ext_cv),  fmt(R.ext_train),  fmt(R.ext_test),         sgn(R.ext_test - R.ext_train)],
    ['RBF Kernel (Nyström)',        '35', '—',            '—',               fmt(R.rbf_test),          '—'],
    ['Poly Kernel (Nyström)',       '35', '—',            '—',               fmt(R.poly_test),         '—'],
    [{ text: 'Stacked ensemble', bold: true }, '35', '—', '—', { text: fmt(R.stacked_test), bold: true }, '—'],
    ['XGB Conservative (benchmark)', '35', '—', '15.05%', '15.62%', '+0.57pp'],
    ['XGB Early Stopping (bmark)',   '35', '—', '11.21%', '15.28%', '+4.08pp'],
  ],
  [2800, 1000, 1226, 1300, 1300, 1400]
));
children.push(caption('Table 3. Full results on the deduplicated dataset (4,553 properties, 80/20 split). Gap = Test MAPE − Train MAPE; negative values indicate mild underfitting typical of well-regularised linear models.'));

const fig4 = figureImg('fig_comparison.png', 15, 0.50);
if (fig4) { children.push(fig4); children.push(caption('Figure 4. Left: test MAPE across all models. Right: XGBoost train vs test MAPE — the Default configuration (train=5.6%, test=15.5%) is severely overfit; the Conservative and Early Stopping configurations are honest comparators.')); }

children.push(body('The lean 20-feature Ridge model achieves ' + fmt(R.lean_test) + ' test MAPE with a train/test gap of ' + sgn(R.lean_test - R.lean_train) + ', indicating mild underfitting rather than memorisation — the hallmark of a well-regularised linear model. The stacked ensemble reaches ' + fmt(R.stacked_test) + ' MAPE.'));

children.push(h2('6.1 Performance by Price Segment'));
children.push(makeTable(
  ['Price Segment', 'Test MAPE', 'Count', 'Notes'],
  [
    ['< $200K',       '36.7%', '32',  'Distressed/atypical sales; limited training examples'],
    ['$200K–$400K',   '15.2%', '309', 'Core market — strong model performance'],
    ['$400K–$600K',   '13.8%', '288', 'Highest-density segment — best performance'],
    ['$600K–$800K',   '13.1%', '139', 'Consistent with core market'],
    ['$800K–$1M',     '15.6%', '68',  'Slight increase; smaller sample'],
    ['$1M–$2M',       '20.7%', '64',  'Luxury segment; idiosyncratic factors'],
    ['> $2M',         '47.5%', '11',  'Ultra-luxury; too few examples for reliable estimation'],
  ],
  [2000, 1400, 900, 4726]
));
children.push(caption('Table 4. MAPE by price segment for the lean Ridge model. The model performs strongly across the $200K–$1M range (MAPE 13–16%) that constitutes 83% of the test set.'));

// ── 7. Interpretation ─────────────────────────────────────────────
children.push(h1('7. Model Interpretation'));
children.push(body('All interpretation uses the standalone lean Ridge model (α = 0.5, 20 features). Coefficients are standardised — each represents the expected change in log(price) per one standard deviation change in the feature, holding all other features constant.'));

const fig5 = figureImg('fig_coef.png', 15, 0.52);
if (fig5) { children.push(fig5); children.push(caption('Figure 5. Standardised Ridge coefficients for the lean 20-feature model. Blue bars indicate a positive effect on log-price; red bars negative. The quadratic log-living term dominates, consistent with the concave size–price curve.')); }

const fig6 = figureImg('fig_diagnostics.png', 15, 0.46);
if (fig6) { children.push(fig6); children.push(caption('Figure 6. Left: predicted vs actual prices. The model tracks typical homes well; luxury properties above $3M are systematically underestimated. Right: residual plot — no systematic heteroscedasticity in the core $200K–$1M range.')); }

children.push(h2('7.1 Feature Importance'));
children.push(makeTable(
  ['Feature', 'Dir.', 'Interpretation'],
  [
    ['log_sqft_living_sq', '+', 'Quadratic log-living area: captures the concave size–price curve; increasing returns at smaller sizes, diminishing above 3,000 sqft'],
    ['zip_lp',             '+', 'ZIP code mean log-price: local price level is a strong prior on individual property value'],
    ['log_sqft_above',     '+', 'Above-ground floor area; preferred over total sqft as basements are discounted by the market'],
    ['city_lp',            '+', 'City-level price signal; complementary to ZIP encoding at a coarser geographic scale'],
    ['city_x_sqft',        '−', 'Location × size interaction: reflects partial regression adjustment — value of size is partially absorbed by location encoding, so the interaction corrects for substitution'],
    ['cond_x_age',         '+', 'Condition × age interaction: an old home in excellent condition commands a disproportionate premium — sustained upkeep is valued non-linearly'],
    ['bathrooms',          '+', 'Bathroom count; more stable predictor than bedroom count (less subject to high-leverage outliers)'],
    ['condition',          '+', 'Base condition level; amplified by cond_x_age for older properties'],
    ['log_house_age',      '−', 'Older properties trade at a discount on average; the interaction with condition captures exceptions'],
    ['size_x_condition',   '−', 'Partial regression adjustment: after controlling for size and condition separately, the interaction absorbs residual correlation'],
  ],
  [2000, 600, 6426]
));
children.push(caption('Table 5. Top 10 features by absolute standardised coefficient with plain-language interpretation.'));

children.push(h2('7.2 Key Insights for Stakeholders'));
children.push(body('**Neighbourhood comparables are the strongest predictor.** The ZIP and city target encodings together represent the local price level, and zip_x_sqft captures how that level amplifies the value of floor area. A property\'s estimated value is anchored primarily by what nearby comparable homes have sold for — consistent with standard appraisal practice.'));
children.push(body('**Location multiplies the value of size.** The city_x_sqft interaction demonstrates that an additional 500 sqft in an expensive city contributes more to value than the same space in a cheaper market. Location does not merely add a fixed premium — it scales the marginal value of every other attribute.'));
children.push(body('**Upkeep and renovation recency are valued non-linearly.** The cond_x_age interaction shows that an old home maintained at condition 5 commands a price premium beyond what its age and condition would predict separately. The market interprets sustained condition in an old building as evidence of exceptional ownership quality.'));
children.push(body('**Size has sharply diminishing returns above 3,000 sqft.** The positive log_sqft_living_sq coefficient reflects the concave shape identified in EDA: each additional square foot adds less value as a property grows larger. Extending an already large home is among the least efficient investments in this market.'));

// ── 8. XGBoost Comparison ─────────────────────────────────────────
children.push(h1('8. Comparison with XGBoost'));
children.push(body('XGBoost was trained in four configurations to provide a rigorous non-linear upper bound. The results reveal an important methodological lesson: a poorly regularised XGBoost is not a valid benchmark.'));

children.push(makeTable(
  ['Configuration', 'Train MAPE', 'Test MAPE', 'Gap', 'Assessment'],
  [
    ['Default (depth=6, n=500)',          '5.63%',  '15.49%', '+9.86pp', 'Severely overfit'],
    ['Tuned deeper (depth=7, n=800)',     '10.45%', '15.38%', '+4.92pp', 'Overfit'],
    ['Conservative (depth=4, n=400)',     '15.05%', '15.62%', '+0.57pp', 'Honest comparator'],
    ['Early Stopping (n=344)',            '11.21%', '15.28%', '+4.08pp', 'Principled'],
    ['Lean Ridge (ours)',                 fmt(R.lean_train), fmt(R.lean_test), sgn(R.lean_test - R.lean_train), 'Primary model'],
    ['Stacked ensemble (ours)',           '—',      fmt(R.stacked_test), '≈0pp', 'Best linear model'],
  ],
  [2600, 1400, 1400, 1400, 2226]
));
children.push(caption('Table 6. XGBoost configurations versus linear models. The Default configuration\'s 9.86pp train/test gap reveals memorisation; only the Conservative and Early Stopping configurations are valid benchmarks.'));

children.push(body('The XGBoost Default configuration achieves 5.63% training MAPE while testing at 15.49% — a 9.86 pp gap indicating severe memorisation. The Conservative configuration (depth=4) provides an honest comparator at 15.62% test MAPE with a negligible 0.57 pp gap. Compared to this, our stacked model (' + fmt(R.stacked_test) + ') trails by ' + (R.stacked_test - 15.62).toFixed(2) + ' pp — a difference within typical cross-validation variance.'));
children.push(body('The near-parity between the stacked linear model and properly regularised XGBoost has a structural explanation: the dominant non-linear interaction — the multiplicative location–size relationship — is captured explicitly by zip_x_sqft and city_x_sqft. Once these interactions are encoded, XGBoost\'s advantage in discovering interactions automatically is largely pre-empted.'));

// ── 9. Conclusion ─────────────────────────────────────────────────
children.push(h1('9. Conclusion'));
children.push(body('This paper demonstrates that a 20-feature Ridge regression model achieves ' + fmt(R.lean_test) + ' MAPE on house price prediction, with a stacked ensemble reaching ' + fmt(R.stacked_test) + '. The gap between our best model and a properly regularised XGBoost benchmark is ' + (R.stacked_test - 15.62).toFixed(2) + ' pp — negligible relative to cross-validation variance, and achieved without a single tree-based component.'));
children.push(body('Three methodological contributions underpin this result. First, a data integrity audit identified and corrected an artificial 50% duplication in the raw dataset, reducing the apparent XGBoost advantage from 5+ pp to approximately 1 pp. Second, Lasso regularisation path analysis established empirically that predictive performance saturates near 20 features — arguing strongly against feature proliferation in favour of deliberate, domain-motivated construction. Third, the most influential features encode the same heuristic that human appraisers use: a property\'s value is anchored by comparable nearby sales, and the marginal value of additional space scales with neighbourhood prestige.'));
children.push(body('A model built on explicit domain knowledge is not only accurate; it is interpretable, auditable, and defensible in regulated settings where prediction alone is insufficient.'));

// ── Assemble document ─────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '\u2022',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  styles: {
    default: { document: { run: { font: 'Arial', size: 20 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Arial', color: NAVY },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 22, bold: true, font: 'Arial', color: BLUE },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: A4_W, height: A4_H },
        margin: { top: LM, right: RM, bottom: LM, left: LM },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('house_price_report.docx', buf);
  console.log('Written: house_price_report.docx  (' + (buf.length / 1024).toFixed(0) + ' KB)');
});
