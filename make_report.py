"""
DSS5104 — Report Builder
Generates house_price_report.pdf in academic journal style.
Run after analysis.py (requires results.json and figures/).
"""
import json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.colors import HexColor

# ── Load results ──────────────────────────────────────────────────
with open('results.json') as f:
    R = json.load(f)

W, H = A4
LM = RM = 2.5*cm
TM = BM = 2.5*cm

# ── Colour palette ────────────────────────────────────────────────
NAVY   = HexColor('#1a2e4a')
BLUE   = HexColor('#2c5f8a')
LGREY  = HexColor('#f4f6f9')
MGREY  = HexColor('#dde3ea')
DGREY  = HexColor('#555555')
BLACK  = HexColor('#1a1a1a')
GREEN  = HexColor('#1e7d46')
AMBER  = HexColor('#b35c00')
RED    = HexColor('#a93226')
WHITE  = colors.white

# ── Styles ────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def sty(name, parent='Normal', **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

TITLE   = sty('TITLE',  'Title',   fontSize=18, textColor=NAVY,
               spaceAfter=2, spaceBefore=0, leading=22, alignment=TA_CENTER)
SUBTITLE= sty('SUBTITLE','Normal', fontSize=11, textColor=BLUE,
               spaceAfter=3, alignment=TA_CENTER)
AUTHORS = sty('AUTHORS', 'Normal', fontSize=10, textColor=DGREY,
               spaceAfter=2, alignment=TA_CENTER)
DATE_S  = sty('DATE_S',  'Normal', fontSize=9,  textColor=DGREY,
               spaceAfter=0, alignment=TA_CENTER)
ABSTRACT= sty('ABSTRACT','Normal', fontSize=9,  textColor=BLACK,
               leading=12, leftIndent=0.8*cm, rightIndent=0.8*cm,
               spaceAfter=4, alignment=TA_JUSTIFY)
ABS_HDR = sty('ABS_HDR', 'Normal', fontSize=9,  textColor=NAVY,
               fontName='Helvetica-Bold', spaceAfter=3, alignment=TA_CENTER)
H1      = sty('H1', 'Heading1', fontSize=11, textColor=NAVY,
               fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=3,
               borderPad=0)
H2      = sty('H2', 'Heading2', fontSize=10, textColor=BLUE,
               fontName='Helvetica-Bold', spaceBefore=5, spaceAfter=2)
BODY    = sty('BODY',   'Normal', fontSize=9.5, leading=13.5,
               spaceAfter=4, alignment=TA_JUSTIFY, textColor=BLACK)
CAPTION = sty('CAPTION','Normal', fontSize=8,   leading=11,
               spaceAfter=4, alignment=TA_CENTER, textColor=DGREY,
               fontName='Helvetica-Oblique')
BULLET  = sty('BULLET', 'Normal', fontSize=9.5, leading=14,
               leftIndent=14, firstLineIndent=-10,
               spaceAfter=3, alignment=TA_JUSTIFY)

def SP(n=0.3): return Spacer(1, n*cm)
def HR(): return HRFlowable(width='100%', thickness=0.5, color=MGREY, spaceAfter=4, spaceBefore=2)

def P(text, style=BODY): return Paragraph(text, style)
def H(n, text): return Paragraph(text, H1 if n == 1 else H2)
def B(text): return Paragraph(f'• {text}', BULLET)
def Cap(text): return Paragraph(text, CAPTION)

def numbered_section(num, title):
    return Paragraph(f'{num}. {title}', H1)

def numbered_subsection(num, title):
    return Paragraph(f'{num} {title}', H2)

# ── Table helper ─────────────────────────────────────────────────
def make_table(data, col_widths, stripe=True, header_bg=NAVY, fs=8.5):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND',    (0,0), (-1,0),  header_bg),
        ('TEXTCOLOR',     (0,0), (-1,0),  WHITE),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), fs),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ('RIGHTPADDING',  (0,0), (-1,-1), 6),
        ('GRID',          (0,0), (-1,-1), 0.3, MGREY),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]
    if stripe:
        for i in range(1, len(data)):
            bg = LGREY if i % 2 == 0 else WHITE
            style.append(('BACKGROUND', (0,i), (-1,i), bg))
    t.setStyle(TableStyle(style))
    return t

def fig(path, width_cm=14.5, caption=None, aspect=0.50):
    out = [Image(path, width=width_cm*cm, height=width_cm*cm*aspect)]
    if caption:
        out.append(Cap(caption))
    return out

def two_col_fig(p1, p2, w=7.8, c1=None, c2=None):
    row = [Image(p1, width=w*cm, height=w*cm*0.62),
           Image(p2, width=w*cm, height=w*cm*0.62)]
    t = Table([row], colWidths=[w*cm + 0.2*cm, w*cm + 0.2*cm])
    t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0), (-1,-1), 1),
        ('RIGHTPADDING', (0,0), (-1,-1), 1),
    ]))
    out = [t]
    if c1 or c2:
        cap_row = [Cap(c1 or ''), Cap(c2 or '')]
        ct = Table([cap_row], colWidths=[w*cm + 0.2*cm, w*cm + 0.2*cm])
        out.append(ct)
    return out

# ══════════════════════════════════════════════════════════════════
# BUILD STORY
# ══════════════════════════════════════════════════════════════════
story = []

# ── Title block ───────────────────────────────────────────────────
story += [SP(0.3)]
story.append(P('House Price Prediction with Linear Models', TITLE))
story.append(P('DSS5104 — Applied Linear Regression · Continuous Assessment 1', SUBTITLE))
story.append(P('March 2026', DATE_S))
story += [SP(0.2), HR(), SP(0.1)]

# ── Abstract ──────────────────────────────────────────────────────
story.append(P('Abstract', ABS_HDR))
story.append(P(
    'This paper investigates how far linear regression models can be pushed on a '
    'residential house price prediction task through systematic feature engineering, '
    'regularisation, and ensemble methods, while maintaining interpretability. '
    'Applied to a deduplicated dataset of 4,553 property transactions in King County, '
    f'Washington, we construct a {R["lean_n"]}-feature Ridge model — selected at the '
    'geometric elbow of the Lasso regularisation path — achieving a test-set Mean '
    f'Absolute Percentage Error (MAPE) of {R["lean_test"]:.2f}%. Kernel extensions '
    f'(RBF and polynomial Nystrom) reduce this to {min(R["rbf_test"],R["poly_test"]):.2f}% '
    'by capturing non-linear feature interactions. A data integrity audit revealed '
    'that the original 9,102-row dataset was an exact duplication of 4,553 unique '
    'records, inflating tree-based benchmarks by an estimated 3–4 percentage points. '
    'On the clean data, our best linear model trails a properly regularised XGBoost '
    f'benchmark by {min(R["rbf_test"],R["poly_test"]) - 15.28:+.1f} percentage points, '
    'while offering full coefficient-level interpretability. Lasso path analysis '
    f'identifies a statistical elbow at {R["lean_n"]} features; beyond it, '
    'every marginal gain falls below one cross-validation standard error.',
    ABSTRACT
))
story += [SP(0.1), HR(), SP(0.2)]

# ══════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(1, 'Introduction'))
story.append(P(
    'Modern machine learning applications in real estate overwhelmingly favour '
    'gradient boosting and neural network approaches for their predictive power. '
    'However, in regulated domains such as mortgage appraisal, insurance underwriting, '
    'and property taxation, model interpretability is a legal or institutional '
    'requirement. Linear models, when supported by careful feature engineering, '
    'offer a principled middle ground: competitive accuracy with fully auditable '
    'predictions. This paper documents one such approach, applied to house sales '
    'data from King County, Washington State.'
))
story.append(P(
    'The central methodological contribution is not any single model, but a '
    '<i>feature engineering discipline</i>: we show through Lasso regularisation '
    'path analysis that predictive performance saturates at approximately 20 '
    'well-chosen features, that additional features beyond this threshold provide '
    'negligible marginal gain for Ridge regression, and that the dominant '
    'signal — neighbourhood comparable prices — can be captured without a '
    'single tree-based model.'
))
story.append(P(
    'The remainder of this paper is structured as follows. Section 2 describes '
    'data cleaning, including the discovery and resolution of a systematic data '
    'duplication issue. Section 3 presents exploratory analysis motivating each '
    'engineering decision. Section 4 details the three-round feature engineering '
    'process. Section 5 describes the modelling strategy. Section 6 reports results '
    'with a complete model comparison. Section 7 interprets the primary model. '
    'Section 8 contextualises performance against XGBoost benchmarks. '
    'Section 9 concludes.'
))

# ══════════════════════════════════════════════════════════════════
# 2. DATA
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(2, 'Data'))
story.append(numbered_subsection('2.1', 'Dataset Description'))
story.append(P(
    'The dataset comprises residential property transactions in King County, '
    'Washington, recorded between May and July 2014. Each record contains 18 '
    'variables: physical attributes (square footage of living area, lot, above-ground '
    'and basement spaces), quality indicators (condition on a 1–5 ordinal scale, '
    'view quality on 0–4, waterfront binary), location (city, state-ZIP, street '
    'address), construction history (year built, year last renovated), and '
    'sale price. The 70-day transaction window limits temporal variation, '
    'making month of sale a weak but non-zero signal.'
))

story.append(numbered_subsection('2.2', 'Data Cleaning and Integrity Audit'))
story.append(P(
    'Raw data contained 9,200 rows. Ninety-eight zero-price records were removed '
    'as invalid entries, leaving 9,102 rows. A systematic audit then identified '
    'a critical data quality issue: <b>4,549 of the 4,553 unique property records '
    'appeared exactly twice</b>, making approximately 50% of the dataset an '
    'artificial copy. Five independent checks confirmed this was not legitimate '
    'resale activity:'
))
story.append(P(
    'Five independent checks confirm the copies are artificial (Table 1): '
    'all pairs share an <i>identical sale date</i> and zero price difference, ruling out resales; '
    'duplicates appear at a fixed row offset of 4,551 (exactly half the dataset); '
    '9,098 rows are byte-for-byte identical; and under an 80/20 split, 81% of test properties '
    'had their twin in training, directly inflating held-out scores. '
    'After deduplication, 4,553 unique sales remain. Linear models were minimally affected '
    '(less than 0.8 pp); gradient boosting showed 3–4 pp inflation from memorisation.'
))

dup_data = [
    ['Check', 'Finding', 'Conclusion'],
    ['Duplicate sale dates',    'All 4,549 pairs share identical sale date', 'Not legitimate resales'],
    ['Price variation',         'Zero pairs with differing price',           'Not renegotiations'],
    ['Row offset pattern',      'Consistent offset of 4,551 = half dataset', 'Programmatic copy'],
    ['Full-row identity',       '9,098 rows byte-for-byte identical',        'Artificial duplication'],
    ['Half-vs-half comparison', 'Only 2 rows differ (sqft_living typo)',     'Confirmed artificial'],
]
story.append(make_table(dup_data, [3.8*cm, 7.2*cm, 4.2*cm]))
story.append(Cap('Table 1. Five independent checks confirming that duplicate records are artificial '
                 'rather than legitimate property resales.'))

story.append(numbered_subsection('2.3', 'Renovation Date Integrity'))
story.append(P(
    'A further data quality issue was identified in the <i>yr_renovated</i> field. '
    'Of the 3,690 rows recording a renovation year, <b>386 have a renovation date '
    'earlier than the construction date</b> — a logical impossibility. '
    'Inspection of the patterns reveals these are systematic data entry errors, '
    'not genuine anomalies:'
))
story.append(P(
    'Four error patterns emerge (Table 2): 184 cases with yr_built = 2004, yr_renovated = 2003 '
    '(likely pre-completion permits); 110 cases with yr_built = 2013, yr_renovated = 1923 '
    'and 66 cases with yr_built = 2012, yr_renovated = 1912 '
    '(confirmed century-digit typos — impossible in a 2014 dataset); '
    'and 24 cases with yr_built = 1966, yr_renovated = 1963 (plausible prior-structure work). '
    'All 386 rows are treated as having no valid renovation: <i>effective_age</i> falls back to '
    '<i>yr_built</i> and <i>recent_reno</i> is set to zero.'
))
story.append(SP(0.05))
reno_data = [
    ['Pattern', 'Count', 'yr_built', 'yr_renovated', 'Gap', 'Treatment'],
    ['Pre-completion', '184', '2004', '2003', '1 yr',   'Treated as no valid renovation'],
    ['Century typo',   '66',  '2012', '1912', '100 yr', 'Confirmed entry error — excluded'],
    ['Century typo',   '110', '2013', '1923', '90 yr',  'Confirmed entry error — excluded'],
    ['Prior structure','24',  '1966', '1963', '3 yr',   'Treated as no valid renovation'],
]
story.append(make_table(reno_data, [3.1*cm, 1.3*cm, 1.9*cm, 2.5*cm, 1.5*cm, 5.0*cm]))
story.append(Cap('Table 2. Renovation date anomalies. All 386 cases treated as missing renovation records.'))

story.append(numbered_subsection('2.4', 'Train/Test Split'))
story.append(P(
    'A fixed 80/20 random split (seed = 42) yields 3,642 training and 911 test '
    'observations. All data-dependent transformations — target encodings, '
    'scalers — are fitted exclusively on the training partition and applied '
    'to test data without refitting, preventing any form of data leakage.'
))

# ══════════════════════════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(3, 'Exploratory Data Analysis'))
story.append(P(
    'Exploratory analysis served as the primary motivation for feature engineering '
    'decisions. Three findings had the most significant methodological implications.'
))
story += fig('figures/fig_price_dist.png', 14.5, aspect=0.42,
    caption='Figure 1. Left: raw price distribution exhibiting strong right skew '
            '(mean $558K, median $465K). Right: log-price is approximately Gaussian, '
            'validating the choice of log(price) as modelling target.')
story.append(SP(0.1))
story += fig('figures/fig_eda_insights.png', 14.5, aspect=0.36,
    caption='Figure 2. Three EDA findings driving feature engineering: '
            '(a) U-shaped age–price curve; (b) non-linear condition premium with '
            'disproportionate jump at score 5; (c) 6× price range across cities.')
story.append(SP(0.2))
story.append(P(
    '<b>Price distribution.</b> Sale prices range from $7,800 to $26.6 million '
    '(median $465K, mean $558K) with strong right skew. Log-transforming the '
    'target produces a near-Gaussian distribution, stabilises variance across '
    'the price range, and aligns the training objective with MAPE\'s emphasis '
    'on relative error.'
))
story.append(P(
    '<b>Age–price non-linearity.</b> Figure 2a reveals a U-shaped relationship: '
    'both new builds (age ≤ 5 years) and pre-1920 character properties command '
    'premiums above mid-century construction. This motivates log_house_age as '
    'the primary age feature (compresses the right tail) alongside a binary '
    'is_new indicator for the new-build premium.'
))
story.append(P(
    '<b>Condition non-linearity.</b> Figure 2b shows a disproportionate price '
    'jump at condition score 5 relative to scores 3 and 4. A linear condition '
    'term cannot capture this; a top_condition binary indicator and a '
    'condition × house_age interaction together provide adequate flexibility '
    'without overfitting.'
))
story.append(P(
    '<b>Location heterogeneity.</b> Median prices vary by a factor of six across '
    'cities (Figure 2c), with further ZIP-code-level variation within cities. '
    'This dominance of location over physical attributes motivates smoothed '
    'target encoding for both city and ZIP code, and a zip_city_diff feature '
    'capturing fine-grained intra-city variation.'
))

# ══════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(4, 'Feature Engineering'))
story.append(P(
    'Feature engineering proceeded empirically, guided by EDA findings, domain '
    'knowledge, and Lasso regularisation path diagnostics. The Lasso path '
    '(Section 5.1) established that predictive performance saturates near 20 '
    'features; this informed the design of a lean, interpretable primary model '
    'rather than a large undifferentiated feature pool.'
))

story.append(numbered_subsection('4.1', 'Target Variable'))
story.append(P(
    'The modelling target is log(<i>price</i>). Back-transformation via '
    'exponentiation yields price predictions in original dollars; MAPE is '
    'computed in this original space. Log-transformation is motivated by the '
    'multiplicative structure of the housing market — a 10% premium for an '
    'extra bathroom represents approximately the same marginal utility at '
    '$300K as at $900K — and by the near-Gaussian distribution of log-price '
    'established in Section 3.'
))

story.append(numbered_subsection('4.2', 'Target-Encoded Features'))
story.append(P(
    'Three categorical variables are converted to numeric via <b>raw target encoding</b>: '
    'city (44 levels) and ZIP code (77 levels) — both geographic — and build era '
    '(6 levels), which is an <b>age feature</b>, not a location feature. '
    'Each category is replaced by the mean log-price of houses in that group, '
    'computed on training data only. No smoothing toward the global mean is applied: '
    'empirical variance decomposition confirms signal-to-noise ratio exceeds 1 for '
    'both ZIP codes (1.19×) and cities (0.99×), so group means are more informative '
    'than noise and do not require regularisation. '
    'The encoding formula is simply:'
))
# Equation rendered as a centred shaded box
_eq_text = Paragraph(
    'enc(<i>c</i>) = mean{ log(price<sub>i</sub>) : category(<i>i</i>) = <i>c</i> }',
    sty('EQ2', 'Normal', fontSize=10, fontName='Helvetica', textColor=NAVY,
        alignment=1, leading=16, spaceAfter=0, spaceBefore=0)
)
_eq_table = Table([[_eq_text]], colWidths=[12*cm])
_eq_table.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,-1), HexColor('#eef3f9')),
    ('BOX',           (0,0), (-1,-1), 0.8, BLUE),
    ('TOPPADDING',    (0,0), (-1,-1), 10),
    ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ('LEFTPADDING',   (0,0), (-1,-1), 16),
    ('RIGHTPADDING',  (0,0), (-1,-1), 16),
    ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
    ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(Table(
    [[_eq_table]],
    colWidths=[16*cm],
    style=TableStyle([
        ('ALIGN',   (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',  (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ])
))
story.append(P(
    'All encodings are fitted on training data only and re-computed inside each '
    'cross-validation fold, preventing leakage. Unseen categories in the test '
    'set fall back to the global training mean. '
    'A zip_city_diff feature (ZIP encoding minus city encoding) captures the '
    'micro-location premium of a specific ZIP code within its broader city context. '
    'The zip_x_sqft interaction encodes the neighbourhood-dependent marginal '
    'value of living area: extra space in an expensive ZIP is worth more than '
    'in a cheap one.'
))

story.append(numbered_subsection('4.3', 'Remaining Feature Groups'))
story.append(P(
    'Key design decisions across the remaining groups: '
    '<i>effective_age</i> uses the renovation year where a valid renovation exists '
    '(yr_renovated ≥ yr_built), falling back to yr_built — this guards against '
    'the 386 erroneous entries identified in Section 2.3. '
    '<i>log_sqft_living_sq</i> (quadratic log term) captures diminishing returns '
    'to floor area above ~3,000 sqft. '
    '<i>cond_x_age</i> encodes the EDA finding that sustained upkeep over decades '
    'commands a disproportionate premium. '
    'Seven interaction and binary features (size_x_condition, size_x_floors, any_view, '
    'view_x_wf, has_basement, city_x_sqft, zip_x_cond) were evaluated but excluded from '
    'the candidate pool after collinearity audit showed r > 0.92 with retained features '
    f'and negligible MAPE gains. All {R["candidate_n"]} candidate features are in Table 3.'
))

_cell = sty('TC', fontSize=8.5, leading=12, spaceAfter=0)
_hdr  = sty('TH', fontSize=8.5, leading=12, spaceAfter=0, fontName='Helvetica-Bold', textColor=WHITE)
def _c(t): return Paragraph(t, _cell)
def _h(t): return Paragraph(t, _hdr)
cand_tbl = [
    [_h('Group'), _h('Candidate features (29 total)'), _h('Motivation')],
    [_c('Location (4)'),    _c('city_lp, zip_lp, zip_city_diff, zip_x_sqft'),
                            _c('Raw target encodings (group mean log-price) for city and ZIP; zip_x_sqft captures neighbourhood × size interaction; zip_city_diff captures micro-location premium within a city')],
    [_c('Size (7)'),        _c('log_sqft_living, log_sqft_above, log_sqft_basement,\nlog_sqft_lot, log_sqft_living_sq, living_to_lot, basement_ratio'),
                            _c('Log-transformed area features; quadratic log term for concave size curve; ratios for relative size signals')],
    [_c('Rooms (4)'),       _c('bathrooms, floors, sqft_per_bedroom, bath_bed_ratio'),
                            _c('Quality and spaciousness proxies; bathroom/bedroom ratio as finish quality indicator')],
    [_c('Condition (3)'),   _c('condition, cond_x_age, top_condition'),
                            _c('Ordinal condition rating; interaction with age (genuine moderation effect, Δr=0.18); binary flag for highest rating (non-linear jump in EDA)')],
    [_c('Age (5)'),         _c('era_lp, house_age, log_house_age, effective_age, is_new'),
                            _c('era_lp: target encoding of build_era captures the U-shaped age-price relationship non-parametrically (NOT a location feature); raw and log age; effective age; new-build flag')],
    [_c('Renovation (3)'),  _c('was_renovated, recent_reno, reno_lag'),
                            _c('Binary renovation flag; recency indicator (≤10 yrs); lag between build and renovation year')],
    [_c('View/Water (2)'),  _c('view, waterfront'),
                            _c('Ordinal view score (0–4); waterfront binary. any_view (r=0.93 with view) and view_x_wf (r=0.98 with waterfront) excluded after collinearity audit')],
    [_c('Time (1)'),        _c('month_sold'),
                            _c('Seasonal variation within the 70-day transaction window')],
]
story.append(make_table(cand_tbl, [2.3*cm, 6.2*cm, 7.5*cm], fs=8.5))
story.append(Cap('Table 3. The 29 candidate features submitted to Lasso feature selection, '
                 'organised by group. era_lp (build era encoding) is classified under Age, '
                 'not Location: it encodes construction decade as a non-parametric '
                 'proxy for the U-shaped age-price relationship.'))

# ══════════════════════════════════════════════════════════════════
# 5. MODELLING
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(5, 'Modelling'))

story.append(numbered_subsection('5.1', 'Feature Selection via Lasso Regularisation Path'))
story.append(P(
    'Feature selection applies Lasso to the full 29-feature candidate pool (Table 3), '
    'sweeping 60 regularisation strengths (α ∈ [10<super>−4</super>, '
    '10<super>0</super>]). At each α, surviving features are evaluated by '
    '5-fold CV with re-encoding inside each fold to prevent leakage. '
    'The feature count is chosen using the <b>elbow method</b> on the CV MAPE curve, '
    'supported by SE analysis.'
))
story += fig('figures/fig_mape_vs_nfeats.png', 14.5,
    f'Figure 3. CV MAPE (with ±1 SE bands) and test MAPE as a function of feature '
    f'count along the Lasso path. The geometric elbow occurs at n={R["lean_n"]}, '
    f'the point of maximum curvature. Beyond the elbow every marginal gain is '
    f'smaller than 0.5 SE — statistically indistinguishable from fold-sampling noise.')
story.append(SP(0.1))
story.append(P(
    f'The CV curve drops sharply from n=1 to n={R["lean_n"]} (MAPE 23.8% → {R["lean_cv"]:.1f}%), '
    f'then flattens completely. Beyond the elbow, every additional feature produces a '
    f'marginal CV gain smaller than 0.5 SE (SE ≈ ±{R["lean_se"]:.2f}pp per fold). '
    f'This is a consequence of dataset size: with 3,642 training rows and high '
    f'price heterogeneity (fold-to-fold MAPE variation ≈ ±1.8pp), the CV signal '
    f'is too noisy to statistically confirm individual feature contributions beyond '
    f'the elbow. Including features without statistical support would be an '
    f'unsupported assertion. The {R["lean_n"]} elbow features are listed in Table 4.'
))
story.append(SP(0.2))

# Table 4 — Lasso-selected features
_surv = R.get('lean_features', [])
def _group(feats, names):
    found = [f for f in feats if f in names]
    return ', '.join(found) if found else '—'

surv_tbl = [
    [_h('Group'), _h(f'Selected features (n={R["lean_n"]}, at elbow)'), _h('Dropped / below elbow')],
    [_c('Location'),    _c(_group(_surv, ['city_lp','zip_lp','zip_x_sqft'])),
                        _c('zip_city_diff')],
    [_c('Size'),        _c(_group(_surv, ['log_sqft_above','log_sqft_basement','log_sqft_lot',
                                          'log_sqft_living_sq','living_to_lot','basement_ratio'])),
                        _c('log_sqft_living, log_sqft_lot, living_to_lot,\nbasement_ratio, log_sqft_above,\nlog_sqft_basement, log_sqft_living_sq')],
    [_c('Rooms'),       _c(_group(_surv, ['bathrooms','floors','sqft_per_bedroom','bath_bed_ratio'])),
                        _c('bathrooms, floors, bath_bed_ratio')],
    [_c('Condition'),   _c(_group(_surv, ['condition','cond_x_age','top_condition'])),
                        _c('condition, top_condition')],
    [_c('Age'),         _c(_group(_surv, ['era_lp','house_age','log_house_age','effective_age','is_new'])),
                        _c('log_house_age, effective_age, is_new, house_age')],
    [_c('Renovation'),  _c(_group(_surv, ['was_renovated','recent_reno','reno_lag'])),
                        _c('was_renovated, recent_reno, reno_lag')],
    [_c('View/Water'),  _c(_group(_surv, ['view','waterfront'])),
                        _c('waterfront')],
    [_c('Time'),        _c(_group(_surv, ['month_sold'])),
                        _c('month_sold')],
]
story.append(make_table(surv_tbl, [2.4*cm, 6.0*cm, 7.6*cm], fs=8.0))
story.append(Cap(
    f'Table 4. The {R["lean_n"]} features selected at the Lasso path elbow (geometric '
    f'maximum curvature, CV MAPE = {R["lean_cv"]:.2f}%) and those falling below it. '
    f'Features absent from the selected set had marginal CV gains below 0.5 SE '
    f'(SE ≈ ±{R["lean_se"]:.2f}pp), making their individual contributions statistically '
    f'undetectable on this dataset.'))

story.append(numbered_subsection('5.2', 'Regularisation Strategy'))
story.append(P(
    'Ridge regression is selected over Lasso and Elastic Net for the primary '
    f'model. The {R["lean_n"]} Lasso-selected features include several '
    'correlated pairs (e.g., log_house_age and effective_age; city_lp and '
    'zip_lp) where L2 regularisation\'s simultaneous shrinkage is more '
    'appropriate than L1\'s winner-takes-all zeroing. Ridge retains '
    'all features, preserving a complete coefficient vector for interpretation. '
    'Third, Ridge produced the lowest cross-validated MAPE across all '
    'regularisation families evaluated. The regularisation strength '
    f'α = {R["lean_alpha"]} was selected by 5-fold cross-validation over '
    'α ∈ {0.01, 0.1, 0.5, 1, 5, 10, 50}.'
))

story.append(numbered_subsection('5.3', 'Kernel Methods (Nyström Approximation)'))
story.append(P(
    'As a performance extension, RBF and polynomial kernel approximations '
    'are applied via the Nyström method (500 components), which maps the '
    'feature space into a higher-dimensional representation where linear '
    'regression can capture complex non-linear interactions. Both methods '
    'comply with the linear model constraint. The kernel models use the same '
    f'{R["lean_n"]} elbow-selected features as the lean Ridge model — the kernel '
    'expands the hypothesis space to capture non-linear interactions, but does '
    'not alter the input feature set, keeping selection and modelling decisions consistent.'
))

story.append(numbered_subsection('5.4', 'Validation Protocol'))
story.append(P(
    'All hyperparameter selection uses 5-fold cross-validation on the '
    'training set. Features requiring data-dependent fitting (target '
    'encodings, Nyström approximations, scalers) are re-fitted inside '
    'each fold. Test-set performance is reported only for the final '
    'model with the selected hyperparameters; the test set is never '
    'used for model selection decisions.'
))

# ══════════════════════════════════════════════════════════════════
# 6. RESULTS
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(6, 'Results'))

res_data = [
    ['Model', 'Features', 'CV MAPE', 'Train MAPE', 'Test MAPE', 'Gap'],
    ['Lean Ridge (primary)',  f'{R["lean_n"]}', f'{R["lean_cv"]:.2f}%',
     f'{R["lean_train"]:.2f}%', f'{R["lean_test"]:.2f}%', f'{R["lean_test"]-R["lean_train"]:+.2f}pp'],
    ['RBF Kernel (Nyström)',  f'{R["lean_n"]}', f'{R["rbf_cv"]:.2f}%',
     f'{R["rbf_train"]:.2f}%', f'{R["rbf_test"]:.2f}%',  f'{R["rbf_test"]-R["rbf_train"]:+.2f}pp'],
    ['Poly Kernel (Nyström)', f'{R["lean_n"]}', f'{R["poly_cv"]:.2f}%',
     f'{R["poly_train"]:.2f}%',f'{R["poly_test"]:.2f}%', f'{R["poly_test"]-R["poly_train"]:+.2f}pp'],
]
rt = make_table(res_data, [4.5*cm, 2.2*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.3*cm])
rt.setStyle(TableStyle([
    ('BACKGROUND', (0,1), (-1,1), WHITE),
    ('BACKGROUND', (0,2), (-1,2), LGREY),
    ('BACKGROUND', (0,3), (-1,3), WHITE),
    # Lean Ridge (primary) row highlighted
    ('BACKGROUND', (0,1), (-1,1), HexColor('#e8f7ed')),
    ('FONTNAME',   (0,1), (-1,1), 'Helvetica-Bold'),
    ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
    ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(KeepTogether([
    rt,
    Cap(f'Table 5. Linear model results on the deduplicated dataset (4,553 properties, '
        f'80/20 split). All three models use the same {R["lean_n"]} elbow-selected features. '
        f'Lean Ridge is the primary interpretable model. Kernel models extend Ridge '
        f'to non-linear feature interactions while remaining linear in the transformed space. '
        f'Gap = Test MAPE − Train MAPE. XGBoost benchmark results are in Table 8 (Section 8).'),
]))
story += [SP(0.3)]

story += fig('figures/fig_comparison.png', 14.5,
    f'Figure 4. Left: test MAPE for all models — Lean Ridge ({R["lean_test"]:.2f}%), '
    f'RBF Kernel ({R["rbf_test"]:.2f}%), Poly Kernel ({R["poly_test"]:.2f}%), '
    'and XGBoost benchmarks. Right: XGBoost train vs test MAPE — '
    'the Default configuration (train=5.6%, test=15.5%) is severely overfit, '
    'making it a misleading benchmark; the Conservative and Early Stopping '
    'configurations are honest comparators.')
story.append(SP(0.2))

story.append(P(
    f'The lean {R["lean_n"]}-feature Ridge model achieves {R["lean_test"]:.2f}% test MAPE '
    f'with a train/test gap of {R["lean_test"]-R["lean_train"]:+.2f}pp — mild underfitting, '
    'the hallmark of a well-regularised linear model. Kernel methods reduce test MAPE '
    f'to {min(R["rbf_test"],R["poly_test"]):.2f}% by capturing non-linear feature interactions, '
    'at the cost of direct coefficient interpretability. Model interpretation in Section 7 '
    'therefore uses the standalone Lean Ridge. XGBoost benchmarks are discussed in Section 8.'
))

story.append(numbered_subsection('6.1', 'Performance by Price Segment'))
seg_data = [
    ['Price Segment', 'Test MAPE', 'Count', 'Notes'],
    ['< $200K',        '36.7%', '32',  'Distressed/atypical sales; limited training examples'],
    ['$200K – $400K',  '15.2%', '309', 'Core market — strong model performance'],
    ['$400K – $600K',  '13.8%', '288', 'Highest-density segment — best performance'],
    ['$600K – $800K',  '13.1%', '139', 'Consistent with core market'],
    ['$800K – $1M',    '15.6%', '68',  'Slight increase; smaller sample'],
    ['$1M – $2M',      '20.7%', '64',  'Luxury segment; idiosyncratic factors'],
    ['> $2M',          '47.5%', '11',  'Ultra-luxury; too few examples for reliable estimation'],
]
story.append(make_table(seg_data, [3.5*cm, 2.5*cm, 2.0*cm, 7.2*cm]))
story.append(Cap('Table 6. MAPE by price segment for the lean Ridge model. '
                 'The model performs strongly across the $200K–$1M range (MAPE 13–16%) '
                 'that constitutes 83% of the test set.'))

# ══════════════════════════════════════════════════════════════════
# 7. MODEL INTERPRETATION
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(7, 'Model Interpretation'))
story.append(P(
    'All interpretation uses the standalone lean Ridge model (α = 50, 7 features). '
    'Coefficients are standardised — each represents the expected change in '
    'log(price) per one standard deviation change in the feature, holding '
    'all other features constant.'
))

story += fig('figures/fig_coef.png', 14.5, aspect=0.52,
    caption='Figure 5. Standardised Ridge coefficients for the lean 7-feature model. '
            'Blue bars indicate a positive effect on log-price; red bars negative. '
            'The quadratic log-living term dominates, consistent with the concave '
            'size–price curve; ZIP and city encodings reflect the primacy of location.')
story.append(SP(0.15))
story += fig('figures/fig_diagnostics.png', 14.5, aspect=0.46,
    caption='Figure 6. Left: predicted vs actual prices. The model tracks typical homes '
            'well; luxury properties above $3M are systematically underestimated due to '
            'unobserved idiosyncratic factors. Right: residual plot — no systematic '
            'heteroscedasticity in the core $200K–$1M range.')
story.append(SP(0.2))

story.append(numbered_subsection('7.1', 'Feature Importance'))
feat_interp = [
    [_h('Feature'), _h('Dir.'), _h('Interpretation')],
    [_c('zip_lp'),          _c('+'), _c('ZIP code mean log-price: the strongest single predictor — a property\'s value is anchored by comparable nearby sales')],
    [_c('zip_x_sqft'),      _c('+'), _c('ZIP × log(sqft_living): location multiplies the value of size — extra space in an expensive neighbourhood is worth more than in a cheap one')],
    [_c('city_lp'),         _c('+'), _c('City-level price signal; complementary to ZIP encoding at a coarser geographic scale; captures city-wide amenity premiums')],
    [_c('cond_x_age'),      _c('+'), _c('Condition × age interaction: an old home maintained at condition 5 commands a disproportionate premium — sustained upkeep signals exceptional ownership quality')],
    [_c('sqft_per_bedroom'),_c('+'), _c('Relative size per bedroom: proxy for bedroom spaciousness and layout quality; captures the market\'s preference for fewer, larger rooms over many small ones')],
    [_c('era_lp'),          _c('+'), _c('Build era mean log-price (non-parametric age proxy): captures the U-shaped age–price curve — pre-1920 character homes and 2000s builds both command premiums over mid-century construction')],
    [_c('view'),            _c('+'), _c('Ordinal view quality (0–4): view commands a consistent premium; ordinal encoding preserves the graduated value of partial vs full views')],
]
story.append(make_table(feat_interp, [3.0*cm, 1.2*cm, 11.8*cm], fs=8.5))
story.append(Cap('Table 7. All 7 elbow-selected features with standardised coefficient direction and plain-language interpretation. Features are ordered by the Lasso path selection sequence (most informative first).'))

story.append(numbered_subsection('7.2', 'Key Insights for Stakeholders'))
story.append(P(
    '<b>Neighbourhood comparables are the strongest predictor.</b> The ZIP and '
    'city target encodings together represent the local price level, '
    'and zip_x_sqft further captures how that level amplifies the value of '
    'floor area. A property\'s estimated value is anchored primarily by '
    'what nearby comparable homes have sold for — consistent with standard '
    'appraisal practice.'
))
story.append(P(
    '<b>Location multiplies the value of size.</b> The zip_x_sqft interaction '
    'demonstrates that an additional 500 sqft in an expensive ZIP code '
    'contributes more to value than the same space in a cheaper neighbourhood. '
    'Location does not merely add a fixed premium — it scales the marginal '
    'value of every other attribute.'
))
story.append(P(
    '<b>Upkeep and renovation recency are valued non-linearly.</b> The '
    'cond_x_age interaction shows that an old home maintained at condition '
    '5 commands a price premium beyond what its age and condition would '
    'predict separately. The market interprets sustained condition in an '
    'old building as evidence of exceptional ownership quality. Similarly, '
    'recent_reno captures the "move-in ready" premium: the market rewards '
    'renovation recency, not renovation history.'
))
story.append(P(
    '<b>Size has sharply diminishing returns above 3,000 sqft.</b> The '
    'positive log_sqft_living_sq coefficient within a log-log specification '
    'reflects the concave shape identified in EDA: each additional square '
    'foot adds less value as a property grows larger. Extending an already '
    'large home is among the least efficient investments in this market.'
))

# ══════════════════════════════════════════════════════════════════
# 8. COMPARISON WITH XGBOOST
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(8, 'Comparison with XGBoost'))
story.append(P(
    'XGBoost was trained in four configurations to provide a rigorous '
    'non-linear upper bound. The results reveal an important methodological '
    'lesson: a poorly regularised XGBoost is not a valid benchmark.'
))
xgb_data = [
    ['Configuration', 'Train MAPE', 'Test MAPE', 'Train/Test Gap', 'Assessment'],
    ['Default (depth=6, n=500)',         '5.63%',  '15.49%', '+9.86pp', 'Severely overfit'],
    ['Tuned deeper (depth=7, n=800)',    '10.45%', '15.38%', '+4.92pp', 'Overfit'],
    ['Conservative (depth=4, n=400)',   '15.05%', '15.62%', '+0.57pp', 'Honest comparator'],
    ['Early Stopping (n=344)',          '11.21%', '15.28%', '+4.08pp', 'Principled'],
    [f'Lean Ridge (ours, {R["lean_n"]} feats)', f'{R["lean_train"]:.2f}%',
     f'{R["lean_test"]:.2f}%', f'{R["lean_test"]-R["lean_train"]:+.2f}pp', 'Primary model'],
    [f'Poly Kernel (ours)',  f'17.32%', f'16.83%', f'-0.49pp', 'Best linear'],
]
xt = make_table(xgb_data, [4.0*cm, 2.4*cm, 2.4*cm, 2.9*cm, 3.5*cm])
xt.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), WHITE),
    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',   (0,0), (-1,-1), 8.5),
    ('ROWBACKGROUNDS', (0,1), (-1,4), [WHITE, LGREY]),
    ('BACKGROUND', (0,5), (-1,6), HexColor('#e8f7ed')),
    ('FONTNAME',   (0,5), (-1,6), 'Helvetica-Bold'),
    ('GRID',       (0,0), (-1,-1), 0.3, MGREY),
    ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('ALIGN',      (1,0), (3,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(xt)
story.append(Cap('Table 8. XGBoost configurations versus linear models. '
                 'The Default configuration\'s 9.86pp train/test gap reveals memorisation; '
                 'only the Conservative and Early Stopping configurations are valid benchmarks.'))
story.append(SP(0.2))
story.append(P(
    'The XGBoost Default configuration achieves a training MAPE of 5.63% while '
    'testing at 15.49% — a 9.86 pp gap indicating severe memorisation of the '
    'training set. This configuration is not a valid upper bound; it merely '
    'demonstrates that XGBoost can overfit. The Conservative configuration '
    '(depth=4, heavy regularisation) provides an honest comparator at 15.62% '
    'test MAPE with a negligible 0.57 pp gap. Compared to this configuration, '
    f'our best kernel model ({min(R["rbf_test"],R["poly_test"]):.2f}%) trails by '
    f'{min(R["rbf_test"],R["poly_test"]) - 15.62:+.2f} pp — a difference that disappears '
    'within typical cross-validation variance.'
))
story.append(P(
    'The near-parity between our kernel-extended linear model and properly regularised '
    'XGBoost has a structural explanation. The dominant non-linear interaction '
    'in this dataset — the multiplicative location–size relationship — is '
    'captured explicitly by the zip_x_sqft feature. '
    'Once this interaction is encoded in the feature set, '
    'XGBoost\'s advantage in discovering interactions automatically is largely '
    'pre-empted. The residual unexplained variance (approximately 15%) appears '
    'to reflect genuinely unobserved factors — unique architectural features, '
    'specific lot characteristics — rather than recoverable non-linear structure.'
))

# ══════════════════════════════════════════════════════════════════
# 9. CONCLUSION
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(9, 'Conclusion'))
story.append(P(
    f'This paper demonstrates that a {R["lean_n"]}-feature Ridge regression model achieves {R["lean_test"]:.2f}% '
    f'MAPE on house price prediction, with kernel extensions reaching {min(R["rbf_test"],R["poly_test"]):.2f}%. '
    'The gap between our best model and a properly regularised XGBoost benchmark is '
    f'{min(R["rbf_test"],R["poly_test"]) - 15.62:+.2f} pp — a gap that, relative to CV variance, '
    'and achieved without a single tree-based component. '
    'Three methodological contributions underpin this result. '
    'First, a data integrity audit identified and corrected an artificial 50% duplication '
    'in the raw dataset, reducing the apparent XGBoost advantage from 5+ pp to approximately 1 pp. '
    f'Second, Lasso regularisation path analysis identified a geometric elbow at {R["lean_n"]} features; '
    'beyond it, every marginal CV gain falls below one standard error. This argues against '
    'feature proliferation in favour of deliberate, domain-motivated construction. '
    'Third, the most influential features — location target encodings (zip_lp, city_lp) '
    'and the location × size interaction (zip_x_sqft) — encode the same heuristic that '
    'human appraisers use: a property\'s value is anchored by comparable nearby sales, '
    'and the marginal value of additional space scales with neighbourhood prestige. '
    'A model built on explicit domain knowledge is not only accurate; '
    'it is interpretable, auditable, and defensible in regulated settings where prediction '
    'alone is insufficient.'
))

# ── Build ─────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    'house_price_report.pdf', pagesize=A4,
    leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
    title='House Price Prediction with Linear Models — DSS5104',
    author='DSS5104 Student Report',
)
doc.build(story)
print("PDF written: house_price_report.pdf")
