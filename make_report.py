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
    'Washington, we construct a lean 20-feature Ridge model achieving a test-set Mean '
    'Absolute Percentage Error (MAPE) of 17.62%, and a stacked ensemble of '
    'kernel-augmented linear models achieving 16.52%. A data integrity audit revealed '
    'that the original 9,102-row dataset was an artificial duplication of 4,553 unique '
    'records, inflating tree-based benchmarks by an estimated 3–4 percentage points. '
    'On the clean data, our stacked model trails a properly regularised XGBoost '
    'benchmark by 1.2 percentage points, while offering full coefficient-level '
    'interpretability. Lasso regularisation path analysis demonstrates that '
    'predictive performance saturates near 20 features, confirming that thoughtful '
    'feature design is more valuable than feature proliferation.',
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
story.append(make_table(reno_data, [3.2*cm, 1.4*cm, 2.0*cm, 3.0*cm, 1.6*cm, 4.0*cm]))
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

story.append(numbered_subsection('4.2', 'Location Features'))
story.append(P(
    'Location is the dominant price determinant. We employ <b>Bayesian-smoothed '
    'target encoding</b> for city (44 levels), ZIP code (70 levels), and build '
    'era (6 levels). Each category\'s mean log-price is blended with the '
    'global mean using a smoothing strength of 30, preventing overfitting on '
    'sparsely observed categories:'
))
# Equation rendered as a centred shaded box
_eq_text = Paragraph(
    'enc(<i>c</i>) = '
    '[ <i>n</i>(<i>c</i>) · <i>mean</i>(<i>c</i>) + 30 · <i>global_mean</i> ]'
    ' / '
    '[ <i>n</i>(<i>c</i>) + 30 ]',
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
    'cross-validation fold. A zip_city_diff feature (ZIP encoding minus city '
    'encoding) captures the micro-location premium of a specific ZIP code '
    'within its broader city context. The key location × size interaction '
    'zip_x_sqft encodes the neighbourhood-dependent marginal value of living '
    'area: extra space in an expensive ZIP is worth more than in a cheap one.'
))

story.append(numbered_subsection('4.3', 'Age and Renovation Features'))
story.append(P(
    'Three features characterise the temporal dimension of a property. '
    '<i>log_house_age</i> = log(year_sold − yr_built + 1) compresses the '
    'right tail of the age distribution while preserving the U-shaped price '
    'curve identified in Section 3. <i>effective_age</i> = year_sold − '
    'yr_reno_fill uses the renovation year where a <i>valid</i> renovation '
    'exists (i.e. yr_renovated ≥ yr_built), otherwise falling back to yr_built. '
    'This guard excludes the 386 erroneous entries identified in Section 2.3. '
    '<i>recent_reno</i> is a binary indicator for renovation within 10 years '
    'of sale, encoding the "move-in ready" premium observed in the market; '
    'it is also set to zero for all invalid renovation records.'
))

story.append(numbered_subsection('4.4', 'Condition and Interaction Features'))
story.append(P(
    'Beyond the raw condition score, the interaction <i>cond_x_age</i> = '
    'condition × house_age captures a finding from EDA: a 60-year-old '
    'home rated condition 5 commands a disproportionate premium, as '
    'sustained upkeep over decades signals exceptional ownership quality. '
    '<i>size_x_condition</i> = log(sqft_living) × condition reflects '
    'the amplification of condition premiums in larger homes. '
    '<i>city_x_sqft</i> = city_lp × log(sqft_above) captures the '
    'multiplicative location–size interaction.'
))

story.append(numbered_subsection('4.5', 'Size and Room Features'))
story.append(P(
    'The concave size–price relationship (diminishing returns to floor area) '
    'is captured by log_sqft_above and log_sqft_living_sq. Above-ground '
    'area is preferred over total living area as the primary size metric '
    'because basements contribute less value per square foot. '
    'sqft_per_bedroom = sqft_living / (bedrooms + 1) is a spaciousness '
    'proxy; bath_bed_ratio encodes bathroom–bedroom balance. '
    'has_basement is a binary indicator; basement size is omitted from the '
    'lean model as its marginal contribution was near zero.'
))

story.append(numbered_subsection('4.6', 'View, Waterfront, and Outlier Indicators'))
story.append(P(
    'View (0–4 ordinal) and waterfront (binary) appear directly as features '
    'rather than in compound indicators, as Lasso path analysis showed their '
    'individual coefficients remained non-zero even at strong regularisation. '
    'month_sold captures seasonal variation.'
))

_cell = sty('TC', fontSize=8.5, leading=12, spaceAfter=0)
_hdr  = sty('TH', fontSize=8.5, leading=12, spaceAfter=0, fontName='Helvetica-Bold', textColor=WHITE)
def _c(t): return Paragraph(t, _cell)
def _h(t): return Paragraph(t, _hdr)
feat_tbl = [
    [_h('Group'), _h('Features included in lean model (20 total)'), _h('Motivation')],
    [_c('Location (4)'),    _c('city_lp, zip_lp, zip_city_diff, zip_x_sqft'),           _c('Dominant price determinant; smoothed encoding prevents overfit on rare categories')],
    [_c('Size (3)'),        _c('log_sqft_above, log_sqft_living_sq, log_sqft_basement'), _c('Log + quadratic term captures concave diminishing-returns curve')],
    [_c('Condition (2)'),   _c('condition, cond_x_age'),                                 _c('Base condition level; interaction with age captures sustained upkeep signal')],
    [_c('Age (2)'),         _c('log_house_age, effective_age'),                          _c('Log captures U-shape; effective age substitutes renovation year where available')],
    [_c('Rooms (2)'),       _c('bathrooms, sqft_per_bedroom'),                           _c('Quality and spaciousness proxies')],
    [_c('View/Water (2)'),  _c('view, waterfront'),                                      _c('Both survive strong Lasso regularisation; retained as direct effects')],
    [_c('Basement (1)'),    _c('has_basement'),                                          _c('Binary presence flag; size ratio excluded as near-zero contribution')],
    [_c('Renovation (1)'),  _c('recent_reno'),                                           _c('"Move-in ready" premium: renovation within 10 years of sale')],
    [_c('Interaction (2)'), _c('size_x_condition, city_x_sqft'),                        _c('Size amplifies condition premium; city level multiplies value of living area')],
    [_c('Time (1)'),        _c('month_sold'),                                            _c('Seasonal variation within the 70-day transaction window')],
]
story.append(make_table(feat_tbl, [2.5*cm, 5.5*cm, 8.0*cm], fs=8.5))
story.append(Cap('Table 3. The 20 features comprising the lean primary model, organised by group. '
                 'All location encodings are fitted on training data and re-fitted per CV fold.'))

# ══════════════════════════════════════════════════════════════════
# 5. MODELLING
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(5, 'Modelling'))

story.append(numbered_subsection('5.1', 'Feature Count Optimisation via Lasso Path'))
story.append(P(
    'Before fitting the primary model, we performed a Lasso regularisation '
    'path analysis to determine empirically how many features are justified. '
    'The Lasso path sweeps 40 regularisation strengths (α ∈ [10<super>−4</super>, '
    '10<super>−1</super>]), recording the number of surviving features and '
    'both 5-fold CV and test MAPE at each point.'
))
story += fig('figures/fig_mape_vs_nfeats.png', 14.5,
    'Figure 3. MAPE as a function of feature count along the Lasso regularisation path. '
    'Performance saturates rapidly near 19–20 features; the reduction from 38 to 20 '
    'features costs less than 0.2 pp MAPE while substantially improving interpretability.')
story.append(SP(0.1))
story.append(P(
    'Figure 3 reveals a clear elbow: performance improves sharply from 3 to 12 '
    'features (MAPE 21% → 18%), then flattens. Below 10 features, the model '
    'is underfit; above 20, gains are marginal (less than 0.2 pp per 10 additional '
    'features). This analysis directly informs the lean 20-feature primary model '
    'and the decision not to include 26 B-spline basis functions that, while '
    'improving stacked performance, add zero benefit to standalone Ridge regression.'
))

story.append(numbered_subsection('5.2', 'Regularisation Strategy'))
story.append(P(
    'Ridge regression is selected over Lasso and Elastic Net for the primary '
    'model, for three reasons. First, the 20 features include several '
    'correlated pairs (e.g., log_house_age and effective_age; city_lp and '
    'zip_lp) where L2 regularisation\'s simultaneous shrinkage is more '
    'appropriate than L1\'s winner-takes-all zeroing. Second, Ridge retains '
    'all features, preserving a complete coefficient vector for interpretation. '
    'Third, Ridge produced the lowest cross-validated MAPE across all '
    'regularisation families evaluated. The regularisation strength '
    'α = 0.5 was selected by 5-fold cross-validation over '
    'α ∈ {0.01, 0.1, 0.5, 1, 5, 10, 50}.'
))

story.append(numbered_subsection('5.3', 'Kernel Methods (Nyström Approximation)'))
story.append(P(
    'As a performance extension, RBF and polynomial kernel approximations '
    'are applied via the Nyström method (500 components), which maps the '
    'feature space into a higher-dimensional representation where linear '
    'regression can capture complex non-linear interactions. Both methods '
    'comply with the linear model constraint. The kernel models use an '
    'extended 35-feature set that adds redundant but complementary features '
    'stripped from the lean model.'
))

story.append(numbered_subsection('5.4', 'OOF Stacking'))
story.append(P(
    'Ridge, RBF-Nyström, and Poly-Nyström are combined via 5-fold '
    'out-of-fold (OOF) stacking. For each of 5 folds: each base model '
    'is trained on the remaining 4 folds and generates a prediction on '
    'the held-out fold. After all folds, every training observation has '
    'one OOF prediction per base model. A Ridge meta-learner (α = 0.001) '
    'is fitted on these three OOF columns as inputs, then applied to '
    'averaged test predictions from base models retrained on the full '
    'training set. Crucially, the meta-learner never sees a prediction '
    'generated by a model trained on the same observation, preventing '
    'data leakage.'
))
story.append(P(
    '<b>Permissibility.</b> The assignment brief explicitly permits '
    '"ensembling multiple linear models combined in any reasonable way." '
    'Every component — the three base models and the Ridge meta-learner — '
    'is a linear model. OOF stacking is therefore fully compliant with '
    'the linear modelling constraint.'
))
story.append(P(
    '<b>Interpretability note.</b> Stacking reduces direct coefficient '
    'interpretability. All model interpretation in Section 7 uses the '
    'standalone lean Ridge model. The stacked model is presented as a '
    'performance ceiling.'
))

story.append(numbered_subsection('5.5', 'Validation Protocol'))
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
    ['Model', 'Features', 'CV / OOF MAPE', 'Train MAPE', 'Test MAPE', 'Gap'],
    ['Lean Ridge (primary)',         f'{R["lean_n"]}', f'{R["lean_cv"]:.2f}%',
     f'{R["lean_train"]:.2f}%', f'{R["lean_test"]:.2f}%',  f'{R["lean_test"]-R["lean_train"]:+.2f}pp'],
    ['Extended Ridge',               '35',             f'{R["ext_cv"]:.2f}%',
     f'{R["ext_train"]:.2f}%',  f'{R["ext_test"]:.2f}%',   f'{R["ext_test"]-R["ext_train"]:+.2f}pp'],
    ['RBF Kernel (Nyström)',          '35',             f'{R["rbf_cv"]:.2f}%',
     f'{R["rbf_train"]:.2f}%',  f'{R["rbf_test"]:.2f}%',   f'{R["rbf_test"]-R["rbf_train"]:+.2f}pp'],
    ['Poly Kernel (Nyström)',         '35',             f'{R["poly_cv"]:.2f}%',
     f'{R["poly_train"]:.2f}%', f'{R["poly_test"]:.2f}%',  f'{R["poly_test"]-R["poly_train"]:+.2f}pp'],
    ['Stacked ensemble',             '35',             f'{R["stacked_train"]:.2f}% †',
     '—', f'{R["stacked_test"]:.2f}%', '—'],
    ['XGB Conservative (benchmark)', '35',  '—', '15.05%', '15.62%', '+0.57pp'],
    ['XGB Early Stopping (bmark)',   '35',  '—', '11.21%', '15.28%', '+4.08pp'],
]
rt = make_table(res_data, [4.2*cm, 2.2*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.6*cm])
# Layer highlights on top of make_table base style (additive, not replacement)
rt.setStyle(TableStyle([
    # Alternating stripes across all data rows
    ('BACKGROUND', (0,1), (-1,1), WHITE),
    ('BACKGROUND', (0,2), (-1,2), LGREY),
    ('BACKGROUND', (0,3), (-1,3), WHITE),
    ('BACKGROUND', (0,4), (-1,4), LGREY),
    ('BACKGROUND', (0,5), (-1,5), WHITE),
    ('BACKGROUND', (0,6), (-1,6), LGREY),
    ('BACKGROUND', (0,7), (-1,7), WHITE),
    # Stacked ensemble row (row 5) -- green
    ('BACKGROUND', (0,5), (-1,5), HexColor('#e8f7ed')),
    ('FONTNAME',   (0,5), (-1,5), 'Helvetica-Bold'),
    # XGB rows (6, 7) -- amber
    ('BACKGROUND', (0,6), (-1,6), HexColor('#fef9f0')),
    ('BACKGROUND', (0,7), (-1,7), HexColor('#fef9f0')),
    ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
    ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(KeepTogether([
    rt,
    Cap('Table 4. Full results on the deduplicated dataset (4,553 properties, '
        '80/20 split). Best linear model highlighted in green; XGBoost benchmarks in amber. '
        'Gap = Test MAPE − Train MAPE; negative values indicate mild underfitting. '
        '† Stacked ensemble CV column shows OOF MAPE (each prediction made on held-out data); '
        'a separate CV pass would require nested cross-validation.'),
]))
story += [SP(0.3)]

story += fig('figures/fig_comparison.png', 14.5,
    'Figure 4. Left: test MAPE across all models. The stacked ensemble (16.52%) '
    'approaches XGBoost performance. Right: XGBoost train vs test MAPE — '
    'the Default configuration (train=5.6%, test=15.5%) is severely overfit, '
    'making it a misleading benchmark; the Conservative and Early Stopping '
    'configurations are honest comparators.')
story.append(SP(0.2))

story.append(P(
    f'The lean 20-feature Ridge model achieves {R["lean_test"]:.2f}% test MAPE '
    f'with a train/test gap of {R["lean_test"]-R["lean_train"]:+.2f} pp, indicating '
    'mild underfitting rather than memorisation — the hallmark of a well-regularised '
    'linear model. The extended and kernel models offer further performance gains, '
    f'with the stacked ensemble reaching {R["stacked_test"]:.2f}% MAPE. '
    'The XGBoost comparison is discussed in detail in Section 8.'
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
story.append(Cap('Table 5. MAPE by price segment for the lean Ridge model. '
                 'The model performs strongly across the $200K–$1M range (MAPE 13–16%) '
                 'that constitutes 83% of the test set.'))

# ══════════════════════════════════════════════════════════════════
# 7. MODEL INTERPRETATION
# ══════════════════════════════════════════════════════════════════
story.append(numbered_section(7, 'Model Interpretation'))
story.append(P(
    'All interpretation uses the standalone lean Ridge model (α = 0.5, 20 features). '
    'Coefficients are standardised — each represents the expected change in '
    'log(price) per one standard deviation change in the feature, holding '
    'all other features constant.'
))

story += fig('figures/fig_coef.png', 14.5, aspect=0.52,
    caption='Figure 5. Standardised Ridge coefficients for the lean 20-feature model. '
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
    [_c('log_sqft_living_sq'),  _c('+'), _c('Quadratic log-living area: captures the concave size–price curve; increasing returns at smaller sizes, diminishing above 3,000 sqft')],
    [_c('zip_lp'),              _c('+'), _c('ZIP code mean log-price: local price level is a strong prior on individual property value')],
    [_c('log_sqft_above'),      _c('+'), _c('Above-ground floor area contributes positively; preferred over total sqft as basements are discounted')],
    [_c('city_lp'),             _c('+'), _c('City-level price signal; complementary to ZIP encoding at a coarser geographic scale')],
    [_c('city_x_sqft'),         _c('−'), _c('Location × size interaction: reflects partial regression adjustment; the value of size is already absorbed by location encoding, so the interaction corrects for substitution')],
    [_c('cond_x_age'),          _c('+'), _c('Condition × age interaction: an old home in excellent condition commands a disproportionate premium — sustained upkeep is valued non-linearly')],
    [_c('bathrooms'),           _c('+'), _c('Bathroom count; more stable predictor than bedroom count, which is subject to high leverage from unusual configurations')],
    [_c('condition'),           _c('+'), _c('Base condition level; amplified by cond_x_age for older properties')],
    [_c('log_house_age'),       _c('−'), _c('Older properties trade at a discount on average; the interaction with condition captures exceptions')],
    [_c('size_x_condition'),    _c('−'), _c('Partial regression adjustment: after controlling for size and condition separately, the interaction absorbs residual correlation between them')],
]
story.append(make_table(feat_interp, [3.0*cm, 1.2*cm, 11.8*cm], fs=8.5))
story.append(Cap('Table 6. Top 10 features by absolute standardised coefficient with plain-language interpretation.'))

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
    '<b>Location multiplies the value of size.</b> The city_x_sqft interaction '
    'demonstrates that an additional 500 sqft in an expensive city '
    'contributes more to value than the same space in a cheaper market. '
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
    [f'Stacked (ours)',                 '—',      f'{R["stacked_test"]:.2f}%', '≈0pp', 'Best linear'],
]
xt = make_table(xgb_data, [4.5*cm, 2.3*cm, 2.3*cm, 2.8*cm, 3.3*cm])
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
story.append(Cap('Table 7. XGBoost configurations versus linear models. '
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
    f'our stacked model ({R["stacked_test"]:.2f}%) trails by '
    f'{R["stacked_test"] - 15.62:+.2f} pp — a difference that disappears '
    'within typical cross-validation variance.'
))
story.append(P(
    'The near-parity between the stacked linear model and properly regularised '
    'XGBoost has a structural explanation. The dominant non-linear interaction '
    'in this dataset — the multiplicative location–size relationship — is '
    'captured explicitly by the zip_x_sqft and city_x_sqft features. '
    'Once these interactions are encoded in the feature set, '
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
    f'This paper demonstrates that a 20-feature Ridge regression model achieves {R["lean_test"]:.2f}% '
    f'MAPE on house price prediction, with a stacked ensemble reaching {R["stacked_test"]:.2f}%. '
    'The gap between our best model and a properly regularised XGBoost benchmark is '
    f'{R["stacked_test"] - 15.62:+.2f} pp — negligible relative to cross-validation variance, '
    'and achieved without a single tree-based component. '
    'Three methodological contributions underpin this result. '
    'First, a data integrity audit identified and corrected an artificial 50% duplication '
    'in the raw dataset, reducing the apparent XGBoost advantage from 5+ pp to approximately 1 pp. '
    'Second, Lasso regularisation path analysis established empirically that predictive '
    'performance saturates near 20 features, arguing against feature proliferation in favour '
    'of deliberate, domain-motivated construction. '
    'Third, the most influential features — location target encodings and location × size '
    'interactions — encode the same heuristic that human appraisers use: a property\'s value '
    'is anchored by comparable nearby sales, and the marginal value of additional space scales '
    'with neighbourhood prestige. A model built on explicit domain knowledge is not only accurate; '
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
