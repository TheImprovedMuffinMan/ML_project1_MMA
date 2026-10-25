"""
Feature categories and BRFSS encoding specifications.
"""

# ============================================================================ #
# 1) Identifiants, logistique de l'appel, poids de sondage
COLS_ADMIN = [
    "_STATE",
    "Id",
    "SEQNO",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "_PSU",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_LLCPWT",
    "_CLLCPWT",
    "_DUALUSE",
    "_DUALCOR",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "CTELNUM1",
    "CELLFON2",
    "PVTRESD2",
    "CCLGHOUS",
    "CSTATE",
    "LANDLINE",
    "QSTVER",
    "QSTLANG",
    "LADULT",
    "CADULT",
    "NUMHHOL2",
    "SXORIENT",
    "TRNSGNDR",
    "CPDEMO1",
    "ARTHDIS2",
    "FLSHTMY2",
    "IMFVPLAC",
    "HIVTSTD3",
    "WHRTST10",
    "CAREGIV1",
    "CRGVREL1",
    "CRGVHRS1",
    "CIMEMLOS",
    "SXORIENT",
    "TRNSGNDR",
    "MSCODE",
    "HLTHPLN1",
    "ARTHSOCL",
]

# 2) Variables redondantes (on garde une seule version par concept)
COLS_REDUNDANT = [
    # IMC -> on garde _BMI5
    "WEIGHT2",
    "HEIGHT3",
    "HTIN4",
    "HTM4",
    "WTKG3",
    "_BMI5CAT",
    "_RFBMI5",
    "HEIGHT2",
    # Âge -> on garde _AGE80
    "_AGE65YR",
    "_AGE_G",
    "_AGEG5YR",
    # Tabac -> on garde _SMOKER3 (+ USENOW3)
    "SMOKE100",
    "SMOKDAY2",
    "LASTSMK2",
    "STOPSMK2",
    "_RFSMOK3",
    # Alcool -> on garde _DRNKWEK, _RFBING5, _RFDRHV5
    "ALCDAY5",
    "AVEDRNK2",
    "DRNK3GE5",
    "MAXDRNKS",
    "DRNKANY5",
    "DROCDY3_",
    # Hypertension / cholestérol -> on garde _RFHYPE5, _RFCHOL, _CHOLCHK
    "BPHIGH4",
    "TOLDHI2",
    "BLOODCHO",
    "CHOLCHK",
    # Activité physique -> on garde _TOTINDA, _PACAT1, PA1MIN_, _PASTRNG
    "EXERANY2",
    "EXRACT11",
    "EXEROFT1",
    "EXERHMM1",
    "EXRACT21",
    "EXEROFT2",
    "EXERHMM2",
    "STRENGTH",
    "METVL11_",
    "METVL21_",
    "ACTIN11_",
    "ACTIN21_",
    "PADUR1_",
    "PADUR2_",
    "PAFREQ1_",
    "PAFREQ2_",
    "_MINAC11",
    "_MINAC21",
    "STRFREQ_",
    "PAMISS1_",
    "PAMIN11_",
    "PAMIN21_",
    "PAVIG11_",
    "PAVIG21_",
    "PA1VIGM_",
    "_PAINDX1",
    "_PA150R2",
    "_PA300R2",
    "_PA30021",
    "_PAREC1",
    "_PASTAE1",
    # Fruits / légumes -> on garde _FRUTSUM, _VEGESUM
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "FTJUDA1_",
    "FRUTDA1_",
    "BEANDAY_",
    "GRENDAY_",
    "ORNGDAY_",
    "VEGEDA1_",
    "_MISFRTN",
    "_MISVEGN",
    "_FRTRESP",
    "_VEGRESP",
    "_FRTLT1",
    "_VEGLT1",
    "_FRT16",
    "_VEG23",
    "_FRUITEX",
    "_VEGETEX",
    # Origine ethnique -> on garde _RACE
    "_PRACE1",
    "_MRACE1",
    "_HISPANC",
    "_RACEG21",
    "_RACEGR3",
    "_RACE_G1",
    "_CHISPNC",
    "_CRACE1",
    "_CPRACE",
    # Revenu / éducation / santé générale / ménage -> on garde _INCOMG, _EDUCAG, GENHLTH, _CHLDCNT
    "INCOME2",
    "EDUCA",
    "_RFHLTH",
    "CHILDREN",
    "NUMADULT",
    "NUMMEN",
    "NUMWOMEN",
    "HHADULT",
    # Ceinture, vaccins, VIH, asthme, arthrite -> versions calculées gardées
    "SEATBELT",
    "_RFSEAT2",
    "FLUSHOT6",
    "PNEUVAC3",
    "HIVTST6",
    "_CASTHM1",
    "ASTHMA3",
    "ASTHNOW",
    "HAVARTH3",
]

# 3) Colonnes vides à plus de 80 % sur x_train (modules optionnels : diabète,
#    aidants, vision, asthme détaillé, dépistages cancer, dépression...)
COLS_MOSTLY_NAN = [
    "DIABAGE2",
    "NUMPHON2",
    "PREGNANT",
    "PDIABTST",
    "PREDIAB1",
    "INSULIN",
    "BLDSUGAR",
    "FEETCHK2",
    "DOCTDIAB",
    "CHKHEMO3",
    "FEETCHK",
    "EYEEXAM",
    "DIABEYE",
    "DIABEDU",
    "CRGVREL1",
    "CRGVLNG1",
    "CRGVHRS1",
    "CRGVPRB1",
    "CRGVPERS",
    "CRGVHOUS",
    "CRGVMST2",
    "CRGVEXPT",
    "VIDFCLT2",
    "VIREDIF3",
    "VIPRFVS2",
    "VINOCRE2",
    "VIEYEXM2",
    "VIINSUR2",
    "VICTRCT4",
    "VIGLUMA2",
    "VIMACDG2",
    "CDHOUSE",
    "CDASSIST",
    "CDHELP",
    "CDSOCIAL",
    "CDDISCUS",
    "WTCHSALT",
    "LONGWTCH",
    "DRADVISE",
    "ASTHMAGE",
    "ASATTACK",
    "ASERVIST",
    "ASDRVIST",
    "ASRCHKUP",
    "ASACTLIM",
    "ASYMPTOM",
    "ASNOSLEP",
    "ASTHMED3",
    "ASINHALR",
    "HAREHAB1",
    "STREHAB1",
    "CVDASPRN",
    "ASPUNSAF",
    "RLIVPAIN",
    "RDUCHART",
    "RDUCSTRK",
    "ARTTODAY",
    "ARTHWGT",
    "ARTHEXER",
    "ARTHEDU",
    "TETANUS",
    "HPVADVC2",
    "HPVADSHT",
    "SHINGLE2",
    "HADMAM",
    "HOWLONG",
    "HADPAP2",
    "LASTPAP2",
    "HPVTEST",
    "HPLSTTST",
    "HADHYST2",
    "PROFEXAM",
    "LENGEXAM",
    "BLDSTOOL",
    "LSTBLDS3",
    "HADSIGM3",
    "HADSGCO1",
    "LASTSIG3",
    "PCPSAAD2",
    "PCPSADI1",
    "PCPSARE1",
    "PSATEST1",
    "PSATIME",
    "PCPSARS1",
    "PCPSADE1",
    "PCDMDECN",
    "SCNTMNY1",
    "SCNTMEL1",
    "SCNTPAID",
    "SCNTWRK1",
    "SCNTLPAD",
    "SCNTLWK1",
    "RCSGENDR",
    "RCSRLTN2",
    "CASTHDX2",
    "CASTHNO2",
    "EMTSUPRT",
    "LSATISFY",
    "ADPLEASR",
    "ADDOWN",
    "ADSLEEP",
    "ADENERGY",
    "ADEAT1",
    "ADFAIL",
    "ADTHINK",
    "ADMOVE",
    "MISTMNT",
    "ADANXEV",
]

COLS_TO_DROP = COLS_ADMIN + COLS_REDUNDANT + COLS_MOSTLY_NAN

# ============================================================================ #
# Binary / binary-like categorical variables

BINARY_SPECS = {
    "MEDCOST": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "CVDSTRK3": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "CHCSCNCR": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "CHCOCNCR": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "CHCCOPD1": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "ADDEPEV2": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "CHCKIDNY": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "VETERAN3": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "INTERNET": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "QLACTLM2": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "USEEQUIP": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "BLIND": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "DECIDE": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "DIFFWALK": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "DIFFDRES": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "DIFFALON": {
        "positive": [1],
        "negative": [2],
        "unknown": [7, 9],
    },
    "_RFHYPE5": {
        "positive": [2],
        "negative": [1],
        "unknown": [9],
    },
    "_LTASTH1": {
        "positive": [2],
        "negative": [1],
        "unknown": [9],
    },
    "_AGE65YR": {
        "positive": [2],
        "negative": [1],
        "unknown": [3],
    },
    "_AIDTST3": {
        "positive": [1],
        "negative": [2],
        "unknown": [9],
    },
    "_RFSEAT3": {
        "positive": [1],
        "negative": [2],
        "unknown": [9],
    },
    "_FLSHOT6": {
        "positive": [1],
        "negative": [2],
        "unknown": [0, 9],
    },
    "_PNEUMO2": {
        "positive": [1],
        "negative": [2],
        "unknown": [0, 9],
    },
    "_PASTRNG": {
        "positive": [1],
        "negative": [2],
        "unknown": [9],
    },
    "LMTJOIN3": {
        "positive": [1],
        "negative": [2],
        "unknown": [0, 7, 9],
    },
    "_HCVU651": {
        "positive": [1],
        "negative": [2],
        "unknown": [9],
    },
    "_RFCHOL": {
        "positive": [2],
        "negative": [1],
        "unknown": [9],
    },
    "_RFBING5": {
        "positive": [2],
        "negative": [1],
        "unknown": [9],
    },
    "_RFDRHV5": {
        "positive": [2],
        "negative": [1],
        "unknown": [9],
    },
    "_TOTINDA": {
        "positive": [1],
        "negative": [2],
        "unknown": [9],
    },
    "BPMEDS": {
        "positive": [1],
        "negative": [2],
        "unknown": [0, 7, 9],
    },
}


# ============================================================================ #
# True binary variables
DIRECT_BINARY_SPECS = {
    # 1 = Male, 2 = Female
    # Female -> 1, Male -> 0
    "SEX": {
        "positive": [2],
        "negative": [1],
    },
    # 1 = Arthritis, 2 = No arthritis
    "_DRDXAR1": {
        "positive": [1],
        "negative": [2],
    },
}


# ============================================================================ #
# Numerical / ordinal variables requiring special treatment
NUMERIC_SPECS = {
    "GENHLTH": {
        "unknown": [7, 9],
        "fill_value": 0,
    },
    "_CHOLCHK": {
        "unknown": [9],
        "fill_value": 0,
    },
    "_CHLDCNT": {
        "unknown": [9],
        "fill_value": 0,
    },
    "_EDUCAG": {
        "unknown": [9],
        "fill_value": 0,
    },
    "_INCOMG": {
        "unknown": [9],
        "fill_value": 0,
    },
    "_PACAT1": {
        "unknown": [9],
        "fill_value": 0,
    },
    "CHECKUP1": {
        "replace": {8: 5},
        "unknown": [7, 9],
        "fill_value": 0,
    },
    "PHYSHLTH": {
        "replace": {88: 0},
        "unknown": [77, 99],
        "fill_value": 0,
    },
    "MENTHLTH": {
        "replace": {88: 0},
        "unknown": [77, 99],
        "fill_value": 0,
    },
    "POORHLTH": {
        "replace": {88: 0},
        "unknown": [77, 99],
        "fill_value": 0,
    },
    "_DRNKWEK": {
        "unknown": [99900],
        "fill_value": 0,
    },
    "MAXVO2_": {
        "unknown": [999],
        "fill_value": 0,
    },
    "FC60_": {
        "unknown": [999],
        "fill_value": 0,
    },
    "JOINPAIN": {
        "unknown": [77, 99],
        "fill_value": 0,
    },
}


# ============================================================================ #
# Nominal categorical variables

NOMINAL_SPECS = {
    "PERSDOC2": {
        "unknown": [7, 9],
    },
    "RENTHOM1": {
        "unknown": [7, 9],
    },
    "USENOW3": {
        "unknown": [7, 9],
    },
    "_ASTHMS1": {
        "unknown": [9],
    },
    "_RACE": {
        "unknown": [9],
    },
    "_SMOKER3": {
        "unknown": [9],
    },
    "_LMTACT1": {
        "unknown": [9],
    },
    "_LMTWRK1": {
        "unknown": [9],
    },
    "_LMTSCL1": {
        "unknown": [9],
    },
    "EMPLOY1": {
        "unknown": [9],
    },
    "MARITAL": {
        "unknown": [9],
    },
    "DIABETE3": {
        "unknown": [7, 9],
    },
}


# ============================================================================ #
# Continuous variables to standardise

CONTINUOUS_FEATURES = [
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "_AGE80",
    "_BMI5",
    "_DRNKWEK",
    "_FRUTSUM",
    "_VEGESUM",
    "MAXVO2_",
    "FC60_",
    "PA1MIN_",
]


# ============================================================================ #
# Ordinal variables that may also be standardised

ORDINAL_FEATURES = [
    "GENHLTH",
    "CHECKUP1",
    "_CHOLCHK",
    "_CHLDCNT",
    "_EDUCAG",
    "_INCOMG",
    "_PACAT1",
]


FEATURES_TO_SCALE = CONTINUOUS_FEATURES + ORDINAL_FEATURES

# ============================================================================ #
# Features to impute with median values (for continuous variables)
MEDIAN_IMPUTE_FEATURES = [
    "_BMI5",
    "_FRUTSUM",
    "_VEGESUM",
    "PA1MIN_",
]
