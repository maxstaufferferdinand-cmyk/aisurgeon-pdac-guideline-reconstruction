"""PubMed chapter queries for the ESMO PDAC 2015 -> August 2023 PoC.

The date restriction is intentionally NOT embedded in these queries. The runner
passes publication-date limits to NCBI via datetype=pdat, mindate, and maxdate,
which enables deterministic half-year batching and automatic sub-splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True, slots=True)
class ChapterQuery:
    chapter_id: str
    title: str
    query: str


def _clean(text: str) -> str:
    return dedent(text).strip()


PDAC_CORE = _clean(
    r'''
    (
      "Pancreatic Neoplasms"[Mesh]
      OR "pancreatic cancer"[tiab]
      OR "pancreatic cancers"[tiab]
      OR "pancreatic carcinoma"[tiab]
      OR "pancreatic carcinomas"[tiab]
      OR "pancreatic adenocarcinoma"[tiab]
      OR "pancreatic ductal adenocarcinoma"[tiab]
      OR PDAC[tiab]
      OR "exocrine pancreatic cancer"[tiab]
      OR "cancer of the pancreas"[tiab]
    )
    '''
)

PDAC_WITH_PRECURSORS = _clean(
    r'''
    (
      "Pancreatic Neoplasms"[Mesh]
      OR "Pancreatic Cyst"[Mesh]
      OR "pancreatic cancer"[tiab]
      OR "pancreatic cancers"[tiab]
      OR "pancreatic carcinoma"[tiab]
      OR "pancreatic carcinomas"[tiab]
      OR "pancreatic adenocarcinoma"[tiab]
      OR "pancreatic ductal adenocarcinoma"[tiab]
      OR PDAC[tiab]
      OR "exocrine pancreatic cancer"[tiab]
      OR "cancer of the pancreas"[tiab]
      OR PanIN[tiab]
      OR "pancreatic intraepithelial neoplasia"[tiab]
      OR IPMN[tiab]
      OR "intraductal papillary mucinous neoplasm"[tiab]
      OR "intraductal papillary mucinous neoplasms"[tiab]
      OR "mucinous cystic neoplasm"[tiab]
      OR "mucinous cystic neoplasms"[tiab]
    )
    '''
)

PDAC_WITH_RARE_EXOCRINE = _clean(
    r'''
    (
      "Pancreatic Neoplasms"[Mesh]
      OR "pancreatic cancer"[tiab]
      OR "pancreatic cancers"[tiab]
      OR "pancreatic carcinoma"[tiab]
      OR "pancreatic carcinomas"[tiab]
      OR "pancreatic adenocarcinoma"[tiab]
      OR "pancreatic ductal adenocarcinoma"[tiab]
      OR PDAC[tiab]
      OR "exocrine pancreatic cancer"[tiab]
      OR "cancer of the pancreas"[tiab]
      OR "pancreatic acinar cell carcinoma"[tiab]
      OR "acinar cell carcinoma of the pancreas"[tiab]
    )
    '''
)

COMMON_LIMITS = _clean(
    r'''
    NOT
    (
      "Animals"[Mesh]
      NOT
      "Humans"[Mesh]
    )
    NOT
    (
      "Pancreatic Neuroendocrine Tumors"[Majr]
      OR "pancreatic neuroendocrine tumor"[ti]
      OR "pancreatic neuroendocrine tumors"[ti]
      OR "pancreatic neuroendocrine tumour"[ti]
      OR "pancreatic neuroendocrine tumours"[ti]
      OR pNET[ti]
      OR pNETs[ti]
    )
    '''
)


def _compose(disease_block: str, topic_block: str) -> str:
    return f"{disease_block}\nAND\n({topic_block.strip()})\n{COMMON_LIMITS}"


CHAPTER_1_TOPIC = _clean(
    r'''
    (
      epidemiology[sh]
      OR "Incidence"[Mesh]
      OR "Prevalence"[Mesh]
      OR "Mortality"[Mesh]
      OR "Survival Rate"[Mesh]
      OR epidemiolog*[tiab]
      OR incidence[tiab]
      OR prevalence[tiab]
      OR mortality[tiab]
      OR "cancer mortality"[tiab]
      OR "cancer-specific mortality"[tiab]
      OR "five-year survival"[tiab]
      OR "5-year survival"[tiab]
      OR "population survival"[tiab]
      OR "population-based"[tiab]
      OR "cancer registry"[tiab]
      OR "cancer registries"[tiab]
      OR "registry-based"[tiab]
      OR "age-standardized"[tiab]
      OR "age-standardised"[tiab]
      OR "age-specific incidence"[tiab]
      OR "sex-specific incidence"[tiab]
      OR "incidence trend"[tiab]
      OR "incidence trends"[tiab]
      OR "mortality trend"[tiab]
      OR "mortality trends"[tiab]
      OR "temporal trend"[tiab]
      OR "temporal trends"[tiab]
      OR projection*[tiab]
      OR burden[tiab]
      OR demographic*[tiab]
      OR geographic*[tiab]
    )
    OR
    (
      (
        "Genetic Predisposition to Disease"[Mesh]
        OR "family history"[tiab]
        OR "familial pancreatic cancer"[tiab]
        OR familial[tiab]
        OR hereditary[tiab]
        OR inherited[tiab]
        OR germline[tiab]
        OR "high-risk individual"[tiab]
        OR "high-risk individuals"[tiab]
        OR "high-risk population"[tiab]
        OR "high-risk populations"[tiab]
        OR BRCA1[tiab]
        OR BRCA2[tiab]
        OR CDKN2A[tiab]
        OR p16[tiab]
        OR ATM[tiab]
        OR STK11[tiab]
        OR PRSS1[tiab]
        OR PRSS2[tiab]
        OR SPINK1[tiab]
        OR PALB2[tiab]
        OR "mismatch repair"[tiab]
        OR Lynch[tiab]
        OR "hereditary pancreatitis"[tiab]
        OR "hereditary breast and ovarian cancer"[tiab]
        OR "Peutz-Jeghers"[tiab]
        OR "ataxia telangiectasia"[tiab]
        OR "familial atypical multiple mole melanoma"[tiab]
        OR FAMMM[tiab]
        OR "Li-Fraumeni"[tiab]
      )
      AND
      (
        risk[tiab]
        OR risks[tiab]
        OR predisposition[tiab]
        OR susceptibility[tiab]
        OR penetrance[tiab]
        OR screening[tiab]
        OR surveillance[tiab]
        OR "early detection"[tiab]
      )
    )
    OR
    (
      (
        "Risk Factors"[Mesh]
        OR smoking[tiab]
        OR tobacco[tiab]
        OR cigarette*[tiab]
        OR "environmental tobacco smoke"[tiab]
        OR "secondhand smoke"[tiab]
        OR obesity[tiab]
        OR overweight[tiab]
        OR "body mass index"[tiab]
        OR BMI[tiab]
        OR diabetes[tiab]
        OR "diabetes mellitus"[tiab]
        OR "new-onset diabetes"[tiab]
        OR pancreatitis[tiab]
        OR "chronic pancreatitis"[tiab]
        OR alcohol[tiab]
        OR "Helicobacter pylori"[tiab]
        OR "hepatitis B"[tiab]
        OR HBV[tiab]
        OR HIV[tiab]
        OR "human immunodeficiency virus"[tiab]
        OR ABO[tiab]
        OR "non-O blood group"[tiab]
        OR diet*[tiab]
        OR "red meat"[tiab]
        OR "processed meat"[tiab]
        OR "saturated fat"[tiab]
        OR butter[tiab]
        OR fruit*[tiab]
        OR vegetable*[tiab]
        OR folate[tiab]
        OR "occupational exposure"[tiab]
        OR "occupational exposures"[tiab]
        OR "environmental exposure"[tiab]
        OR "environmental exposures"[tiab]
        OR chemical*[tiab]
        OR chlorobenzoil[tiab]
        OR chlorobenzoyl[tiab]
        OR "chlorinated hydrocarbon"[tiab]
        OR "chlorinated hydrocarbons"[tiab]
        OR nickel[tiab]
        OR chromium[tiab]
        OR silica[tiab]
      )
      AND
      (
        risk[tiab]
        OR risks[tiab]
        OR association[tiab]
        OR associated[tiab]
        OR etiology[tiab]
        OR aetiology[tiab]
        OR causal[tiab]
        OR causation[tiab]
        OR attributable[tiab]
        OR prevention[tiab]
        OR preventable[tiab]
        OR modifiable[tiab]
      )
    )
    '''
)

CHAPTER_2_TOPIC = _clean(
    r'''
    (
      "Diagnosis"[Mesh]
      OR diagnos*[tiab]
      OR "clinical presentation"[tiab]
      OR "presenting symptom"[tiab]
      OR "presenting symptoms"[tiab]
      OR "early symptom"[tiab]
      OR "early symptoms"[tiab]
      OR "early detection"[tiab]
      OR "mass effect"[tiab]
      OR jaundice[tiab]
      OR icterus[tiab]
      OR "abdominal pain"[tiab]
      OR "back pain"[tiab]
      OR "weight loss"[tiab]
      OR steatorrhea[tiab]
      OR steatorrhoea[tiab]
      OR "new-onset diabetes"[tiab]
      OR "new onset diabetes"[tiab]
      OR "bile duct obstruction"[tiab]
      OR "biliary obstruction"[tiab]
      OR "pancreatic duct obstruction"[tiab]
      OR "duodenal obstruction"[tiab]
      OR "gastric outlet obstruction"[tiab]
      OR "tumor location"[tiab]
      OR "tumour location"[tiab]
      OR "pancreatic head"[tiab]
      OR "pancreatic body"[tiab]
      OR "pancreatic tail"[tiab]
    )
    OR
    (
      histopatholog*[tiab]
      OR patholog*[tiab]
      OR morphology[tiab]
      OR morphologic*[tiab]
      OR histolog*[tiab]
      OR immunohistochem*[tiab]
      OR differentiation[tiab]
      OR "well differentiated"[tiab]
      OR "poorly differentiated"[tiab]
      OR "duct-forming"[tiab]
      OR "ductal phenotype"[tiab]
      OR "ductal cell of origin"[tiab]
      OR "acinar cell carcinoma"[tiab]
      OR "acinar cell carcinomas"[tiab]
      OR "colloid carcinoma"[tiab]
      OR "medullary carcinoma"[tiab]
      OR "adenosquamous carcinoma"[tiab]
      OR "undifferentiated carcinoma"[tiab]
      OR "osteoclast-like giant cell"[tiab]
      OR desmoplasia[tiab]
      OR desmoplastic[tiab]
      OR stroma[tiab]
      OR stromal[tiab]
      OR "tumor microenvironment"[tiab]
      OR "tumour microenvironment"[tiab]
      OR "stromal barrier"[tiab]
      OR "drug delivery"[tiab]
      OR "chemotherapy resistance"[tiab]
      OR "cystic neoplasm"[tiab]
      OR "cystic neoplasms"[tiab]
      OR "serous cystadenoma"[tiab]
      OR "serous cystic neoplasm"[tiab]
      OR "mucinous cystadenoma"[tiab]
      OR "mucinous cystadenocarcinoma"[tiab]
      OR "malignant progression"[tiab]
      OR "malignant transformation"[tiab]
      OR precursor*[tiab]
      OR "precursor lesion"[tiab]
      OR "precursor lesions"[tiab]
      OR "adenoma-carcinoma sequence"[tiab]
    )
    OR
    (
      carcinogenesis[tiab]
      OR tumorigenesis[tiab]
      OR tumourigenesis[tiab]
      OR genomic*[tiab]
      OR genome[tiab]
      OR "whole-genome sequencing"[tiab]
      OR "whole genome sequencing"[tiab]
      OR transcriptom*[tiab]
      OR proteom*[tiab]
      OR "copy number variation"[tiab]
      OR "copy-number variation"[tiab]
      OR "structural variation"[tiab]
      OR "structural variants"[tiab]
      OR "chromosomal rearrangement"[tiab]
      OR "chromosomal rearrangements"[tiab]
      OR "gene disruption"[tiab]
      OR "gene disruptions"[tiab]
      OR "somatic aberration"[tiab]
      OR "somatic aberrations"[tiab]
      OR "molecular subtype"[tiab]
      OR "molecular subtypes"[tiab]
      OR "genomic subtype"[tiab]
      OR "genomic subtypes"[tiab]
      OR "stable subtype"[tiab]
      OR "locally rearranged"[tiab]
      OR "scattered subtype"[tiab]
      OR "unstable subtype"[tiab]
      OR
      (
        (
          KRAS[tiab]
          OR TP53[tiab]
          OR CDKN2A[tiab]
          OR p16[tiab]
          OR SMAD4[tiab]
          OR MLH1[tiab]
          OR hMLH1[tiab]
          OR MSH2[tiab]
          OR ARID1A[tiab]
          OR ROBO2[tiab]
          OR KDM6A[tiab]
          OR PREX2[tiab]
        )
        AND
        (
          mutation*[tiab]
          OR alteration*[tiab]
          OR aberration*[tiab]
          OR inactivation[tiab]
          OR activation[tiab]
          OR rearrangement*[tiab]
          OR subtype*[tiab]
          OR carcinogenesis[tiab]
          OR tumorigenesis[tiab]
          OR tumourigenesis[tiab]
        )
      )
    )
    '''
)

CHAPTER_3_TOPIC = _clean(
    r'''
    (
      "Neoplasm Staging"[Mesh]
      OR staging[tiab]
      OR "stage classification"[tiab]
      OR TNM[tiab]
      OR AJCC[tiab]
      OR "clinical stage"[tiab]
      OR "radiologic stage"[tiab]
      OR "radiological stage"[tiab]
      OR "anatomic resectability"[tiab]
      OR "anatomical resectability"[tiab]
      OR "biological resectability"[tiab]
      OR "biologic resectability"[tiab]
      OR "conditional resectability"[tiab]
      OR resectab*[tiab]
      OR unresectab*[tiab]
      OR "borderline resectable"[tiab]
      OR "vascular invasion"[tiab]
      OR "vascular involvement"[tiab]
      OR "vessel involvement"[tiab]
      OR "vessel contact"[tiab]
      OR "tumor-vessel contact"[tiab]
      OR "tumour-vessel contact"[tiab]
      OR abutment[tiab]
      OR encasement[tiab]
      OR deformation[tiab]
      OR distortion[tiab]
      OR "superior mesenteric artery"[tiab]
      OR SMA[tiab]
      OR "celiac axis"[tiab]
      OR "coeliac axis"[tiab]
      OR "common hepatic artery"[tiab]
      OR "accessory right hepatic artery"[tiab]
      OR "variant arterial anatomy"[tiab]
      OR "portal vein"[tiab]
      OR "superior mesenteric vein"[tiab]
      OR SMV[tiab]
      OR "portomesenteric vein"[tiab]
      OR "portal vein thrombosis"[tiab]
      OR "venous thrombosis"[tiab]
      OR "lymph node staging"[tiab]
      OR "nodal staging"[tiab]
      OR "lymph node metastasis"[tiab]
      OR "liver metastasis"[tiab]
      OR "hepatic metastasis"[tiab]
      OR "peritoneal metastasis"[tiab]
      OR "occult metastasis"[tiab]
    )
    OR
    (
      "CA-19-9 Antigen"[Mesh]
      OR "CA 19-9"[tiab]
      OR CA19-9[tiab]
      OR "carbohydrate antigen 19-9"[tiab]
      OR "Lewis antigen"[tiab]
      OR "Lewis blood group"[tiab]
      OR "Lewis-negative"[tiab]
      OR "Lewis negative"[tiab]
      OR
      (
        (bilirubin[tiab] OR cholestasis[tiab])
        AND
        ("CA 19-9"[tiab] OR CA19-9[tiab] OR "tumor marker"[tiab] OR "tumour marker"[tiab])
      )
      OR "preoperative CA 19-9"[tiab]
      OR "preoperative CA19-9"[tiab]
      OR "baseline CA 19-9"[tiab]
      OR "baseline CA19-9"[tiab]
      OR "disease burden"[tiab]
      OR "tumor burden"[tiab]
      OR "tumour burden"[tiab]
    )
    OR
    (
      "Tomography, X-Ray Computed"[Mesh]
      OR "Magnetic Resonance Imaging"[Mesh]
      OR "Endosonography"[Mesh]
      OR imaging[tiab]
      OR radiolog*[tiab]
      OR "computed tomography"[tiab]
      OR "CT scan"[tiab]
      OR "CT scans"[tiab]
      OR "multidetector computed tomography"[tiab]
      OR MDCT[tiab]
      OR "pancreatic protocol CT"[tiab]
      OR "pancreas protocol CT"[tiab]
      OR "CT angiography"[tiab]
      OR "arterial phase"[tiab]
      OR "pancreatic phase"[tiab]
      OR "portal venous phase"[tiab]
      OR MRI[tiab]
      OR "magnetic resonance imaging"[tiab]
      OR MRCP[tiab]
      OR "magnetic resonance cholangiopancreatography"[tiab]
      OR EUS[tiab]
      OR "endoscopic ultrasound"[tiab]
      OR endosonograph*[tiab]
      OR "diagnostic accuracy"[tiab]
      OR sensitivity[tiab]
      OR specificity[tiab]
      OR "radiology reporting template"[tiab]
      OR "structured reporting"[tiab]
    )
    OR
    (
      "fine needle aspiration"[tiab]
      OR "fine-needle aspiration"[tiab]
      OR EUS-FNA[tiab]
      OR FNA[tiab]
      OR "fine needle biopsy"[tiab]
      OR "fine-needle biopsy"[tiab]
      OR EUS-FNB[tiab]
      OR FNB[tiab]
      OR biopsy[tiab]
      OR cytolog*[tiab]
      OR "tissue acquisition"[tiab]
      OR "tissue confirmation"[tiab]
      OR "histologic confirmation"[tiab]
      OR "histological confirmation"[tiab]
      OR "lymph node sampling"[tiab]
      OR "liver biopsy"[tiab]
      OR "metastasis biopsy"[tiab]
      OR "needle tract seeding"[tiab]
      OR "needle-track seeding"[tiab]
      OR "percutaneous pancreatic biopsy"[tiab]
    )
    OR
    (
      "Positron Emission Tomography Computed Tomography"[Mesh]
      OR PET[tiab]
      OR "PET/CT"[tiab]
      OR "FDG-PET"[tiab]
      OR "Cholangiopancreatography, Endoscopic Retrograde"[Mesh]
      OR ERCP[tiab]
      OR "endoscopic retrograde cholangiopancreatography"[tiab]
      OR "double duct sign"[tiab]
      OR "double-duct sign"[tiab]
      OR "Laparoscopy"[Mesh]
      OR "staging laparoscopy"[tiab]
      OR "diagnostic laparoscopy"[tiab]
      OR "peritoneal cytology"[tiab]
    )
    OR
    (
      (
        "performance status"[tiab]
        OR ECOG[tiab]
        OR frailty[tiab]
        OR "nutritional status"[tiab]
        OR comorbid*[tiab]
      )
      AND
      (
        assessment[tiab]
        OR eligibility[tiab]
        OR risk[tiab]
        OR staging[tiab]
        OR "treatment selection"[tiab]
        OR surgery[tiab]
      )
    )
    '''
)

CHAPTER_4_1_TOPIC = _clean(
    r'''
    (
      "resectable pancreatic cancer"[tiab]
      OR "resectable pancreatic adenocarcinoma"[tiab]
      OR "localized pancreatic cancer"[tiab]
      OR "localised pancreatic cancer"[tiab]
      OR "curative-intent"[tiab]
      OR "curative intent"[tiab]
      OR "multidisciplinary team"[tiab]
      OR "multidisciplinary care"[tiab]
      OR "multidisciplinary treatment"[tiab]
      OR "Pancreatectomy"[Mesh]
      OR pancreatectom*[tiab]
      OR "pancreatic resection"[tiab]
      OR "pancreatic surgery"[tiab]
      OR "upfront surgery"[tiab]
      OR "primary surgery"[tiab]
      OR pancreaticoduodenectom*[tiab]
      OR pancreatoduodenectom*[tiab]
      OR Whipple[tiab]
      OR "pylorus-preserving pancreaticoduodenectomy"[tiab]
      OR "distal pancreatectomy"[tiab]
      OR "total pancreatectomy"[tiab]
      OR "radical antegrade modular pancreatosplenectomy"[tiab]
      OR RAMPS[tiab]
      OR "open pancreatectomy"[tiab]
      OR "open pancreatic surgery"[tiab]
      OR "laparoscopic pancreatectomy"[tiab]
      OR "laparoscopic pancreaticoduodenectomy"[tiab]
      OR "laparoscopic distal pancreatectomy"[tiab]
      OR "robotic pancreatectomy"[tiab]
      OR "robotic pancreaticoduodenectomy"[tiab]
      OR "robotic distal pancreatectomy"[tiab]
      OR "minimally invasive pancreatectomy"[tiab]
      OR "minimally invasive pancreatic surgery"[tiab]
      OR "artery-first approach"[tiab]
      OR "artery first approach"[tiab]
      OR "SMA-first approach"[tiab]
      OR "superior mesenteric artery dissection"[tiab]
      OR mesopancreas[tiab]
      OR "medial clearance"[tiab]
      OR "vascular resection"[tiab]
      OR "venous resection"[tiab]
      OR "portal vein resection"[tiab]
      OR "superior mesenteric vein resection"[tiab]
      OR "porto-mesenteric vein resection"[tiab]
      OR "portomesenteric vein resection"[tiab]
      OR "venous reconstruction"[tiab]
      OR "arterial resection"[tiab]
      OR "superior mesenteric artery resection"[tiab]
      OR "hepatic artery resection"[tiab]
      OR "celiac axis resection"[tiab]
      OR "coeliac axis resection"[tiab]
      OR "R0 resection"[tiab]
      OR "R1 resection"[tiab]
      OR "resection margin"[tiab]
      OR "resection margins"[tiab]
      OR "margin status"[tiab]
      OR "margin clearance"[tiab]
      OR "one millimeter rule"[tiab]
      OR "1-mm rule"[tiab]
      OR "1 mm margin"[tiab]
      OR "circumferential resection margin"[tiab]
      OR "superior mesenteric artery margin"[tiab]
      OR "uncinate margin"[tiab]
      OR "medial margin"[tiab]
      OR "posterior margin"[tiab]
      OR "anterior margin"[tiab]
      OR "pancreatic transection margin"[tiab]
      OR "bile duct margin"[tiab]
      OR "enteric margin"[tiab]
      OR "frozen section"[tiab]
      OR "specimen examination"[tiab]
      OR "pathology protocol"[tiab]
      OR RCPath[tiab]
      OR "Royal College of Pathologists"[tiab]
      OR ISGPS[tiab]
      OR lymphadenectom*[tiab]
      OR "standard lymphadenectomy"[tiab]
      OR "extended lymphadenectomy"[tiab]
      OR "lymph node yield"[tiab]
      OR "lymph node harvest"[tiab]
      OR "number of lymph nodes"[tiab]
      OR "lymph node ratio"[tiab]
      OR "node ratio"[tiab]
      OR "nodal ratio"[tiab]
      OR
      (
        (
          elderly[tiab]
          OR octogenarian*[tiab]
          OR age[tiab]
          OR frailty[tiab]
          OR "operative risk"[tiab]
          OR "risk score"[tiab]
          OR SOAR[tiab]
        )
        AND
        (pancreatectom*[tiab] OR "pancreatic resection"[tiab] OR "pancreatic surgery"[tiab])
      )
      OR "perioperative mortality"[tiab]
      OR "postoperative mortality"[tiab]
      OR "preoperative biliary drainage"[tiab]
      OR "pre-operative biliary drainage"[tiab]
      OR "preoperative biliary decompression"[tiab]
      OR "biliary drainage before surgery"[tiab]
      OR "preoperative biliary stent"[tiab]
      OR "preoperative biliary stenting"[tiab]
      OR
      (
        (
          "plastic stent"[tiab]
          OR "metal stent"[tiab]
          OR "biliary stent"[tiab]
          OR "biliary stenting"[tiab]
        )
        AND
        (preoperative[tiab] OR "pre-operative"[tiab] OR "before surgery"[tiab] OR resectable[tiab])
      )
      OR
      (
        (cholangitis[tiab] OR jaundice[tiab] OR bilirubin[tiab])
        AND
        ("pancreatic surgery"[tiab] OR pancreatectom*[tiab] OR resection[tiab])
      )
      OR "time to surgery"[tiab]
      OR "timing of surgery"[tiab]
    )
    OR
    (
      (
        adjuvant[tiab]
        OR postoperative[tiab]
        OR "post-operative"[tiab]
        OR "after resection"[tiab]
        OR "following resection"[tiab]
        OR "resected pancreatic cancer"[tiab]
        OR "resected pancreatic adenocarcinoma"[tiab]
      )
      AND
      (
        "Adjuvant Chemotherapy"[Mesh]
        OR chemotherapy[tiab]
        OR chemoradiation[tiab]
        OR chemoradiotherapy[tiab]
        OR radiotherapy[tiab]
        OR radiation[tiab]
        OR gemcitabine[tiab]
        OR fluorouracil[tiab]
        OR "5-FU"[tiab]
        OR "folinic acid"[tiab]
        OR leucovorin[tiab]
        OR capecitabine[tiab]
        OR "nab-paclitaxel"[tiab]
        OR "albumin-bound paclitaxel"[tiab]
        OR "gemcitabine-capecitabine"[tiab]
        OR GemCap[tiab]
        OR FOLFIRINOX[tiab]
        OR "modified FOLFIRINOX"[tiab]
        OR mFOLFIRINOX[tiab]
        OR "S-1"[tiab]
        OR tegafur[tiab]
        OR "margin-positive"[tiab]
        OR "node-positive"[tiab]
        OR "treatment completion"[tiab]
        OR "chemotherapy completion"[tiab]
        OR "time to adjuvant chemotherapy"[tiab]
        OR "timing of adjuvant chemotherapy"[tiab]
        OR "adjuvant treatment duration"[tiab]
        OR "duration of adjuvant chemotherapy"[tiab]
      )
    )
    OR
    (
      (
        neoadjuvant[tiab]
        OR "neo-adjuvant"[tiab]
        OR perioperative[tiab]
        OR "peri-operative"[tiab]
      )
      AND
      (
        "resectable pancreatic cancer"[tiab]
        OR "resectable pancreatic adenocarcinoma"[tiab]
        OR "upfront resectable"[tiab]
        OR "initially resectable"[tiab]
      )
    )
    '''
)

CHAPTER_4_2_TOPIC = _clean(
    r'''
    (
      "borderline resectable"[tiab]
      OR "borderline-resectable"[tiab]
      OR "borderline resectability"[tiab]
      OR BRPC[tiab]
      OR "potentially resectable"[tiab]
      OR "locally advanced"[tiab]
      OR LAPC[tiab]
      OR "local advanced"[tiab]
      OR "locally unresectable"[tiab]
      OR "unresectable nonmetastatic"[tiab]
      OR "unresectable non-metastatic"[tiab]
      OR "nonmetastatic unresectable"[tiab]
      OR "non-metastatic unresectable"[tiab]
    )
    AND
    (
      neoadjuvant[tiab]
      OR "neo-adjuvant"[tiab]
      OR preoperative[tiab]
      OR "pre-operative"[tiab]
      OR perioperative[tiab]
      OR "peri-operative"[tiab]
      OR induction[tiab]
      OR consolidation[tiab]
      OR "total neoadjuvant therapy"[tiab]
      OR "neoadjuvant systemic therapy"[tiab]
      OR chemotherapy[tiab]
      OR chemoradiation[tiab]
      OR chemoradiotherapy[tiab]
      OR radiotherapy[tiab]
      OR radiation[tiab]
      OR "stereotactic body radiotherapy"[tiab]
      OR "stereotactic body radiation therapy"[tiab]
      OR SBRT[tiab]
      OR "intensity-modulated radiotherapy"[tiab]
      OR IMRT[tiab]
      OR proton[tiab]
      OR FOLFIRINOX[tiab]
      OR "modified FOLFIRINOX"[tiab]
      OR mFOLFIRINOX[tiab]
      OR gemcitabine[tiab]
      OR "nab-paclitaxel"[tiab]
      OR paclitaxel[tiab]
      OR capecitabine[tiab]
      OR fluorouracil[tiab]
      OR "5-FU"[tiab]
      OR leucovorin[tiab]
      OR "folinic acid"[tiab]
      OR oxaliplatin[tiab]
      OR irinotecan[tiab]
      OR cisplatin[tiab]
      OR "multimodal therapy"[tiab]
      OR "combined modality therapy"[tiab]
      OR "multimodality treatment"[tiab]
      OR downstag*[tiab]
      OR conversion[tiab]
      OR "conversion surgery"[tiab]
      OR "conversion resection"[tiab]
      OR "secondary resection"[tiab]
      OR exploration[tiab]
      OR "surgical exploration"[tiab]
      OR restaging[tiab]
      OR reassessment[tiab]
      OR "resectability reassessment"[tiab]
      OR resection[tiab]
      OR pancreatectom*[tiab]
      OR "R0 resection"[tiab]
      OR "margin-negative resection"[tiab]
      OR "resection rate"[tiab]
      OR "surgical conversion"[tiab]
      OR "objective response"[tiab]
      OR "response rate"[tiab]
      OR "disease control"[tiab]
      OR "local control"[tiab]
      OR "progression-free survival"[tiab]
      OR "overall survival"[tiab]
      OR "treatment sequence"[tiab]
      OR "treatment sequencing"[tiab]
      OR "treatment duration"[tiab]
      OR "irreversible electroporation"[tiab]
      OR IRE[tiab]
      OR ablation[tiab]
      OR "local ablative therapy"[tiab]
      OR "radiofrequency ablation"[tiab]
      OR "microwave ablation"[tiab]
      OR "high-intensity focused ultrasound"[tiab]
      OR HIFU[tiab]
      OR "local treatment"[tiab]
    )
    '''
)

CHAPTER_4_3_TOPIC = _clean(
    r'''
    (
      metastatic[tiab]
      OR metastases[tiab]
      OR metastasis[tiab]
      OR "stage IV"[tiab]
      OR "stage 4"[tiab]
      OR "distant disease"[tiab]
      OR "advanced pancreatic cancer"[tiab]
      OR "advanced pancreatic adenocarcinoma"[tiab]
      OR "unresectable advanced"[tiab]
    )
    AND
    (
      "Palliative Care"[Mesh]
      OR palliat*[tiab]
      OR "supportive care"[tiab]
      OR "best supportive care"[tiab]
      OR "symptom control"[tiab]
      OR "symptom management"[tiab]
      OR "biliary obstruction"[tiab]
      OR "malignant biliary obstruction"[tiab]
      OR "biliary drainage"[tiab]
      OR "biliary stent"[tiab]
      OR "biliary stenting"[tiab]
      OR "self-expandable metal stent"[tiab]
      OR "self-expanding metal stent"[tiab]
      OR SEMS[tiab]
      OR "plastic stent"[tiab]
      OR "endoscopic stent"[tiab]
      OR "percutaneous drainage"[tiab]
      OR hepaticojejunostom*[tiab]
      OR hepatojejunostom*[tiab]
      OR "surgical bypass"[tiab]
      OR "duodenal obstruction"[tiab]
      OR "gastric outlet obstruction"[tiab]
      OR "duodenal stent"[tiab]
      OR "endoscopic gastrojejunostomy"[tiab]
      OR "EUS-guided gastroenterostomy"[tiab]
      OR "EUS-guided gastrojejunostomy"[tiab]
      OR "enteral stent"[tiab]
      OR gastrojejunostom*[tiab]
      OR pain[tiab]
      OR analges*[tiab]
      OR "cancer pain"[tiab]
      OR "celiac plexus block"[tiab]
      OR "coeliac plexus block"[tiab]
      OR "celiac plexus neurolysis"[tiab]
      OR "coeliac plexus neurolysis"[tiab]
      OR "EUS-guided neurolysis"[tiab]
      OR "endoscopic ultrasound-guided neurolysis"[tiab]
      OR "palliative radiotherapy"[tiab]
      OR "pancreatic enzyme replacement"[tiab]
      OR "pancreatic enzyme supplementation"[tiab]
      OR "pancreatic exocrine insufficiency"[tiab]
      OR malnutrition[tiab]
      OR cachexia[tiab]
      OR nutrition*[tiab]
      OR "weight loss"[tiab]
      OR "quality of life"[tiab]
      OR "patient-reported outcome"[tiab]
      OR "patient-reported outcomes"[tiab]
      OR psychosocial[tiab]
      OR chemotherapy[tiab]
      OR "systemic therapy"[tiab]
      OR "first-line"[tiab]
      OR "first line"[tiab]
      OR FOLFIRINOX[tiab]
      OR "modified FOLFIRINOX"[tiab]
      OR mFOLFIRINOX[tiab]
      OR gemcitabine[tiab]
      OR "nab-paclitaxel"[tiab]
      OR "albumin-bound paclitaxel"[tiab]
      OR "gemcitabine plus nab-paclitaxel"[tiab]
      OR "gemcitabine and nab-paclitaxel"[tiab]
      OR GemNab[tiab]
      OR NALIRIFOX[tiab]
      OR "nal-IRI"[tiab]
      OR "liposomal irinotecan"[tiab]
      OR "nanoliposomal irinotecan"[tiab]
      OR MM-398[tiab]
      OR fluorouracil[tiab]
      OR "5-FU"[tiab]
      OR leucovorin[tiab]
      OR "folinic acid"[tiab]
      OR irinotecan[tiab]
      OR oxaliplatin[tiab]
      OR cisplatin[tiab]
      OR capecitabine[tiab]
      OR erlotinib[tiab]
      OR "combination chemotherapy"[tiab]
      OR monotherapy[tiab]
      OR "performance status"[tiab]
      OR "poor performance status"[tiab]
      OR ECOG[tiab]
      OR frailty[tiab]
      OR frail[tiab]
      OR comorbid*[tiab]
      OR "life expectancy"[tiab]
      OR bilirubin[tiab]
      OR "treatment selection"[tiab]
      OR "treatment choice"[tiab]
      OR "treatment response"[tiab]
      OR "treatment monitoring"[tiab]
      OR "response evaluation"[tiab]
      OR "response assessment"[tiab]
      OR RECIST[tiab]
      OR "imaging assessment"[tiab]
      OR "eight-week assessment"[tiab]
      OR "8-week assessment"[tiab]
      OR "second-line"[tiab]
      OR "second line"[tiab]
      OR "later-line"[tiab]
      OR "later line"[tiab]
      OR "subsequent therapy"[tiab]
      OR salvage[tiab]
      OR refractory[tiab]
      OR "gemcitabine-refractory"[tiab]
      OR "treatment sequencing"[tiab]
      OR "maintenance therapy"[tiab]
      OR "maintenance treatment"[tiab]
      OR de-escalation[tiab]
      OR "treatment duration"[tiab]
      OR "treatment discontinuation"[tiab]
      OR "therapy sequencing"[tiab]
      OR "OFF regimen"[tiab]
      OR "oxaliplatin folinic acid fluorouracil"[tiab]
      OR FOLFOX[tiab]
      OR FOLFIRI[tiab]
      OR "acinar cell carcinoma"[tiab]
      OR "adenosquamous carcinoma"[tiab]
      OR "colloid carcinoma"[tiab]
      OR "medullary carcinoma"[tiab]
      OR "undifferentiated carcinoma"[tiab]
      OR "osteoclast-like giant cell"[tiab]
    )
    '''
)

CHAPTER_5_TOPIC = _clean(
    r'''
    "Precision Medicine"[Mesh]
    OR "Biomarkers, Tumor"[Mesh]
    OR "Genetic Testing"[Mesh]
    OR "High-Throughput Nucleotide Sequencing"[Mesh]
    OR "Molecular Targeted Therapy"[Mesh]
    OR "personalized medicine"[tiab]
    OR "personalised medicine"[tiab]
    OR "precision medicine"[tiab]
    OR "precision oncology"[tiab]
    OR "molecularly guided"[tiab]
    OR "biomarker-guided"[tiab]
    OR biomarker*[tiab]
    OR "predictive marker"[tiab]
    OR "predictive markers"[tiab]
    OR "predictive biomarker"[tiab]
    OR "predictive biomarkers"[tiab]
    OR "prognostic marker"[tiab]
    OR "prognostic markers"[tiab]
    OR "prognostic biomarker"[tiab]
    OR "prognostic biomarkers"[tiab]
    OR "companion diagnostic"[tiab]
    OR "molecular profiling"[tiab]
    OR "genomic profiling"[tiab]
    OR "comprehensive genomic profiling"[tiab]
    OR sequencing[tiab]
    OR "next-generation sequencing"[tiab]
    OR NGS[tiab]
    OR "whole-exome sequencing"[tiab]
    OR "whole-genome sequencing"[tiab]
    OR "germline testing"[tiab]
    OR "germline mutation"[tiab]
    OR "germline mutations"[tiab]
    OR "somatic testing"[tiab]
    OR "somatic mutation"[tiab]
    OR "somatic mutations"[tiab]
    OR "actionable alteration"[tiab]
    OR "actionable alterations"[tiab]
    OR "actionable mutation"[tiab]
    OR "actionable mutations"[tiab]
    OR targetable[tiab]
    OR "targeted therapy"[tiab]
    OR "targeted therapies"[tiab]
    OR "basket trial"[tiab]
    OR "basket trials"[tiab]
    OR "molecular tumor board"[tiab]
    OR "molecular tumour board"[tiab]
    OR "tumor-agnostic"[tiab]
    OR "tumour-agnostic"[tiab]
    OR "universal germline testing"[tiab]
    OR "cascade testing"[tiab]
    OR "tumor heterogeneity"[tiab]
    OR "tumour heterogeneity"[tiab]
    OR "intratumoral heterogeneity"[tiab]
    OR "intra-tumoral heterogeneity"[tiab]
    OR "intertumoral heterogeneity"[tiab]
    OR "inter-tumoral heterogeneity"[tiab]
    OR "liquid biopsy"[tiab]
    OR ctDNA[tiab]
    OR "circulating tumor DNA"[tiab]
    OR "circulating tumour DNA"[tiab]
    OR KRAS[tiab]
    OR "KRAS G12C"[tiab]
    OR TP53[tiab]
    OR CDKN2A[tiab]
    OR SMAD4[tiab]
    OR BRCA1[tiab]
    OR BRCA2[tiab]
    OR PALB2[tiab]
    OR ATM[tiab]
    OR "homologous recombination deficiency"[tiab]
    OR HRD[tiab]
    OR "DNA damage repair"[tiab]
    OR "homologous recombination repair"[tiab]
    OR "genomic instability"[tiab]
    OR "DNA repair deficiency"[tiab]
    OR "mismatch repair"[tiab]
    OR dMMR[tiab]
    OR MLH1[tiab]
    OR hMLH1[tiab]
    OR MSH2[tiab]
    OR MSH6[tiab]
    OR PMS2[tiab]
    OR "microsatellite instability"[tiab]
    OR MSI[tiab]
    OR "tumor mutational burden"[tiab]
    OR "tumour mutational burden"[tiab]
    OR TMB[tiab]
    OR "MSI-high"[tiab]
    OR "MSI high"[tiab]
    OR "TMB-high"[tiab]
    OR "TMB high"[tiab]
    OR SPARC[tiab]
    OR hENT1[tiab]
    OR SLC29A1[tiab]
    OR EGFR[tiab]
    OR pERK[tiab]
    OR pAKT[tiab]
    OR STK11[tiab]
    OR LKB1[tiab]
    OR PTCH[tiab]
    OR PTCH1[tiab]
    OR hedgehog[tiab]
    OR SMO[tiab]
    OR ERBB2[tiab]
    OR HER2[tiab]
    OR "MET amplification"[tiab]
    OR "MET mutation"[tiab]
    OR "MET alteration"[tiab]
    OR "MET inhibitor"[tiab]
    OR FGFR1[tiab]
    OR FGFR2[tiab]
    OR CDK6[tiab]
    OR PIK3CA[tiab]
    OR PIK3R3[tiab]
    OR NTRK[tiab]
    OR "NTRK fusion"[tiab]
    OR "NTRK fusions"[tiab]
    OR NRG1[tiab]
    OR "NRG1 fusion"[tiab]
    OR "NRG1 fusions"[tiab]
    OR BRAF[tiab]
    OR "BRAF V600E"[tiab]
    OR "RAF fusion"[tiab]
    OR "RAF fusions"[tiab]
    OR RET[tiab]
    OR ALK[tiab]
    OR ROS1[tiab]
    OR
    (
      (
        BRCA1[tiab]
        OR BRCA2[tiab]
        OR PALB2[tiab]
        OR ATM[tiab]
        OR HRD[tiab]
        OR "homologous recombination"[tiab]
        OR "DNA repair deficiency"[tiab]
      )
      AND
      (platinum[tiab] OR cisplatin[tiab] OR oxaliplatin[tiab] OR "platinum sensitivity"[tiab])
    )
    OR "PARP inhibitor"[tiab]
    OR "PARP inhibitors"[tiab]
    OR olaparib[tiab]
    OR rucaparib[tiab]
    OR niraparib[tiab]
    OR talazoparib[tiab]
    OR "immune checkpoint inhibitor"[tiab]
    OR "immune checkpoint inhibitors"[tiab]
    OR immunotherapy[tiab]
    OR pembrolizumab[tiab]
    OR nivolumab[tiab]
    OR "PD-1"[tiab]
    OR "PD-L1"[tiab]
    OR "CTLA-4"[tiab]
    OR dostarlimab[tiab]
    OR larotrectinib[tiab]
    OR entrectinib[tiab]
    OR zenocutuzumab[tiab]
    OR selpercatinib[tiab]
    OR pralsetinib[tiab]
    OR trastuzumab[tiab]
    OR pertuzumab[tiab]
    OR sotorasib[tiab]
    OR adagrasib[tiab]
    OR dabrafenib[tiab]
    OR trametinib[tiab]
    OR "mTOR inhibitor"[tiab]
    OR "mTOR inhibitors"[tiab]
    OR everolimus[tiab]
    OR "smoothened inhibitor"[tiab]
    OR "smoothened inhibitors"[tiab]
    OR saridegib[tiab]
    '''
)

CHAPTER_6_TOPIC = _clean(
    r'''
    "Follow-Up Studies"[Mesh]
    OR "Survivorship"[Mesh]
    OR follow-up[tiab]
    OR followup[tiab]
    OR "follow-up visit"[tiab]
    OR "follow-up visits"[tiab]
    OR "post-treatment surveillance"[tiab]
    OR "posttreatment surveillance"[tiab]
    OR "postoperative surveillance"[tiab]
    OR "post-operative surveillance"[tiab]
    OR "surveillance after resection"[tiab]
    OR "surveillance following resection"[tiab]
    OR "routine surveillance"[tiab]
    OR "routine follow-up"[tiab]
    OR "intensive follow-up"[tiab]
    OR "intensive surveillance"[tiab]
    OR "follow-up interval"[tiab]
    OR "follow-up intervals"[tiab]
    OR "follow-up schedule"[tiab]
    OR "follow-up schedules"[tiab]
    OR "surveillance interval"[tiab]
    OR "surveillance intervals"[tiab]
    OR "surveillance schedule"[tiab]
    OR "surveillance schedules"[tiab]
    OR
    (
      (
        "after curative treatment"[tiab]
        OR "after curative-intent treatment"[tiab]
        OR "after curative intent treatment"[tiab]
        OR "after resection"[tiab]
        OR "following resection"[tiab]
        OR "resected pancreatic cancer"[tiab]
        OR "resected pancreatic adenocarcinoma"[tiab]
        OR postoperative[tiab]
        OR "post-operative"[tiab]
        OR survivorship[tiab]
      )
      AND
      (
        surveillance[tiab]
        OR monitoring[tiab]
        OR recurrence[tiab]
        OR recurrent[tiab]
        OR relapse[tiab]
        OR "recurrence detection"[tiab]
        OR "early recurrence"[tiab]
        OR "asymptomatic recurrence"[tiab]
        OR "recurrence pattern"[tiab]
        OR "recurrence patterns"[tiab]
        OR "time to recurrence"[tiab]
        OR "disease-free survival"[tiab]
        OR "recurrence-free survival"[tiab]
        OR "overall survival"[tiab]
        OR "survival benefit"[tiab]
        OR "computed tomography"[tiab]
        OR "CT scan"[tiab]
        OR "CT scans"[tiab]
        OR MRI[tiab]
        OR "magnetic resonance imaging"[tiab]
        OR "CA 19-9"[tiab]
        OR CA19-9[tiab]
        OR "carbohydrate antigen 19-9"[tiab]
        OR symptom*[tiab]
        OR "quality of life"[tiab]
        OR "patient-reported outcome"[tiab]
        OR "patient-reported outcomes"[tiab]
        OR nutrition*[tiab]
        OR malnutrition[tiab]
        OR "weight loss"[tiab]
        OR "pancreatic exocrine insufficiency"[tiab]
        OR "pancreatic enzyme replacement"[tiab]
        OR diabetes[tiab]
        OR psychosocial[tiab]
        OR "psychosocial support"[tiab]
        OR "supportive care"[tiab]
      )
    )
    '''
)


CHAPTERS: dict[str, ChapterQuery] = {
    "1": ChapterQuery(
        "1",
        "Incidence and epidemiology",
        _compose(PDAC_CORE, CHAPTER_1_TOPIC),
    ),
    "2": ChapterQuery(
        "2",
        "Diagnosis, pathology and molecular biology",
        _compose(PDAC_WITH_PRECURSORS, CHAPTER_2_TOPIC),
    ),
    "3": ChapterQuery(
        "3",
        "Staging and risk assessment",
        _compose(PDAC_CORE, CHAPTER_3_TOPIC),
    ),
    "4.1": ChapterQuery(
        "4.1",
        "Treatment of localised disease",
        _compose(PDAC_CORE, CHAPTER_4_1_TOPIC),
    ),
    "4.2": ChapterQuery(
        "4.2",
        "Treatment of non-resectable disease: borderline resectable and locally advanced",
        _compose(PDAC_CORE, CHAPTER_4_2_TOPIC),
    ),
    "4.3": ChapterQuery(
        "4.3",
        "Treatment of advanced/metastatic disease",
        _compose(PDAC_WITH_RARE_EXOCRINE, CHAPTER_4_3_TOPIC),
    ),
    "5": ChapterQuery(
        "5",
        "Personalised medicine",
        _compose(PDAC_WITH_RARE_EXOCRINE, CHAPTER_5_TOPIC),
    ),
    "6": ChapterQuery(
        "6",
        "Follow-up and long-term implications",
        _compose(PDAC_CORE, CHAPTER_6_TOPIC),
    ),
}
