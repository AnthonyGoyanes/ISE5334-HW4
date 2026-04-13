"""
pipeline/catalog.py
Defines all 60 pipeline configurations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PipelineConfig:
    id: str
    name: str
    group: str
    pp: str
    fe: str
    ib: str
    fs: str
    cl: str
    requires_plane_model: bool = field(init=False)
    requires_ae: bool = field(init=False)
    apply_pca_post: bool = field(init=False)

    def __post_init__(self):
        from pipeline.features import PLANE_REQUIRED_FE, AE_REQUIRED_FE, PCA_FE
        self.requires_plane_model = self.fe in PLANE_REQUIRED_FE
        self.requires_ae = self.fe in AE_REQUIRED_FE
        self.apply_pca_post = self.fe in PCA_FE


def _p(id_, name, group, pp, fe, ib, fs, cl) -> PipelineConfig:
    return PipelineConfig(id=id_, name=name, group=group,
                          pp=pp, fe=fe, ib=ib, fs=fs, cl=cl)


PIPELINE_CATALOG: List[PipelineConfig] = [

    # ── BASELINE ─────────────────────────────────────────────────────────────
    _p("P-001", "Baseline RF",       "Baseline", "PP-B", "FE-A", "IB-0", "FS-0", "CL-1"),
    _p("P-002", "Baseline SVM",      "Baseline", "PP-B", "FE-A", "IB-0", "FS-0", "CL-2"),
    _p("P-003", "Baseline XGBoost",  "Baseline", "PP-B", "FE-A", "IB-0", "FS-0", "CL-3"),
    _p("P-004", "Baseline LogReg",   "Baseline", "PP-B", "FE-A", "IB-0", "FS-0", "CL-6"),

    # ── PLANE REMOVAL (no canonical alignment) ───────────────────────────────
    _p("P-005", "Plane-removed RF",                 "Plane Removal", "PP-C", "FE-A", "IB-0", "FS-0", "CL-1"),
    _p("P-006", "Plane-removed XGBoost",            "Plane Removal", "PP-C", "FE-A", "IB-0", "FS-0", "CL-3"),
    _p("P-007", "Plane-removed + Down RF",          "Plane Removal", "PP-E", "FE-A", "IB-0", "FS-0", "CL-1"),
    _p("P-008", "Plane-removed + Down Voxel RF",    "Plane Removal", "PP-E", "FE-B", "IB-0", "FS-0", "CL-1"),

    # ── PLANE-ALIGNED FRAME ───────────────────────────────────────────────────
    _p("P-009", "Plane-frame RF",            "Plane-Aligned Frame", "PP-C*", "FE-A", "IB-0", "FS-0", "CL-1"),
    _p("P-010", "Plane-frame XGBoost",       "Plane-Aligned Frame", "PP-C*", "FE-A", "IB-0", "FS-0", "CL-3"),
    _p("P-011", "Plane-frame + Down RF",     "Plane-Aligned Frame", "PP-E*", "FE-A", "IB-0", "FS-0", "CL-1"),
    _p("P-012", "Plane-frame + Down Voxel",  "Plane-Aligned Frame", "PP-E*", "FE-B", "IB-0", "FS-0", "CL-1"),

    # ── PLANE-RELATIVE FEATURES ───────────────────────────────────────────────
    _p("P-013", "Plane-rel only RF",             "Plane-Relative Features", "PP-C*", "FE-K", "IB-0", "FS-0", "CL-1"),
    _p("P-014", "Plane-rel only XGBoost",        "Plane-Relative Features", "PP-C*", "FE-K", "IB-0", "FS-0", "CL-3"),
    _p("P-015", "Geo + Plane-rel RF",            "Plane-Relative Features", "PP-C*", "FE-L", "IB-0", "FS-0", "CL-1"),
    _p("P-016", "Geo + Plane-rel XGBoost",       "Plane-Relative Features", "PP-C*", "FE-L", "IB-0", "FS-0", "CL-3"),
    _p("P-017", "Geo + Plane-rel SVM",           "Plane-Relative Features", "PP-C*", "FE-L", "IB-0", "FS-0", "CL-2"),
    _p("P-018", "Geo + Plane-rel + Down RF",     "Plane-Relative Features", "PP-E*", "FE-L", "IB-0", "FS-0", "CL-1"),
    _p("P-019", "Full + Plane-rel RF",           "Plane-Relative Features", "PP-C*", "FE-M", "IB-0", "FS-0", "CL-1"),
    _p("P-020", "Full + Plane-rel XGBoost",      "Plane-Relative Features", "PP-C*", "FE-M", "IB-0", "FS-0", "CL-3"),
    _p("P-021", "Full + Plane-rel + Down XGB",   "Plane-Relative Features", "PP-E*", "FE-M", "IB-0", "FS-0", "CL-3"),

    # ── VOXEL & PROJECTIONS ───────────────────────────────────────────────────
    _p("P-022", "Plane-frame Voxel RF",        "Voxel & Projections", "PP-C*", "FE-B", "IB-0", "FS-0", "CL-1"),
    _p("P-023", "Plane-frame Proj RF",         "Voxel & Projections", "PP-C*", "FE-C", "IB-0", "FS-0", "CL-1"),
    _p("P-024", "Plane-frame Proj XGBoost",    "Voxel & Projections", "PP-C*", "FE-C", "IB-0", "FS-0", "CL-3"),
    _p("P-025", "Plane-frame Geo+Voxel RF",    "Voxel & Projections", "PP-C*", "FE-D", "IB-0", "FS-0", "CL-1"),
    _p("P-026", "Plane-frame Geo+Proj RF",     "Voxel & Projections", "PP-C*", "FE-E", "IB-0", "FS-0", "CL-1"),
    _p("P-027", "Plane-frame Full RF",         "Voxel & Projections", "PP-C*", "FE-G", "IB-0", "FS-0", "CL-1"),
    _p("P-028", "Plane-frame Full XGBoost",    "Voxel & Projections", "PP-C*", "FE-G", "IB-0", "FS-0", "CL-3"),
    _p("P-029", "Plane-frame Full GradBoost",  "Voxel & Projections", "PP-C*", "FE-G", "IB-0", "FS-0", "CL-4"),
    _p("P-030", "Plane-frame Full MLP",        "Voxel & Projections", "PP-C*", "FE-G", "IB-0", "FS-0", "CL-5"),

    # ── IMBALANCE CORRECTION ──────────────────────────────────────────────────
    _p("P-031", "Geo+PlaneRel + weight RF",        "Imbalance Correction", "PP-C*", "FE-L", "IB-1", "FS-0", "CL-1"),
    _p("P-032", "Geo+PlaneRel + SMOTE RF",         "Imbalance Correction", "PP-C*", "FE-L", "IB-2", "FS-0", "CL-1"),
    _p("P-033", "Geo+PlaneRel + SMOTE+wt RF",      "Imbalance Correction", "PP-C*", "FE-L", "IB-3", "FS-0", "CL-1"),
    _p("P-034", "Full+PlaneRel + weight XGB",      "Imbalance Correction", "PP-C*", "FE-M", "IB-1", "FS-0", "CL-3"),
    _p("P-035", "Full+PlaneRel + SMOTE XGB",       "Imbalance Correction", "PP-C*", "FE-M", "IB-2", "FS-0", "CL-3"),
    _p("P-036", "Full+PlaneRel + SMOTE+wt XGB",    "Imbalance Correction", "PP-C*", "FE-M", "IB-3", "FS-0", "CL-3"),
    _p("P-037", "Full+PlaneRel + SMOTE MLP",       "Imbalance Correction", "PP-C*", "FE-M", "IB-2", "FS-0", "CL-5"),
    _p("P-038", "Full+PlaneRel + weight SVM",      "Imbalance Correction", "PP-C*", "FE-M", "IB-1", "FS-0", "CL-2"),

    # ── FEATURE SELECTION ─────────────────────────────────────────────────────
    _p("P-039", "Full+PlaneRel + SelectK RF",      "Feature Selection", "PP-C*", "FE-M", "IB-1", "FS-1", "CL-1"),
    _p("P-040", "Full+PlaneRel + RFE RF",          "Feature Selection", "PP-C*", "FE-M", "IB-1", "FS-2", "CL-1"),
    _p("P-041", "Full+PlaneRel + PCA SVM",         "Feature Selection", "PP-C*", "FE-M", "IB-1", "FS-3", "CL-2"),
    _p("P-042", "Geo+PlaneRel + SelectK XGB",      "Feature Selection", "PP-C*", "FE-L", "IB-1", "FS-1", "CL-3"),
    _p("P-043", "Full+PlaneRel+Down + RFE GBM",    "Feature Selection", "PP-E*", "FE-M", "IB-1", "FS-2", "CL-4"),

    # ── PCA LATENT ────────────────────────────────────────────────────────────
    _p("P-044", "PCA latent RF",           "PCA Latent", "PP-C*", "FE-H", "IB-0", "FS-0", "CL-1"),
    _p("P-045", "PCA latent + weight SVM", "PCA Latent", "PP-C*", "FE-H", "IB-1", "FS-0", "CL-2"),
    _p("P-046", "PCA latent + SMOTE XGB",  "PCA Latent", "PP-C*", "FE-H", "IB-2", "FS-0", "CL-3"),
    _p("P-047", "PCA latent + weight MLP", "PCA Latent", "PP-C*", "FE-H", "IB-1", "FS-0", "CL-5"),

    # ── AUTOENCODER LATENT ────────────────────────────────────────────────────
    _p("P-048", "AE latent RF",                 "Autoencoder Latent", "PP-C*", "FE-I", "IB-0", "FS-0", "CL-1"),
    _p("P-049", "AE latent + weight SVM",       "Autoencoder Latent", "PP-C*", "FE-I", "IB-1", "FS-0", "CL-2"),
    _p("P-050", "AE latent + SMOTE XGB",        "Autoencoder Latent", "PP-C*", "FE-I", "IB-2", "FS-0", "CL-3"),
    _p("P-051", "AE latent + weight MLP",       "Autoencoder Latent", "PP-C*", "FE-I", "IB-1", "FS-0", "CL-5"),
    _p("P-052", "AE+Geo + weight RF",           "Autoencoder Latent", "PP-C*", "FE-J", "IB-1", "FS-0", "CL-1"),
    _p("P-053", "AE+Geo + SMOTE+wt XGB",        "Autoencoder Latent", "PP-C*", "FE-J", "IB-3", "FS-0", "CL-3"),
    _p("P-054", "AE+Geo+PlaneRel + weight RF",  "Autoencoder Latent", "PP-C*", "FE-N", "IB-1", "FS-0", "CL-1"),
    _p("P-055", "AE+Geo+PlaneRel + SMOTE+wt XGB","Autoencoder Latent","PP-C*", "FE-N", "IB-3", "FS-0", "CL-3"),
    _p("P-056", "AE+Geo+PlaneRel+Down XGB",     "Autoencoder Latent", "PP-E*", "FE-N", "IB-3", "FS-0", "CL-3"),

    # ── LIGHTWEIGHT REFERENCE ─────────────────────────────────────────────────
    _p("P-057", "Geo+PlaneRel + weight KNN",    "Lightweight Reference", "PP-C*", "FE-L", "IB-1", "FS-0", "CL-7"),
    _p("P-058", "PCA latent + weight LogReg",   "Lightweight Reference", "PP-C*", "FE-H", "IB-1", "FS-0", "CL-6"),
    _p("P-059", "Geo+PlaneRel + SelectK LogReg","Lightweight Reference", "PP-C*", "FE-L", "IB-1", "FS-1", "CL-6"),

    # ── KITCHEN SINK ──────────────────────────────────────────────────────────
    _p("P-060", "Kitchen Sink XGB",
       "Kitchen Sink", "PP-E*", "FE-N", "IB-3", "FS-2", "CL-3"),
]

# Lookup dict
PIPELINE_CATALOG_DICT: Dict[str, PipelineConfig] = {p.id: p for p in PIPELINE_CATALOG}

# Group lookup
PIPELINE_GROUPS: Dict[str, List[str]] = {}
for _p_cfg in PIPELINE_CATALOG:
    PIPELINE_GROUPS.setdefault(_p_cfg.group, []).append(_p_cfg.id)

ALL_GROUPS = list(PIPELINE_GROUPS.keys())
