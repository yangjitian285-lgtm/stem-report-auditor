"""「脱・量産型AI」独自性スコアラー (De-Homogenizer / Uniqueness Scorer)."""

from .math_engine import (
    UniquenessScorer,
    ScoringConfig,
    ScoringResult,
    ClaimDiagnostic,
    vendi_score,
    marginal_vendi_gain,
)
from .embeddings import (
    LocalHybridEmbedder,
    SentenceTransformerBackend,
    OpenAIBackend,
    GeminiBackend,
    create_backend,
)
from .domain_rules import DomainProfile, PROFILES, detect_profiles, is_stem_document
from .stem_engine import (
    StemUniquenessScorer,
    StemScoringResult,
    DataSeriesInput,
    NumericAuditResult,
    audit_numeric_data,
)
from .doc_parser import ParsedDocument, ParsedTable, parse_report_file
from .docx_patcher import PatchOperationResult, patch_docx_safe
from .stem_auditor import AuditFinding, FullStemAuditReport, audit_stem_document
from .table_checker import TableCheckResult, check_table, check_table_text

__all__ = [
    "UniquenessScorer",
    "ScoringConfig",
    "ScoringResult",
    "ClaimDiagnostic",
    "vendi_score",
    "marginal_vendi_gain",
    "LocalHybridEmbedder",
    "SentenceTransformerBackend",
    "DomainProfile",
    "PROFILES",
    "detect_profiles",
    "is_stem_document",
    "OpenAIBackend",
    "GeminiBackend",
    "create_backend",
    "StemUniquenessScorer",
    "StemScoringResult",
    "DataSeriesInput",
    "NumericAuditResult",
    "audit_numeric_data",
    "ParsedDocument",
    "ParsedTable",
    "parse_report_file",
    "PatchOperationResult",
    "patch_docx_safe",
    "AuditFinding",
    "FullStemAuditReport",
    "audit_stem_document",
    "TableCheckResult",
    "check_table",
    "check_table_text",
]
