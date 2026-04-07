# CHANGELOG

All notable changes to the ADDS research platform are documented here.

## [3.5.0] - 2026-04-05 - Repository Professionalization

### Fixed
- Removed all hardcoded Windows absolute paths from 15+ source files
- Replaced with BASE_DIR = Path(os.environ.get("ADDS_BASE_DIR", ...)) pattern
- Fixed syntax errors in 9 files caused by incorrect BASE_DIR block placement
- Fixed bare "from utils import" to try/except compatible pattern in 10 source files
- Fixed .gitignore UnicodeDecodeError (pure ASCII)

### CI/CD
- GitHub Actions: Syntax Check + Science Tests + Security Scan all green
- Removed "|| true" from Ruff check - now enforces real E9 errors
- Added test_unit_modules.py: 37 real unit tests importing actual src/ modules
- Added pytest-cov coverage reporting for src/utils, src/evaluation, src/data
- CI run 518+: All 3 jobs green

### Structure
- Removed figures/ (135 files) from git tracking
- Added src/adds/__init__.py and subpackage __init__.py files
- Updated .gitignore: excludes htmlcov/, benchmark_results/, upload_*.py
- Removed temporary scripts from root

### Testing
- tests/test_science_core.py: 18 tests (math/statistics)
- tests/test_unit_modules.py: 37 tests (real module import + functional)
- Total: 55 tests, 53 passing, 2 skipped

## [3.0.0] - 2026-03-15 - ADDS Platform v3 Launch

### Added
- Multimodal DL pipeline: 4-modal Fusion MLP for PFS/OS/Synergy prediction
- PrPc-RPSA Signalosome mechanistic knowledge base
- CT preprocessing with Swin-UNETR (Dice score 0.60+)
- Cellpose 2.0 cell segmentation pipeline (43,190 cells profiled)

## [2.0.0] - 2026-02-01 - Nature Communications Submission Prep

### Added
- KRAS G12C/G12D/G12V mutant-specific treatment recommendations
- Pritamab synergy analysis framework
- TCGA dataset integration (n=2285)

## [1.0.0] - 2025-12-01 - Initial Research Platform

### Added
- Initial ADDS platform architecture
- Basic drug synergy calculation (Bliss, Loewe, HSA)
