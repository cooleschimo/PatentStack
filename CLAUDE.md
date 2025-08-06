# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A comprehensive full-stack patent classification system that can be configured for any technology domain. Currently focused on quantum computing, this system demonstrates end-to-end data engineering, machine learning, and full-stack development capabilities for technology investment intelligence.

**Value Proposition**: "Universal patent intelligence system that transforms unstructured patent data into actionable business insights using multi-label classification and systematic hierarchical taxonomy - a flexible framework that can be adapted to any technology domain for data-driven investment decisions."

## Architecture

### Full-Stack Components

**Backend**: Python, FastAPI
**Database**: PostgreSQL (production), DuckDB (analytics)  
**ML/Data**: PyTorch, HuggingFace Transformers, PatentSBERTa (multi-label classification)
**Data Sources**: USPTO PatentsView API (US patents) + Google Patents BigQuery API (International patents)
**Frontend**: Streamlit Dashboard (interactive data analysis)
**Deployment**: Docker, GitHub Actions

### Project Structure
```
patent-classifier/
├── src/
│   ├── data_fetch/
│   │   └── patentdata_fetcher.py  # Hybrid USPTO + Google Patents data collection
│   ├── database/
│   │   ├── models.py
│   │   ├── database.py
│   │   └── migrations/
│   ├── ml/
│   │   ├── feature_extraction.py
│   │   ├── multilabel_classifier.py
│   │   ├── patentsberta_classifier.py
│   │   └── evaluation.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   └── schemas/
│   ├── analytics/
│   │   ├── technology_analytics.py
│   │   └── interactive_analysis.py
│   ├── frontend/
│   │   ├── technology_dashboard.py
│   │   ├── manual_review.py
│   │   └── data_input_forms.py
│   └── config/
│       └── taxonomy_definitions.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_evaluation_analysis.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── taxonomies/
│   ├── quantum_computing.yaml
│   ├── biotechnology.yaml
│   └── template.yaml
└── docs/
```

## Hybrid Data Collection Architecture

### Dual Patent Data Sources
**US Patents**: USPTO PatentsView API - Direct access to US patent database with rich metadata
**International Patents**: Google Patents BigQuery API - Global patent coverage with standardized data

### Data Collection Strategy
User inputs CPC codes for technology area → Hybrid fetcher collects from both APIs → Merge and deduplicate → Store in database

## Multi-Label Hierarchical Classification System

### Configurable Technology Taxonomy Framework
The system supports any technology domain through configurable hierarchical taxonomies:

**Layer 1**: Technology Domain (e.g., quantum_computing, biotechnology)
**Layer 2-4**: Technology Stack (hierarchical subcategories)
**Layer 5**: Keywords & Patterns (for validation)

### Example: Quantum Computing Taxonomy
```
quantum_computing:
├── software:
│   ├── applications (drug_discovery, finance, optimization, sensing)
│   └── algorithms (grover, shor, vqe, qaoa, quantum_circuit)
├── middleware:
│   ├── hybrid_system_software (cloud_platforms, orchestration)
│   ├── sdk_language (qiskit, cirq, pennylane)
│   ├── compiler (circuit_optimization, gate_synthesis)
│   └── error_correction (surface_codes, fault_tolerance)
└── hardware:
    ├── qpu (superconducting, trapped_ion, photonic)
    ├── control_systems (pulse_control, feedback_loops)
    └── ancillary_components (cryogenics, lasers, vacuum_systems)
```

## Machine Learning Architecture

### Multi-Label Classification with PatentSBERTa
- **Base Model**: PatentSBERTa (768-dimensional embeddings, frozen)
- **Classification Head**: Multi-layer neural network with sigmoid activations
- **Output**: Multi-label predictions for all hierarchical levels simultaneously
- **Key Feature**: Patents can belong to multiple categories (like drug discovery + algorithms)

## System Workflow

1. **Data Collection**: User inputs CPC codes → Hybrid fetcher from USPTO + Google APIs
2. **Multi-Label Classification**: PatentSBERTa embeddings → Neural network → Multiple simultaneous labels
3. **Manual Review**: Flag low-confidence predictions → Expert validation → Active learning
4. **Interactive Analytics**: User-driven dashboard → Custom analysis → Export capabilities

## Technology Stack

### Core Dependencies
- **Data Processing**: pandas, numpy, duckdb, sqlalchemy
- **Machine Learning**: PyTorch, transformers, scikit-learn, PatentSBERTa, scikit-multilearn
- **Patent Data APIs**: requests (USPTO PatentsView), google-cloud-bigquery
- **Visualization**: plotly, streamlit, matplotlib, seaborn
- **Database**: PostgreSQL (psycopg), DuckDB
- **API**: FastAPI, pydantic
- **Configuration**: PyYAML, python-dotenv

## Implementation Phases

### Phase 1: Enhanced Hybrid Data Collection
- Complete `patentdata_fetcher.py` with robust error handling and batch processing
- CPC code-based filtering and data validation

### Phase 2: Database Design
- PostgreSQL schema for patents and classifications
- DuckDB for analytics workloads and migration scripts

### Phase 3: Multi-Label ML Pipeline
- PatentSBERTa + custom multi-label neural network
- Hierarchical consistency enforcement and confidence scoring

### Phase 4: Manual Review System
- Interactive classification review interface
- Active learning feedback loop and quality metrics

### Phase 5: Interactive Analytics Dashboard
- User-driven data exploration and custom query capabilities
- Dynamic visualizations and export features

## Interactive Analytics Capabilities

### User-Driven Analysis Features
- Technology trend analysis and company intelligence
- Cross-technology discovery and geographic analysis
- Citation network analysis and custom natural language queries
- Dynamic filtering and interactive visualizations
- Export options and saved analysis configurations

### Key Analytics Questions
- "Show patent filing trends for quantum algorithms over the last 5 years"
- "Which companies are most active in quantum error correction?"
- "Find patents that span multiple technology areas"
- "Technology convergence between domains"

## Development Commands

### Data Collection & Processing
```bash
python src/data_fetch/patentdata_fetcher.py --cpc-codes "G06N10,H04L9" --start-date 2020-01-01
python src/database/data_processor.py --source hybrid --validate
python src/config/setup_taxonomy.py --domain biotechnology
```

### Model Training & Evaluation
```bash
python src/ml/multilabel_classifier.py --train --taxonomy quantum_computing.yaml
python src/ml/evaluation.py --model multilabel --test-set validation
```

### Dashboard & Analytics
```bash
streamlit run src/frontend/technology_dashboard.py
streamlit run src/frontend/manual_review.py
uvicorn src.api.main:app --reload
```

## Data Flow Architecture
```
User Input (CPC codes) → patentdata_fetcher.py → 
USPTO PatentsView API + Google Patents BigQuery → 
Database (PostgreSQL/DuckDB) → 
Multi-Label Classifier (PatentSBERTa) → 
Interactive Dashboard (Streamlit) → 
Analytics & Export
```

## Configuration

Technology taxonomies defined in YAML files, database connections via environment variables, and configurable classification thresholds and review parameters.

This architecture provides a comprehensive framework for building a flexible, full-stack patent intelligence system that can be adapted to any technology domain while providing powerful multi-label classification and interactive analytics capabilities.