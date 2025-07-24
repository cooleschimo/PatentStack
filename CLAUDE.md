# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A comprehensive full-stack patent classification system focused on quantum computing and quantum safe communications technologies. This system demonstrates end-to-end data engineering, machine learning, and full-stack development capabilities for technology investment intelligence.

**Value Proposition**: "End-to-end patent intelligence system that transforms unstructured patent data into actionable business insights using modern NLP and systematic classification - exactly the kind of data-driven analysis quantitative firms need for technology investment decisions."

## Architecture

### Full-Stack Components

**Backend**: Python, FastAPI
**Database**: PostgreSQL (production), DuckDB (analytics)  
**ML/Data**: PyTorch, HuggingFace Transformers, PatentSBERTa
**Data Source**: Google Patents BigQuery API
**Frontend**: Streamlit Dashboard
**Deployment**: Docker, GitHub Actions

### Project Structure
```
patent-classifier/
├── src/
│   ├── data/
│   │   ├── bigquery_client.py
│   │   ├── patent_scraper.py
│   │   └── data_processor.py
│   ├── database/
│   │   ├── models.py
│   │   ├── database.py
│   │   └── migrations/
│   ├── ml/
│   │   ├── feature_extraction.py
│   │   ├── classifier.py
│   │   ├── dual_domain_classifier.py
│   │   └── evaluation.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   └── schemas/
│   ├── analytics/
│   │   └── quantum_analytics.py
│   └── frontend/
│       └── quantum_dashboard.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_evaluation_analysis.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
└── docs/
```

## 5-Layer Technology Taxonomy

### Layer 1: Technology Domains
- **quantum_computing**: Main computational domain
- **quantum_safe_communications**: Security-focused domain

### Layer 2-4: Quantum Computing Stack
```
quantum_computing:
├── software:
│   ├── applications (sensing, finance, optimization, drug_discovery, bioinformatics, material_design)
│   └── algorithms (grover, shor, vqe, qaoa, quantum_circuit, variational_algorithm)
├── middleware:
│   ├── hybrid_system_software (cloud platforms, resource management, job scheduling, orchestration)
│   ├── sdk_language (qiskit, q#, qibo, cirq, pennylane)
│   ├── compiler (circuit optimization, gate synthesis, transpilation)
│   └── error_correction (surface codes, stabilizer codes, fault tolerance)
└── hardware:
    ├── qpu (superconducting, trapped_ion, neutral_atom, photonic, spin_qubit, topological)
    ├── control_systems (pulse control, decoherence suppression, feedback loops)
    └── ancillary_components (cryogenics, lasers, vacuum systems, cabling, power_systems)
```

### Layer 2-4: Quantum Safe Communications Stack
```
quantum_safe_communications:
├── cryptographic_protocols:
│   ├── qkd_protocols (BB84, SARG04, E91, continuous_variable_qkd)
│   ├── post_quantum_crypto (lattice_based, hash_based, code_based, multivariate)
│   └── key_management (quantum_key_distribution, key_rotation, key_storage)
├── communication_infrastructure:
│   ├── fiber_networks (quantum_channels, dark_fiber, metropolitan_networks)
│   ├── satellite_qkd (space_based_quantum, ground_stations, free_space_optics)
│   └── quantum_repeaters (quantum_memory, entanglement_swapping, network_nodes)
└── security_applications:
    ├── financial_security (banking_systems, trading_platforms, payment_security)
    ├── government_defense (military_communications, intelligence_networks, national_security)
    └── enterprise_security (data_centers, cloud_security, corporate_networks)
```

### Layer 5: Technology Keywords
Multiple phrasings and variations for each technology enable robust pattern matching and classification.

## Machine Learning Architecture

### Primary Model: PatentSBERTa + Custom Neural Network
```
PatentSBERTa (frozen) → Custom NN Head → Technology Classifications
Architecture: 768 → 512 → 256 → 128 → [quantum_computing + quantum_safe_comms categories]
```

### Training Strategy
- **Transfer Learning**: Leverage PatentSBERTa's patent-specific knowledge
- **Custom Classification Head**: Multi-layer neural network with dropout regularization
- **Hierarchical Classification**: First predict domain, then technology stack category
- **Active Learning**: Iterative improvement with manual review feedback

### Technology Stack

- **Data Processing**: pandas, numpy, duckdb
- **Machine Learning**: PyTorch, transformers, scikit-learn, PatentSBERTa
- **Visualization**: plotly, streamlit
- **Cloud Services**: Google BigQuery (google-cloud-bigquery)
- **Database**: PostgreSQL (psycopg), SQLAlchemy, DuckDB
- **API**: FastAPI
- **Configuration**: PyYAML, python-dotenv
- **CLI**: click

## Implementation Phases

### Phase 1: Enhanced Data Collection & Storage
**Current Focus**: Enhance `src/biqquery_pull.py`
- Domain-aware patent collection
- Dual taxonomy keyword matching  
- Improved error handling and batch processing
- Data validation and logging system

### Phase 2: Database Design
**Database Layer**: `src/database/database.py`
- PostgreSQL schema for patents and classifications
- DuckDB for analytics workloads
- Migration scripts and data models
- Indexing for performance optimization

### Phase 3: ML Pipeline - Dual Domain Classification
**Core ML**: `src/ml/dual_domain_classifier.py`
- PatentSBERTa + custom neural network heads
- Hierarchical classification architecture
- Confidence scoring and review flagging
- Model versioning and A/B testing

### Phase 4: Manual Review Interface
**Review System**: Interactive classification review
- Flag low-confidence predictions
- Expert review workflow
- Active learning feedback loop
- Classification quality metrics

### Phase 5: Analytics Dashboard
**Frontend**: `src/frontend/quantum_dashboard.py`
- Cross-domain trend analysis
- Company intelligence across both domains
- Technology convergence identification
- Export capabilities for further analysis

## Analytics Capabilities

### Quantum Computing Analytics
- "Show IBM's quantum hardware QPU patent distribution by technology type"
- "Track quantum algorithm development trends over time"
- "Which companies lead in quantum error correction research?"
- "Quantum software vs hardware investment patterns"

### Quantum Safe Communications Analytics
- "Satellite QKD patent filings by country and company"
- "Post-quantum cryptography vs QKD protocol development"
- "Financial sector quantum security patent activity"
- "Government vs enterprise quantum security focus"

### Cross-Domain Analytics
- "Companies active in both quantum computing and quantum safe communications"
- "Technology convergence between domains"
- "Investment flow patterns across quantum technology domains"
- "Geographic distribution of quantum technology development"

## Key Files to Implement

### 1. Enhanced bigquery_pull.py
- Domain-aware patent collection
- Dual taxonomy keyword matching
- Improved error handling and batch processing

### 2. Dual Domain Classifier (ml/dual_domain_classifier.py)
- PatentSBERTa + custom neural network heads
- Hierarchical classification architecture  
- Confidence scoring and review flagging

### 3. Analytics Engine (analytics/quantum_analytics.py)
- Cross-domain trend analysis
- Company intelligence across both domains
- Market opportunity identification

### 4. Streamlit Dashboard (frontend/quantum_dashboard.py)
- Interactive visualizations for both domains
- Company comparison tools
- Technology trend monitoring
- Export capabilities for further analysis

## Development Setup

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **BigQuery Setup**: Ensure Google Cloud credentials are configured for BigQuery access

3. **Database Setup**: Configure PostgreSQL and DuckDB connections

4. **Environment Variables**: Configure database connections and API keys using `.env` file

## Common Commands

### Development Commands
- **Enhanced data fetcher**: `python src/biqquery_pull.py --domain quantum_computing`
- **Train dual classifier**: `python src/ml/dual_domain_classifier.py --train`
- **Run analytics**: `python src/analytics/quantum_analytics.py`
- **Start dashboard**: `streamlit run src/frontend/quantum_dashboard.py`
- **API server**: `uvicorn src.api.main:app --reload`

### Data Pipeline Commands
- **Full pipeline**: `python src/data/pipeline.py --full-refresh`
- **Incremental update**: `python src/data/pipeline.py --incremental`
- **Model evaluation**: `python src/ml/evaluation.py --model dual_domain`

### Testing and Quality
- **Run tests**: `python -m pytest tests/`
- **Data validation**: `python src/data/validation.py`
- **Model performance**: `python src/ml/model_monitoring.py`

## Demonstrating Quant-Relevant Skills

### 1. Systematic Classification Framework
- Multi-domain taxonomy showing ability to handle complex categorization
- Hierarchical structure demonstrating systematic thinking
- Industry-aligned categories showing domain expertise

### 2. Advanced Analytics Capabilities
- Cross-domain analysis identifying technology convergence
- Company intelligence for competitive positioning
- Trend detection for investment opportunities
- Risk assessment through confidence scoring

### 3. Production-Ready Engineering
- Scalable architecture supporting multiple technology domains
- Robust data pipeline with monitoring and error handling
- Modular design enabling easy extension to new domains
- Performance optimization for real-time analytics

### 4. Data Pipeline Engineering
- Robust ETL from BigQuery with incremental updates
- Data quality monitoring and validation
- Error handling, logging, and retry logic
- Feature engineering for time-series and NLP

### 5. Model Development & Monitoring
- Transfer learning with PatentSBERTa
- Hyperparameter optimization and model versioning
- Classification accuracy tracking and drift detection
- Active learning for continuous improvement

## Configuration

- Technology taxonomy defined in classification logic
- Database connections configured via environment variables
- Patent classification hierarchy implemented in ML models
- API configuration and rate limiting settings