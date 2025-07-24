"""
bigquery_pull.py
Quantum patent fetcher
- Pulls quantum computing patents from Google Patents BigQuery with hierarchical classification
"""

import argparse
import pathlib
import logging
from datetime import datetime
import pandas as pd

from google.cloud import bigquery
import duckdb

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumPatentFetcher:
    
    def __init__(self, project_id: str = "patent-analyzer-463406"):
        self.quantum_cpc = self._build_quantum_cpc()

        self.client = bigquery.Client(project=project_id)
        self.quantum_terms = self._build_quantum_terms()
        self.qubit_tech_terms = self._build_qubit_tech_terms()
        self.quantum_taxonomy = self._build_quantum_taxonomy()
        
    def _build_quantum_terms(self) -> dict[str, list[str]]:
        """Build comprehensive quantum computing search terms by category"""

