#!/usr/bin/env python3
"""
# BigQuery Patent Data Extractor #
Reads configuration from YAML files and pulls patent data for specified companies
across any technology domains defined in the configuration.
"""
import os, sys, argparse, yaml, logging
from pathlib import Path
from datetime import datetime, timedelta
import re
import time
import pandas as pd

##### Set up #####
# Google Cloud BigQuery client setup
try: 
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound, BadRequest
except ImportError:
    print("Google Cloud libraries not found. Please install them with: pip install google-cloud-bigquery")
    sys.exit(1)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bigquery_pull.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

##### Patent Configuration Parser #####
class PatentConfigParser:
    """
    Loads and parses patent classification configuration from YAML files.
    """

    def __init__(self, config_path: str = "config/cpc_codes.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Load configuration from YAML file.
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

    def get_domains(self) -> list:
        """
        Get list of tech domains defined in the config file.
        """
        if 'domains' not in self.config:
            logger.error("No domains defined in configuration")
            return []
        
        domains = list(self.config.get('domains', {}).keys())
        logger.info(f"Found domains: {domains}")
        return domains

    



##### BigQuery Patent Fetcher #####
class QuantumPatentFetcher:
    
    def __init__(self, project_id: str = "patent-analyzer-463406"):
        self.quantum_cpc = self._build_quantum_cpc()

        self.client = bigquery.Client(project=project_id)
        self.quantum_terms = self._build_quantum_terms()
        self.qubit_tech_terms = self._build_qubit_tech_terms()
        self.quantum_taxonomy = self._build_quantum_taxonomy()
        
    def _build_quantum_terms(self) -> dict[str, list[str]]:
        """Build comprehensive quantum computing search terms by category"""

