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

from typing import Any

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

##### CPC Code Configuration Parser #####
class CPCConfigParser:
    """
    Loads and parses CPC code classification configuration from YAML files.
    """
    config_path: Path
    config: dict[str, Any]
    domains: list[str]

    def __init__(self, config_path: str = "config/cpc_codes.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.domains = self.get_domains()

    def _load_config(self) -> dict[str, Any]:
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
        
        domains: list[str] = list(self.config.get('domains', {}).keys())
        logger.info(f"Found domains: {domains}")
        return domains

    def get_cpc_codes(self, domains: list[str] | None = None) -> dict[str, list[str]]:
        """
        Get CPC codes and keywords for a specific domain or all domains.
        If domain is None, returns all CPC codes across all domains.
        """
        # validity checks
        if domains is None:
            domains_to_process: list[str] = self.domains
        elif isinstance(domains, str):
            domains_to_process: list[str] = [domains]
        else:
            domains_to_process: list[str] = domains
        
        invalid_domains: list[str] = [d for d in domains_to_process if d not in self.domains]
        if invalid_domains:
            raise ValueError(f"Invalid domains: {invalid_domains}. Available: {self.domains}")

        # dict of domain to list of CPC codes under that domain
        cpc_codes: dict[str, list[str]] = {}  
        for dom in domains_to_process:
            # get value (dict of cpc codes) from config dict corresponding to 
            # the key of the dom, then get value (list of cpc codes) from that 
            # dict with key 'cpc_codes'
            cpc_list: list[dict[str, Any]] = self.config['domains'][dom].get('cpc_codes', [])
            codes: list[str] = [cpc['code'] for cpc in cpc_list]
            cpc_codes[dom] = codes

        return cpc_codes

            



 




##### BigQuery Patent Fetcher #####
class QuantumPatentFetcher:
    pass
  

