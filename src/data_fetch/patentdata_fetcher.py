#!/usr/bin/env python3
"""
# Patent Data Fetcher #
Reads configuration from YAML files and pulls patent data for specified companies
across any technology domains defined in the configuration.
- Uses USPTO API for US patents (FREE, no limits)
- Uses BigQuery for international patents (limits apply)
"""
import os, sys, argparse, yaml, logging, requests, time, json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Any

### Set up #####################################################################
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

### CPC Code Configuration Parser ##############################################
class CPCParser:
    """
    Loads and parses CPC code classification configuration from YAML files.
    """
    config_path: Path
    config: dict[str, Any]
    domains: list[str]
    cpc_codes_dict: dict[str, list[str]]
    global_settings: dict[str, Any]

    def __init__(self, config_path: str = "config/cpc_codes.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.domains = self.get_domains()
        self.cpc_codes_dict = self.get_cpc_codes()
        self.global_settings = self.get_global_settings()

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
        cpc_codes_dict: dict[str, list[str]] = {}  
        for dom in domains_to_process:
            # get value (dict of cpc codes) from config dict corresponding to 
            # the key of the dom, then get value (list of cpc codes) from that 
            # dict with key 'cpc_codes'
            cpc_list: list[dict[str, Any]] = self.config['domains'][dom].get('cpc_codes', [])
            codes: list[str] = [cpc['code'] for cpc in cpc_list]
            cpc_codes_dict[dom] = codes

        return cpc_codes_dict

    def get_global_settings(self) -> dict[str, Any]:
        """Get global configuration settings."""
        global_settings: dict[str, Any] = self.config.get('global_settings', {})
        return global_settings

##### USPTO Api Patent Puller ##################################################
class USPTOPatentPuller:
    """
    Handles US patent data extraction using PatentsView API (official USPTO).
    """
    
    def __init__(self, cpc_parser: CPCParser, api_key = API_KEY):
        self.cpc_parser = cpc_parser
        self.api_key: str = api_key
        
        # PatentsView API settings
        self.base_url: str = "https://search.patentsview.org/api/v1"
        self.patents_endpoint: str = f"{self.base_url}/patent/"
        self.rate_limit_delay: float = 1.4 # rate limit: 45 requests/minute = 0.75 requests/second
        self.headers: dict[str, str] = {
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
    def _build_search_query(self, companies: list[str], domains: list[str], 
                            start_date: str, end_date: str) -> dict[str, Any]:
        """
        Build PatentsView API search query using format from "documentation link".
        """

        # get cpc codes #
        cpc_codes_dict: dict[str, list[str]] = self.cpc_parser.cpc_codes_dict
        all_cpc_codes: list[str] = []
        for dom in domains:
            if dom in cpc_codes_dict:
                all_cpc_codes.extend(cpc_codes_dict[dom])

        # build query #
        # list of 1) company conditions, 2) cpc conditions 3) date conditions
        query_conditions: list[dict[str, Any]] = [] 

        if companies:
            company_conditions: list[dict[str, Any]] = []
            for company in companies:
                company_conditions.append({
                    "_text_any": {
                        "assignees.assignee_organization": company
                    }
                })
i edited keybinding again 

