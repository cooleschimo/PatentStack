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
from dotenv import load_dotenv

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
    
    def __init__(self, cpc_parser: CPCParser):
        self.cpc_parser = cpc_parser
        self.api_key: str = self._get_api_key()
        
        # PatentsView API settings
        self.base_url: str = "https://search.patentsview.org/api/v1"
        self.patents_endpoint: str = f"{self.base_url}/patent/"
        self.rate_limit_delay: float = 1.4 # rate limit: 45 requests/minute = 0.75 requests/second
        self.headers: dict[str, str] = {
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }

    def _get_api_key(self) -> str:
        """
        Get API key from environment variable or configuration.
        """
        api_key: str | None = os.getenv('PATENTSVIEW_API_KEY')
        if api_key:
            return api_key
       
        logger.error("PatentsView API key not found in environment variables.\n")
        raise ValueError(
            "\nRequired setup:\n"
            "Create a .env file in your project root with:\n"
            "PATENTSVIEW_API_KEY=your_actual_api_key_here\n"
            "\nGet API key from: https://www.patentsview.org/api/keyrequest"
        )
        
    def _build_search_query(self, companies: list[str], domains: list[str], 
                        start_date: str, end_date: str) -> dict[str, Any]:
        """
        Build PatentsView API search query using correct field structure.
        """
        # Get CPC codes
        cpc_codes_dict: dict[str, list[str]] = self.cpc_parser.cpc_codes_dict
        all_cpc_codes: list[str] = []
        for dom in domains:
            if dom in cpc_codes_dict:
                all_cpc_codes.extend(cpc_codes_dict[dom])

        # Build query conditions
        query_conditions: list[dict[str, Any]] = [] 

        # 1) Company conditions
        if companies:
            company_conditions: list[dict[str, Any]] = []
            for company in companies:
                company_conditions.append({
                    "_text_any": {
                        "assignees.assignee_organization": company
                    }
                })

            if len(company_conditions) == 1:
                query_conditions.append(company_conditions[0])
            else:
                query_conditions.append({"_or": company_conditions})

        # 2) CPC conditions - FIXED to use correct field structure
        if all_cpc_codes:
            cpc_conditions: list[dict[str, Any]] = []
            for code in all_cpc_codes:
                # Use full group code directly from YAML (G06N10/70, H04L9/0852, etc.)
                cpc_conditions.append({
                    "cpc_current.cpc_group_id": code
                })
            
            if cpc_conditions:
                query_conditions.append({"_or": cpc_conditions})

        # 3) Date conditions
        if start_date and end_date:
            query_conditions.extend([
                {"_gte": {"patent_date": start_date}},
                {"_lte": {"patent_date": end_date}}
            ])

        # Combine all conditions
        query = {"_and": query_conditions}

        # Fields to return - FIXED field names
        fields: list[str] = [
            "patent_id",
            "patent_title",
            "patent_abstract", 
            "patent_date",
            "patent_type",
            
            "assignees.assignee_organization",
            "assignees.assignee_city", 
            "assignees.assignee_state",
            "assignees.assignee_country",
            
            "cpc_current.cpc_subclass_id",
            "cpc_current.cpc_group_id",
            
            "inventors.inventor_name_first",
            "inventors.inventor_name_last",
            
            "patent_num_times_cited_by_us_patents"
        ]
        
        # Options
        options: dict[str, Any] = {
            "size": 100,
            "exclude_withdrawn": True
        }

        return {
            "q": query,
            "f": fields,
            "o": options
        }

    def _make_api_request(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        Make rate-limited request to PatentsView API.
        """
        logger.info(f"USPTO Query: {json.dumps(query, indent=2)}")
        response = None
        try:
            time.sleep(self.rate_limit_delay)
        
            response = requests.post(
                self.patents_endpoint,
                json=query,
                headers=self.headers,
                timeout=30
            )

            # [response status code checks] - rate or auth problems
            if response.status_code == 429:
                retry_after: int = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                response = requests.post(
                    self.patents_endpoint,
                    json=query,
                    headers=self.headers
                    )
            elif response.status_code == 403:
                logger.error("API authentication failed, check your API key.")
                raise ValueError("Invalid API key.")
        
            response.raise_for_status()

            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status:{e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")

                if e.response.status_code == 400:
                    logger.error("Bad request, check query format")
                elif e.response.status_code == 403:
                    logger.error("Authentication failed, check your API key")
                elif e.response.status_code == 404:
                    logger.error("Endpoint not found, check URL")

            raise
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            if response is not None and hasattr(response, "text"):
                logger.error(f"Response text: {response.text[:500]}")
            else:
                logger.error("No response text available.")
            raise

    def pull_us_patents(self, companies:list[str], domains: list[str],
                        start_date: str, end_date: str, 
                        max_results: int | None = None) -> pd.DataFrame:

        """
        Pull USPTO US patents for specified companies and domains from 
        PatentsView API.
        """
        logger.info(f"Pulling US patents from USPTO API")
        logger.info(f"Companies: {companies}")
        logger.info(f"Domains: {domains}")
        logger.info(f"Date range: {start_date} to {end_date}")

        base_query: dict[str, Any] = self._build_search_query(
            companies=companies,
            domains=domains,
            start_date=start_date,
            end_date=end_date
        )   
        base_page_size = base_query.get('o', {}).get("size", 100)
        
        all_patents: list[dict[str, Any]] = []
        page_count = 0
        cursor_value = None

        while True:
            page_count += 1
            
            query = base_query.copy()
            pagination_options = query.get('o', {}).copy()
            
            if cursor_value is not None:
                pagination_options['after'] = cursor_value

            query["o"] = pagination_options

            logger.info(f"Fetching page {page_count} (up to {base_page_size} records)")
            if cursor_value:
                logger.info(f"Using cursor: {cursor_value}")    

            try: 
                response: dict[str, Any] = self._make_api_request(query)
                if response.get('error', False):
                    logger.error(f"API error: {response.get('error', 'Unknown error')}")
                    break

                page_patents: list[dict[str, Any]] = response.get('patents', [])
                if not page_patents:
                    logger.info("No more patents found, reached end of results.")
                    break

                all_patents.extend(page_patents)

                # log progress
                returned_count = response.get('count', len(page_patents))
                total_available = response.get('total_hits', 'unknown')

                logger.info(f"Retrieved  {returned_count} patents from page {page_count}")
                logger.info(f"Total patents retrieved so far: {len(all_patents)}")
                logger.info(f"Total patents available: {total_available}")

                # check max results
                if max_results and len(all_patents) >= max_results:
                    all_patents = all_patents[:max_results]
                    logger.info(f"Reached max results limit: {max_results}")
                    break

                # check if last page
                if returned_count < base_page_size:
                    logger.info(f"Retrieved fewer records than page size, this is the last page.")
                    break
                    
                # set cursor for next page (using last record's patent_id)
                last_patent = page_patents[-1]
                cursor_value = last_patent.get('patent_id', None)

                if not cursor_value:
                    logger.warning("No patent_id found in last record, cannot continue pagination.")
                    break

                # prevent infinite loop
                if page_count > 1000:
                    logger.warning("Reached maximum page count (1000), stopping.")
                    break

            except Exception as e:
                logger.error(f"Error fetching page {page_count}: {e}")
                if page_count == 1:
                    logger.error("Failed to fetch any patents, aborting.")
                    raise # if first page fails, reraise the error
                else: # if later page fail, just break and return what we have
                    logger.warning(f"Continuing with {len(all_patents)} patents fetched so far.")
                    break

        logger.info(f"Successfully fetched {len(all_patents)} patents from USPTO.")
        
        df = self._standardize_uspto_data(all_patents, companies, domains)

        return df


    def _standardize_uspto_data(self, patents: list[dict[str, Any]], 
                            companies: list[str], 
                            domains: list[str]) -> pd.DataFrame:
        """
        Standardize the raw patent data from USPTO into a DataFrame.
        """
        standardized_data: list[dict[str, Any]] = []

        for patent in patents:
            standardized_patent: dict[str, Any] = {
                'publication_number': patent.get('patent_id', ''),  # FIXED field name
                'title': patent.get('patent_title', ''),
                'abstract': patent.get('patent_abstract', ''),
                'publication_date': patent.get('patent_date', ''),
                'filing_date': '',
                'country_code': 'US',
                'kind_code': patent.get('patent_type', ''),
                'application_number': '',

                'assignees': self._extract_assignees(patent),
                'inventors': self._extract_inventors(patent),
                'cpc_codes': self._extract_cpc_codes(patent),

                'cited_by_count': patent.get('patent_num_times_cited_by_us_patents', 0),
                'family_id': '',
                'priority_date': '',

                'patent_url': f"https://patents.uspto.gov/patent/{patent.get('patent_id', '')}",
                'extracted_at': datetime.now().isoformat(),
                'target_domains': ','.join(domains),
                'target_companies': ','.join(companies),
                'data_source': 'USPTO_API'
            }
            standardized_data.append(standardized_patent)

        df = pd.DataFrame(standardized_data)
        
        # Add domain relevance columns
        for domain in domains:
            df[f'{domain}_relevant'] = False

        return df

    def _extract_cpc_codes(self, patent: dict[str, Any]) -> str:
        """Extract CPC codes from cpc_current structure."""
        cpc_current = patent.get('cpc_current', [])
        if cpc_current and len(cpc_current) > 0:
            return cpc_current[0].get('cpc_subclass_id', '')
        return ''   
    
    def _extract_assignees(self, patent: dict[str, Any]) -> str:  
        """Extract assignee information from nested structure."""
        assignees = patent.get('assignees', [])
        if assignees and len(assignees) > 0:
            return assignees[0].get('assignee_organization', '')
        return ''

    def _extract_inventors(self, patent: dict[str, Any]) -> str:  
        """Extract inventor information from nested structure."""
        inventors = patent.get('inventors', [])
        if inventors and len(inventors) > 0:
            first = inventors[0].get('inventor_name_first', '')
            last = inventors[0].get('inventor_name_last', '')
            return f"{first} {last}".strip()
        return ''
    
###Google Patent Puller ########################################################
class GooglePatentPuller:
    """
    Handles international patent data extraction using BigQuery.
    """
    project_id: str | None
    cpc_parser: CPCParser
    client: bigquery.Client
    global_settings: dict[str, Any]
    patents_dataset: str

    def __init__(self, cpc_parser: CPCParser) -> None:
        self.project_id = os.getenv('BIGQUERY_PROJECT_ID')
        if not self.project_id:
            raise ValueError(
                "BigQuery project ID not found in environment variables. "
                "Please set BIGQUERY_PROJECT_ID environment variable."
            )
        self.cpc_parser = cpc_parser
        self.client = bigquery.Client(project=self.project_id)
        self.global_settings = cpc_parser.global_settings
        self.patents_dataset = "patents-public-data.patents.publications"

    def _build_intl_query(self, companies: list[str],
                      domains: list[str],
                      start_date: str, end_date: str,
                      exclude_countries: list[str]) -> str:
        """
        Build working BigQuery SQL using correct field names.
        """
        # Get CPC codes and extract subclass IDs
        cpc_codes_dict: dict[str, list[str]] = self.cpc_parser.cpc_codes_dict
        all_cpc_codes: list[str] = []
        for dom in domains:
            if dom in cpc_codes_dict:
                all_cpc_codes.extend(cpc_codes_dict[dom])

        subclass_ids: set[str] = set()
        for code in all_cpc_codes:
            if len(code) >= 6:
                subclass_ids.add(code[:6])

        # Build CPC conditions
        cpc_conditions: list[str] = []
        for code in all_cpc_codes:
            # Use exact match for all codes from YAML (G06N10/70, H04L9/0852, etc.)
            cpc_conditions.append(f"c.code = '{code}'")

        cpc_filter = ' OR '.join(cpc_conditions)

        # Company conditions
        company_conditions: list[str] = []
        for company in companies:
            company_conditions.append(f"LOWER(a.name) LIKE '%{company.lower()}%'")
        
        company_filter = ' OR '.join(company_conditions)

        # Date filter
        start_date_int = start_date.replace('-', '')
        end_date_int = end_date.replace('-', '')

        # Simplified working query
        query = f"""
        SELECT
            publication_number,
            publication_date,
            filing_date,
            country_code,
            kind_code,
            application_number,
            
            (SELECT a.name FROM UNNEST(assignee_harmonized) a LIMIT 1) AS assignee,
            
            ARRAY(
            SELECT c.code
            FROM UNNEST(cpc) AS c
            WHERE {cpc_filter}
            ) AS cpc_codes,
            
            title_localized AS title,
            abstract_localized AS abstract,
            family_id,
            priority_date,
            
            CONCAT('https://patents.google.com/patent/', publication_number) as patent_url,
            CURRENT_TIMESTAMP() as extracted_at,
            '{",".join(domains)}' as target_domains,
            '{",".join(companies)}' as target_companies,
            'BigQuery' as data_source
            
        FROM `patents-public-data.patents.publications`
        WHERE publication_date >= {start_date_int}
            AND publication_date <= {end_date_int}
            AND country_code NOT IN ('{"', '".join(exclude_countries)}')
            AND EXISTS (
                SELECT 1
                FROM UNNEST(assignee_harmonized) a
                WHERE {company_filter}
            )
            AND EXISTS (
                SELECT 1
                FROM UNNEST(cpc) c
                WHERE {cpc_filter}
            )
        ORDER BY publication_date DESC
        """

        return query
    
    def _execute_query(self, query: str, max_cost_usd: float | None) -> pd.DataFrame:
        """
        Execute BigQuery and return results in a df (with optional cost limit)
        """
        logger.info(f"BigQuery SQL: {query}")
        # estimate cost
        job_config: bigquery.QueryJobConfig = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

        try:
            job: bigquery.QueryJob = self.client.query(query, job_config=job_config)
            bytes_processed: int = job.total_bytes_processed
            gb_processed: float = bytes_processed / (1024**3)
            tb_processed: float = bytes_processed / (1024**4)
            estimated_cost: float = max(0, (tb_processed - 1.0)) * 5.0
            
            logger.info(f"International query will process ~{gb_processed:.2f} GB")
            logger.info(f"Estimated cost: ${estimated_cost:.2f}")
            
            # Only check cost limit if one is provided
            if max_cost_usd is not None and estimated_cost > max_cost_usd:
                raise ValueError(
                    f"Query cost (${estimated_cost:.2f}) exceeds maximum allowed (${max_cost_usd:.2f}). "
                    f"To proceed anyway, set max_cost_usd=None or increase the limit."
                )
            elif max_cost_usd is None:
                logger.info("No cost limit set - proceeding with query execution")
            else:
                logger.info(f"Cost ${estimated_cost:.2f} is within limit ${max_cost_usd:.2f} - proceeding")
            
            # execute query
            logger.info("Executing BigQuery for international patents...")
            start_time: float = time.time()
            
            job = self.client.query(query)
            results = job.result()
            df: pd.DataFrame = results.to_dataframe()
            
            execution_time: float = time.time() - start_time
            logger.info(f"BigQuery query completed in {execution_time:.2f} seconds")
            logger.info(f"Retrieved {len(df)} international patent records")
            logger.info(f"Final cost: ${estimated_cost:.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error executing BigQuery query: {e}")
            raise


    def pull_international_patents(self, companies: list[str], 
                                   domains: list[str],
                                   start_date: str, end_date: str, 
                                   exclude_countries: list[str] = ['US'],
                                   max_cost_usd: float | None = None) -> pd.DataFrame:
        """
        Pull international patents (non-US) from BigQuery
        """
        logger.info(f"Pulling international patents using BigQuery")
        logger.info(f"Excluding countries: {exclude_countries}")
        if max_cost_usd is None:
            logger.info("No cost limit set - run query regardless of estimated cost")
        else:
            logger.info(f"Cost limit: ${max_cost_usd}")

        # build query
        query: str = self._build_intl_query(companies, domains, start_date, end_date, exclude_countries)

        df = self._execute_query(query, max_cost_usd)

        df = self._standardize_bigquery_data(df, companies, domains)

        return df

    def _standardize_bigquery_data(self, df: pd.DataFrame, 
                                  companies: list[str], domains: list[str]) -> pd.DataFrame:
        """
        Convert BigQuery data to standardized format (add domain classification columns)
        """
        df['data_source'] = 'BigQuery'
        
        for domain in domains:
            df[f'{domain}_relevant'] = False
        
        return df
    
### Hybrid Puller #########################################################
class HybridPatentPuller:
    """
    Orchestrates both US and international patent data extraction.
    """
    def __init__(self, cpc_parser: CPCParser) -> None:
        self.cpc_parser = cpc_parser
        
        # USPTO puller 
        self.uspto_puller = USPTOPatentPuller(cpc_parser)
        logger.info("USPTO API initialized for US patents (FREE)")

        # BigQuery puller
        try:
            self.bigquery_puller = GooglePatentPuller(cpc_parser)
            logger.info("BigQuery initialized for international patents")
        except Exception as e:
            logger.error(f"BigQuery initialization failed: {e}")
            raise

    def pull_patents_recent_first(self, companies: list[str], 
                                 start_year: int, end_year: int,
                                 domains: list[str] | None = None,
                                 max_international_cost: float = 10.0,
                                 output_dir: str = "data/raw/") -> dict[str, list[pd.DataFrame]]:
        """
        Pull patents starting from most recent year, working backwards.
        
        Args:
            companies: List of company names to search for
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive) 
            domains: Technology domains to search (uses all if None)
            max_international_cost: Maximum BigQuery cost in USD
            output_dir: Directory to save CSV files
            
        Returns:
            Dictionary with 'us_patents' and 'international_patents' DataFrames
        """
        
        # Use all domains if none specified
        if domains is None:
            domains = self.cpc_parser.domains
            logger.info(f"Using all configured domains: {domains}")
        
        # Generate years in reverse order (recent first)
        years: list[int] = list(range(end_year, start_year - 1, -1))
        logger.info(f"Processing years: {years} (recent to older)")
        
        results: dict[str, list[pd.DataFrame]] = {
            'us_patents': [],
            'international_patents': []
        }
        
        total_years = len(years)

        for i, year in enumerate(years):
            year_start: str = f"{year}-01-01"
            year_end: str = f"{year}-12-31"
            
            logger.info(f"Processing year {year} ({i+1}/{total_years})")
            
            try:
                # Process each company separately
                for company in companies:
                    logger.info(f"Processing {company} for year {year}")
                    
                    # Always pull US patents
                    try:
                        us_df = self.uspto_puller.pull_us_patents(
                            companies=[company],
                            domains=domains,
                            start_date=year_start,
                            end_date=year_end
                        )
                        
                        if not us_df.empty:
                            us_output_path = f"{output_dir}/US/{company}/{company}_us_patents_{year}.csv"
                            os.makedirs(os.path.dirname(us_output_path), exist_ok=True)
                            us_df.to_csv(us_output_path, index=False)
                            logger.info(f"Saved {len(us_df)} US patents for {company} {year}")
                            results['us_patents'].append(us_df)
                        else:
                            logger.info(f"No US patents found for {company} {year}")
                    
                    except Exception as e:
                        logger.error(f"Failed to get US patents for {company} {year}: {e}")
                    
                    # Pull international patents
                    try:
                        intl_df = self.bigquery_puller.pull_international_patents(
                            companies=[company],
                            domains=domains,
                            start_date=year_start,
                            end_date=year_end,
                            max_cost_usd=max_international_cost
                        )
                        
                        if not intl_df.empty:
                            intl_output_path = f"{output_dir}/International/{company}/{company}_intl_patents_{year}.csv"
                            os.makedirs(os.path.dirname(intl_output_path), exist_ok=True)
                            intl_df.to_csv(intl_output_path, index=False)
                            logger.info(f"Saved {len(intl_df)} international patents for {company} {year}")
                            results['international_patents'].append(intl_df)
                        else:
                            logger.info(f"No international patents found for {company} {year}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to get international patents for {company} {year}: {e}")
                        logger.info("Continuing with US patents only...")
                
            except Exception as e:
                logger.error(f"Critical error processing year {year}: {e}")
                continue

        # Final summary
        total_us = sum(len(df) for df in results['us_patents'])
        total_intl = sum(len(df) for df in results['international_patents'])
        
        logger.info(f"Patent extraction complete!")
        logger.info(f"Total US patents: {total_us}")
        logger.info(f"Total international patents: {total_intl}")
        
        if total_intl > 0:
            estimated_cost = total_intl * 0.01  # Rough estimate
            logger.info(f"Estimated BigQuery cost: ${estimated_cost:.2f}")
        
        logger.info(f"Data saved to: {output_dir}")
        
        return results
    
### MAIN FUNCTION ##############################################################
def main() -> None:
    """Command-line interface for the hybrid patent puller."""
    load_dotenv('../../.env')

    parser = argparse.ArgumentParser(
        description='Patent Data Fetcher - Pull patent data from USPTO and BigQuery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python patent_fetcher.py --companies "IBM" "Google" --start-year 2022 --end-year 2024 --domains quantum_computing
  
  python patent_fetcher.py --companies "IBM" --start-year 2023 --end-year 2023 --bigquery-project-id my-project-123
        """
    )
    
    # Required arguments
    parser.add_argument('--companies', nargs='+', required=True,
                       help='Company names to search for')
    parser.add_argument('--start-year', type=int, required=True,
                       help='Start year for patents')
    parser.add_argument('--end-year', type=int, required=True,
                       help='End year for patents')
    
    # Optional arguments
    parser.add_argument('--domains', nargs='+',
                       help='Technology domains (default: all domains in config)')
    parser.add_argument('--config', default='../../config/cpc_codes.yaml',
                       help='Path to CPC configuration file')
    parser.add_argument('--max-international-cost', type=float, default=10.0,
                       help='Maximum BigQuery cost in USD')
    parser.add_argument('--output-dir', default='../../data/raw/',
                       help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    # Validation
    if args.start_year > args.end_year:
        parser.error("Start year must be <= end year")
    
    if args.end_year > datetime.now().year:
        parser.error(f"End year cannot be in the future (current year: {datetime.now().year})")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        cpc_parser = CPCParser(args.config)
        
        # Validate domains
        if args.domains:
            invalid_domains = [d for d in args.domains if d not in cpc_parser.domains]
            if invalid_domains:
                logger.error(f"Invalid domains: {invalid_domains}")
                logger.info(f"Available domains: {cpc_parser.domains}")
                sys.exit(1)
        
        # Initialize hybrid puller
        logger.info("Initializing patent data fetcher...")
        puller = HybridPatentPuller(cpc_parser=cpc_parser)
        
        # Display what we're about to do
        logger.info(f"Companies: {args.companies}")
        logger.info(f"Years: {args.start_year} to {args.end_year}")
        logger.info(f"Domains: {args.domains or cpc_parser.domains}")
        logger.info(f"BigQuery cost limit: ${args.max_international_cost}")
        
        # Pull patents
        logger.info("Starting patent extraction...")
        results = puller.pull_patents_recent_first(
            companies=args.companies,
            start_year=args.start_year,
            end_year=args.end_year,
            domains=args.domains,
            max_international_cost=args.max_international_cost,
            output_dir=args.output_dir
        )
        
        # Final summary
        total_us = sum(len(df) for df in results['us_patents'])
        total_intl = sum(len(df) for df in results['international_patents'])
        
        print(f"Patent extraction completed successfully!")
        print(f"Results:")
        print(f"   US patents: {total_us:,}")
        print(f"   International patents: {total_intl:,}")
        print(f"   Total patents: {total_us + total_intl:,}")
        print(f"Data saved to: {args.output_dir}")
        
        if total_intl > 0:
            estimated_cost = total_intl * 0.01
            print(f"Estimated BigQuery cost: ${estimated_cost:.2f}")
        
        print(f"Next steps:")
        print(f"   1. Check your data: ls {args.output_dir}")
        print(f"   2. Start ML training with the CSV files")
        
    except KeyboardInterrupt:
        logger.info("Extraction cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()