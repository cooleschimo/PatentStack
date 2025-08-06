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

        # 2) CPC conditions
        if all_cpc_codes:
            cpc_conditions: list[dict[str, Any]] = []
            for code in all_cpc_codes:
                # Use the correct CPC field name from PatentsView
                cpc_conditions.append({
                    "cpcs.cpc_subclass_id": code
                })
            
            if len(cpc_conditions) == 1:
                query_conditions.append(cpc_conditions[0])
            else:
                query_conditions.append({"_or": cpc_conditions})

        # 3) Date conditions
        if start_date and end_date:
            query_conditions.extend([
                {"_gte": {"patent_date": start_date}},
                {"_lte": {"patent_date": end_date}}
            ])

        # Combine all conditions
        query = {"_and": query_conditions}

        # fields to return
        fields: list[str] = [
            "patent_id",
            "patent_number", 
            "patent_title",
            "patent_abstract",
            "patent_date",           
            "patent_type",           
            
            "assignees.assignee_organization", 
            "assignees.assignee_location_city",      
            "assignees.assignee_location_state",     
            "assignees.assignee_location_country",
        
    
            "cpcs.cpc_subclass_id",                
            "cpcs.cpc_group_id",                   
            "cpcs.cpc_subgroup_id",               
            
            "inventors.inventor_name_first",         
            "inventors.inventor_name_last",         
    
            "patent_num_times_cited_by_us_patents",  

        ]
        
        # options
        options: dict[str, Any] = {
        "size": 100,
        "exclude_withdrawn": True  # only granted patents
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
                if response.get('error', True):
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
            # extract basic fields
            standardized_patent: dict[str, Any] = {
                'publication_number': patent.get('patentNumber', ''),
                'title': patent.get('patentTitle', ''),
                'abstract': patent.get('patentAbstract', ''),
                'publication_date': patent.get('patentIssueDate', ''),
                'filing_date': patent.get('patentApplicationFilingDate', ''),
                'country_code': 'US',
                'kind_code': '', # USPTO doesnt have this
                'application_number': '', # another API call for this

                # assignee info
                'assignees': self._extract_assignees(patent),

                # inventors
                'inventors': self._extract_inventors(patent),

                # cpc_codes
                'cpc_codes': self._extract_cpc_codes(patent),

                # citation information
                'cited_by_count': patent.get('patentCitationCount', 0),

                # Family and priority info (not available from USPTO API)
                'family_id': '',
                'priority_date': '',

                # patent url
                'patent_url': f"https://patents.uspto.gov/patent/{patent.get('patentNumber', '')}",

                 # Extraction metadata
                'extracted_at': datetime.now().isoformat(),
                'target_domains': ','.join(domains),
                'target_companies': ','.join(companies),
                'data_source': 'USPTO_API'
            }

            standardized_data.append(standardized_patent)

        df = pd.DataFrame(standardized_data)

        for domain in domains:
            column_name = f'{domain}_relevant'
            df[column_name] = False

        return df
    
    def _extract_assignees(self, patent: dict[str, Any]) -> list[dict[str, str]]:
        """Extract and normalize assignee information."""
        assignees: list[dict[str, str]] = []
        
        assignee_org: str = patent.get('assigneeOrganization', '')
        if assignee_org:
            assignees.append({
                'name': assignee_org,
                'harmonized_name': assignee_org,  # USPTO doesn't provide harmonized names
                'country': 'US'  # Assume US for USPTO patents
            })
        
        return assignees
    
    def _extract_inventors(self, patent: dict[str, Any]) -> list[dict[str, str]]:
        """Extract and normalize inventor information."""
        inventors: list[dict[str, str]] = []
        
        inventor_names: str = patent.get('inventorName', '')
        if inventor_names:
            # Split multiple inventors (typically separated by semicolons)
            names: list[str] = [name.strip() for name in inventor_names.split(';') if name.strip()]
            
            for name in names:
                inventors.append({
                    'name': name,
                    'harmonized_name': name,
                    'country': 'US'  # Assume US for USPTO patents
                })
        
        return inventors
    
    def _extract_cpc_codes(self, patent: dict[str, Any]) -> list[dict[str, str]]:
        """Extract and normalize CPC code information."""
        cpc_codes: list[dict[str, Any]] = []
        
        cpc_classification: str = patent.get('cpcClassificationCode', '')
        if cpc_classification:
            # Split multiple CPC codes (typically separated by semicolons)
            codes: list[str] = [code.strip() for code in cpc_classification.split(';') if code.strip()]
            
            for i, code in enumerate(codes):
                cpc_codes.append({
                    'code': code,
                    'is_first': i == 0,  # First code is primary
                    'tree': ''  # USPTO doesn't provide tree structure
                })
        
        return cpc_codes
    
###Google Patent Puller ########################################################
class GooglePatentPuller:
    """
    Handles international patent data extraction using BigQuery.
    """
    project_id: str
    cpc_parser: CPCParser
    client: bigquery.Client
    global_settings: dict[str, Any]
    patents_dataset: str

    def __init__(self, project_id: str, cpc_parser: CPCParser) -> None:
        self.project_id = project_id
        self.cpc_parser = cpc_parser
        self.client = bigquery.Client(project=project_id)
        self.global_settings = cpc_parser.global_settings
        self.patents_dataset = "patents-public-data.patents.publications"

    def _build_intl_query(self, companies: list[str],
                          domains: list[str],
                          start_date: str, end_date: str,
                          exclude_countries: list[str]) -> str:
        """
        Build BigQuery SQL for Google patents.
        """
        # get cpc codes #
        cpc_codes_dict: dict[str, list[str]] = self.cpc_parser.cpc_codes_dict
        all_cpc_codes: list[str] = []
        for dom in domains:
            if dom in cpc_codes_dict:
                all_cpc_codes.extend(cpc_codes_dict[dom])

        # company conditions #
        company_conditions: list[str] = []
        for coy in companies:
            company_conditions.append(f"UPPER(assignee_organization) LIKE '%{coy.strip().upper()}%'")

        company_filter: str = f"({' OR '.join(company_conditions)})" if company_conditions else "1=1"

        # cpc conditions # 
        cpc_coonditions: list[str] = []
        for code in all_cpc_codes:
            cpc_coonditions.append(f"cpc_code = '{code}'")

        cpc_filter: str = f"EXISTS (SELECT 1 FROM UNNEST(cpc) AS cpc WHERE {' OR '.join(cpc_coonditions)})"

        # exclude countries (US by default) #
        excluded_countries_str: str = "', '".join(exclude_countries)
        country_filter: str = f"country_code NOT IN ('{excluded_countries_str}')"

         # date filter #
        date_filter: str = f"publication_date >= '{start_date}' AND publication_date <= '{end_date}'"

        # combine conditions
        where_conditions: list[str] = [company_filter, cpc_filter, country_filter, date_filter]
        where_clause: str = " AND ".join(where_conditions)

        # domains and coy strings
        domains_str: str = ','.join(domains)
        companies_str: str = ','.join(companies)

        # build query
        query: str = f"""
        SELECT 
            publication_number,
            publication_date,
            filing_date,
            country_code,
            kind_code,
            application_number,
            title,
            abstract,
            
            -- Assignee information
            ARRAY(
                SELECT AS STRUCT 
                    assignee.name as name,
                    assignee.harmonized as harmonized_name,
                    assignee.country_code as country
                FROM UNNEST(assignee) AS assignee
            ) as assignees,
            
            -- Inventor information
            ARRAY(
                SELECT AS STRUCT
                    inventor.name as name,
                    inventor.harmonized as harmonized_name,
                    inventor.country_code as country
                FROM UNNEST(inventor) AS inventor
            ) as inventors,
            
            -- CPC codes
            ARRAY(
                SELECT AS STRUCT
                    cpc.code as code,
                    cpc.first as is_first,
                    cpc.tree as tree
                FROM UNNEST(cpc) AS cpc
            ) as cpc_codes,
            
            cited_by_count,
            family_id,
            priority_date,
            
            CONCAT('https://patents.google.com/patent/', publication_number) as patent_url,
            CURRENT_TIMESTAMP() as extracted_at,
            '{domains_str}' as target_domains,
            '{companies_str}' as target_companies
            
        FROM `{self.patents_dataset}`
        WHERE {where_clause}
        ORDER BY publication_date DESC
        """

        return query
    
    def _execute_query(self, query: str, max_cost_usd: float | None) -> pd.DataFrame:
        """
        Execute BigQuery and return results in a df (with optional cost limit)
        """
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

        df['data_source'] = 'BigQuery'

        return df

    def _standardize_bigquery_data(self, df: pd.DataFrame, 
                                  companies: list[str], domains: list[str]) -> pd.DataFrame:
        """
        Convert BigQuery data to standardized format (add domain classification columns)
        """
        
        for domain in domains:
            df[f'{domain}_relevant'] = False
        
        return df
    
### Hybrid Puller #########################################################
class HybridPatentPuller:
        