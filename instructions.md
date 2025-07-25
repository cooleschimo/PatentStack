# BigQuery Patent Puller - Requirements & Usage

## Installation Requirements

**BigQuery is required for this application:**

```bash
pip install google-cloud-bigquery pandas pyyaml requests python-dotenv
```

Or add to your `requirements.txt`:
```
google-cloud-bigquery>=3.11.0
pandas>=1.5.0
PyYAML>=6.0
requests>=2.28.0
python-dotenv>=1.0.0
```

**⚠️ Important**: This application requires BigQuery even for US-only patents to maintain consistent architecture.

## Setup Instructions

### **1. Install Dependencies**
```bash
pip install google-cloud-bigquery pandas pyyaml requests python-dotenv
```

### **2. Get API Keys**

#### **PatentsView API Key (Required for US Patents)**
1. Go to [PatentsView API Key Request](https://www.patentsview.org/api/keyrequest)
2. Fill out the form with your details
3. Wait for email with your API key

#### **Google Cloud Setup (Required for International Patents)**
1. Sign up for [Google Cloud](https://console.cloud.google.com/) ($300 free credits)
2. Create a new project
3. Enable BigQuery API
4. Set up authentication:
   ```bash
   # Option A: Use gcloud CLI
   gcloud auth application-default login
   
   # Option B: Download service account key (save as JSON file)
   ```

### **3. Configure Environment Variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual keys
nano .env
```

**Your `.env` file should look like:**
```bash
PATENTSVIEW_API_KEY=abcd1234-your-actual-key-here
BIGQUERY_PROJECT_ID=your-gcp-project-id
```

### **4. Verify Setup**
```bash
# Test your configuration
python setup_check.py --project-id your-gcp-project-id

# If everything works, you'll see:
# ✅ All checks passed! Your system is ready to extract patents.
```

**⚠️ Security Note:** Never commit your `.env` file to git. It's already in `.gitignore`.

## Configuration File Structure

Save your configuration as `config/patent_config.yaml` (the script will read from this path by default):

```yaml
# Your existing configuration structure
domains:
  quantum_computing:
    # ... your quantum computing config
  quantum_safe_communications:
    # ... your quantum communications config
    
global_settings:
  min_abstract_length: 100
  default_batch_size: 1000
```

## Usage Examples

### 1. Basic Usage - Single Company, All Domains
```bash
python bigquery_pull.py \
  --companies "IBM" \
  --project-id "your-gcp-project-id" \
  --limit 100 \
  --output "ibm_patents.csv"
```

### 2. Multiple Companies, Specific Domain
```bash
python bigquery_pull.py \
  --companies "Google" "Microsoft" "Amazon" \
  --domains "quantum_computing" \
  --project-id "your-gcp-project-id" \
  --start-date "2020-01-01" \
  --limit 500 \
  --output "tech_giants_quantum.csv"
```

### 3. Recent Patents with Geographic Focus
```bash
python bigquery_pull.py \
  --companies "QUALCOMM" "Intel" \
  --domains "quantum_computing" "quantum_safe_communications" \
  --project-id "your-gcp-project-id" \
  --start-date "2022-01-01" \
  --countries "US" "EP" \
  --limit 200 \
  --max-cost 5.0 \
  --output "recent_us_eu_patents.csv"
```

### 4. List Available Domains
```bash
python bigquery_pull.py --list-domains --config "config/patent_config.yaml"
```

### 5. Large-Scale Data Pull (Be Careful with Costs!)
```bash
python bigquery_pull.py \
  --companies "IBM" "Google" "Microsoft" "Amazon" "Apple" \
  --project-id "your-gcp-project-id" \
  --start-date "2015-01-01" \
  --end-date "2024-12-31" \
  --limit 5000 \
  --max-cost 25.0 \
  --output "big_tech_patents_decade.csv"
```

## Cost Management Tips

### 1. **Always Set Limits**
- Use `--limit` to cap results
- Use `--max-cost` to prevent expensive queries
- Start with small date ranges

### 2. **Optimize Date Ranges**
```bash
# Good: Specific recent period
--start-date "2023-01-01" --end-date "2023-12-31"

# Expensive: No date filter (queries all historical data)
# (omitting date parameters)
```

### 3. **Country Filtering Saves Costs**
```bash
# More focused and cheaper
--countries "US" "EP"

# Processes all global patents (expensive)
# (omitting --countries)
```

### 4. **Monitor Query Costs**
The script will show estimated costs before execution:
```
Query will process ~15.23 GB
Estimated cost: $0.00 (within free tier)
```

## Understanding the Output

### Key Columns in Results:
- `publication_number`: Patent identifier
- `title`: Patent title  
- `abstract`: Patent abstract text
- `assignees`: Company/organization info (JSON array)
- `inventors`: Inventor information (JSON array)
- `cpc_codes`: Classification codes (JSON array)
- `publication_date`: When patent was published
- `filing_date`: When patent was filed
- `patent_url`: Direct link to Google Patents
- `target_domains`: Which domains this query targeted
- `target_companies`: Which companies this query targeted

### Working with Array Columns:
The assignees, inventors, and cpc_codes columns contain JSON arrays. In pandas:

```python
import pandas as pd
import json

df = pd.read_csv('patents.csv')

# Access first assignee name
df['first_assignee'] = df['assignees'].apply(
    lambda x: json.loads(x)[0]['name'] if x != '[]' else None
)

# Count number of inventors
df['inventor_count'] = df['inventors'].apply(
    lambda x: len(json.loads(x))
)
```

## Troubleshooting

### Common Issues:

1. **"Query cost exceeds maximum"**
   - Reduce date range
   - Add `--limit` parameter
   - Increase `--max-cost` if needed
   - Add country filters

2. **"Domain not found in configuration"**
   - Run `--list-domains` to see available domains
   - Check your config file path with `--config`

3. **Authentication errors**
   - Verify `GOOGLE_APPLICATION_CREDENTIALS` is set
   - Or run `gcloud auth application-default login`

4. **No results returned**
   - Check company name spelling/variations
   - Try broader date ranges
   - Verify the domains have relevant CPC codes

### Query Debugging:
The script saves each query as `query_YYYYMMDD_HHMMSS.sql` for debugging.

## Advanced Configuration

### Adding New Technology Domains:
Edit your `config/patent_config.yaml`:

```yaml
domains:
  # Add your new domain
  artificial_intelligence:
    description: "AI and machine learning technologies"
    cpc_codes:
      - code: "G06N3/02"
        description: "Neural networks"
        keywords:
          - "neural network"
          - "deep learning"
          - "machine learning"
    synonyms:
      - "artificial intelligence"
      - "AI"
      - "ML"
    application_domains:
      - "computer vision"
      - "natural language processing"
```

### Customizing Quality Filters:
```yaml
global_settings:
  min_abstract_length: 200  # Require longer abstracts
  noise_terms:
    exclude_when_alone:
      - "software"  # Add terms that create false positives
```

## Integration with Your Pipeline

The script is designed to integrate with your ML pipeline:

1. **Data Collection**: Use this script to pull raw patent data
2. **Storage**: Results are saved as CSV, ready for database import
3. **ML Classification**: The `{domain}_relevant` columns are pre-added for your classifier
4. **Analytics**: The structured output works with your dashboard

```python
# Example integration
from bigquery_pull import BigQueryPatentPuller, PatentConfigLoader

config = PatentConfigLoader('config/patent_config.yaml')
puller = BigQueryPatentPuller('your-project-id', config)

# Pull data programmatically
df = puller.pull_patents(
    companies=['IBM', 'Google'],
    domains=['quantum_computing'],
    start_date='2023-01-01',
    limit=1000
)

# Now feed to your ML classifier...
```