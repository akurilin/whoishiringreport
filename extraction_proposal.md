# Job Posting Data Extraction Proposal

## Overview
We need to extract structured information from the `full_content` field in `matches.json` (472 matches). The data varies significantly in format, from structured pipe-separated headers to free-form text.

## Fields to Extract
1. **Company name** - e.g., "NVIDIA", "PhysicianX", "Freed"
2. **Role name** - e.g., "Head of Engineering", "Engineering Manager"
3. **Remote status** - boolean or enum (Remote, Onsite, Hybrid, Remote-friendly)
4. **Physical location** - city/state/country if specified (e.g., "Buffalo, NY", "SF / NYC / DC")
5. **Employment type** - Full-time, Part-time, Contract, Fractional
6. **Cash compensation** - salary range (e.g., "$160k–$250k", "$130,000 - $250,000")
7. **Equity compensation** - boolean or description (e.g., "+ equity", "equity ownership")

## Data Characteristics
- **Structured formats**: Many posts use pipe-separated headers like "Company | Role | Location | Type | Compensation"
- **Free-form text**: Some posts embed information in paragraphs
- **Multiple roles**: Some posts list multiple positions
- **Varied compensation formats**: "$160k–$250k", "$130,000 - $250,000", "$178,500 - $320,000 USD"
- **Location variations**: "Remote", "Remote (US only)", "SF / NYC / DC - Hybrid", "Onsite in Philadelphia"
- **HTML entities**: Content includes HTML tags and entities that need cleaning

---

## Option 1: Pure LLM Approach (Recommended for Accuracy)

### Description
Use OpenAI's cheapest model (gpt-4o-mini) to extract all fields via structured output/JSON mode.

### Pros
- **Highest accuracy** - Handles all format variations and edge cases
- **Context understanding** - Can infer information from context
- **Easy to maintain** - No regex patterns to update
- **Handles ambiguity** - Can make reasonable inferences

### Cons
- **Cost** - ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **Latency** - API calls add processing time (can batch)
- **API dependency** - Requires internet and API key

### Cost Estimate
- Average post length: ~500 tokens
- 472 posts × 500 tokens = ~236k input tokens
- Output: ~50 tokens per post = ~23.6k output tokens
- **Estimated cost: ~$0.04-0.05 total** (very cheap!)

### Implementation
```python
import openai
import json

def extract_with_llm(content: str) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "Extract job posting information. Return JSON only."
        }, {
            "role": "user",
            "content": f"Extract: company_name, role_name, is_remote, location, employment_type, cash_compensation, equity_compensation\n\n{content}"
        }],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)
```

### Speed Optimization
- Batch multiple posts in single API call (up to 128k context)
- Use async/parallel requests
- Cache results to avoid re-processing

---

## Option 2: Hybrid Approach (Regex + LLM Fallback)

### Description
Use regex patterns for structured posts (pipe-separated headers), fall back to LLM for free-form text.

### Pros
- **Cost efficient** - Only uses LLM when needed (~30-50% of posts)
- **Fast for structured posts** - Regex is instant
- **Accurate for complex cases** - LLM handles edge cases

### Cons
- **More complex** - Need to maintain regex patterns
- **Pattern maintenance** - May need updates as formats change
- **Still requires LLM** - For free-form posts

### Implementation Strategy
1. Detect structured format (pipe-separated header)
2. Parse with regex if detected
3. Validate extracted fields (confidence check)
4. Fall back to LLM if regex fails or confidence low

### Cost Estimate
- Assume 50% can be regex-parsed
- 236 posts × LLM cost = ~$0.02-0.03 total

---

## Option 3: Rule-Based with Heuristics

### Description
Comprehensive regex patterns + heuristics for common patterns, no LLM.

### Pros
- **No API costs** - Completely free
- **Fast** - All local processing
- **No dependencies** - Works offline

### Cons
- **Lower accuracy** - Will miss edge cases
- **High maintenance** - Need many regex patterns
- **Fragile** - Breaks with format changes
- **Limited inference** - Can't understand context

### Implementation Challenges
- Company name detection (first word? before pipe? in text?)
- Role name extraction (multiple roles in one post?)
- Location parsing (many formats)
- Compensation parsing (currency symbols, ranges, formats)
- Employment type detection (full-time vs full time, etc.)

---

## Option 4: Two-Pass Hybrid (Recommended for Balance)

### Description
1. **First pass**: Regex for obvious structured patterns (quick wins)
2. **Second pass**: LLM for everything else, but with regex hints

### Pros
- **Best of both worlds** - Fast for structured, accurate for complex
- **Cost efficient** - Only LLM for ~40-60% of posts
- **Can improve over time** - Add regex patterns as patterns emerge

### Cons
- **More code** - Two extraction paths
- **Still needs LLM** - For free-form posts

### Implementation
```python
def extract_job_info(content: str) -> dict:
    # Try regex first
    structured_match = parse_structured_header(content)
    if structured_match and validate_extraction(structured_match):
        return structured_match
    
    # Fall back to LLM
    return extract_with_llm(content)
```

---

## Recommendation: **Option 1 (Pure LLM)** or **Option 4 (Two-Pass Hybrid)**

### Why Option 1?
- **Cost is negligible** (~$0.05 for all 472 posts)
- **Highest accuracy** - Handles all edge cases
- **Simplest to implement** - Single extraction path
- **Easiest to maintain** - No regex patterns
- **Fast enough** - Can batch/parallelize API calls

### Why Option 4?
- **Slightly cheaper** (~$0.02-0.03)
- **Faster for structured posts** - Instant regex parsing
- **Good accuracy** - LLM for complex cases
- **Scalable** - Can add more regex patterns over time

---

## Implementation Plan (Option 1 - Pure LLM)

### Step 1: Setup
1. Install `openai` package
2. Add API key configuration (environment variable or config file)
3. Create extraction function with structured output

### Step 2: Integration
1. Add extraction step to `search_engineering_management_roles()` function
2. Extract fields after match is found
3. Add extracted fields to match dictionary

### Step 3: Output Format
Add to each match:
```json
{
  "extracted": {
    "company_name": "NVIDIA",
    "role_name": "Engineering Manager, Deep Learning Inference",
    "is_remote": true,
    "location": "North America preferred",
    "employment_type": "Full-time",
    "cash_compensation": null,
    "equity_compensation": "equity"
  }
}
```

### Step 4: Error Handling
- Handle API failures gracefully
- Retry logic for transient errors
- Log extraction failures for manual review
- Continue processing even if some extractions fail

### Step 5: Performance
- Batch processing (process 10-20 posts per API call)
- Async/parallel requests
- Progress indicators
- Estimated time: ~30-60 seconds for 472 posts

---

## Alternative Models to Consider

### OpenAI Models
- **gpt-4o-mini** (Recommended) - $0.15/$0.60 per 1M tokens, fast, accurate
- **gpt-3.5-turbo** - Slightly cheaper but less accurate

### Other Options
- **Anthropic Claude Haiku** - Similar pricing, good accuracy
- **Local models** (Ollama, etc.) - Free but slower, may need fine-tuning

---

## Next Steps

1. **Choose an option** (recommend Option 1 or 4)
2. **Set up API key** and test with sample posts
3. **Implement extraction function**
4. **Integrate into existing pipeline**
5. **Test on full dataset**
6. **Review and refine extraction prompts**

Would you like me to implement one of these options?

