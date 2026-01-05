# Optimizations

## BAML Prompt Simplification (2026-01-05)

### Summary

Reduced BAML extraction prompt from ~40 lines of detailed instructions to ~4 lines by leveraging BAML's `@description` annotations in the schema.

### Before

```baml
function ExtractJobData(comment_content: string) -> CommentExtraction {
  client GPT4oMini
  prompt #"
    You are a job posting data extractor for Hacker News "Who is hiring?" comments.

    CRITICAL RULES:
    1. One comment may contain MULTIPLE job roles. Extract ALL distinct roles mentioned.
    2. If a role is not a job posting (e.g., a reply, question, or off-topic comment), set is_job_posting=false and return empty roles.
    3. For multi-role comments, each role should have the SAME company info but may have different titles/levels.

    EXAMPLES OF MULTIPLE ROLES:
    - "We're hiring: Senior Backend Engineer, Staff Frontend Engineer, Engineering Manager" -> 3 roles
    - "Looking for: Go developers (junior and senior levels)" -> 2 roles (Junior and Senior)
    - Bullet-pointed or dash-listed positions -> one role per bullet/dash

    EXTRACTION RULES:
    - salary_min/salary_max: Parse from ranges. "$150k-$250k" -> 150000, 250000. "$170-225K" -> 170000, 225000.
    - salary_currency: Default to "USD" for $ amounts unless explicitly specified otherwise.
    - is_remote: True if "remote", "remote-first", "remote-friendly", "WFH", "work from home". False if "onsite only", "in-person required".
    - remote_regions: Extract geographic restrictions like "US only", "North America", "EU timezone", "EMEA".
    - employment_type: Normalize to enum values. "Full Time" -> "Full-time", "FT" -> "Full-time".
    - company_stage: Look for "Series A/B/C", "Seed", "bootstrapped", "public company", funding amounts.
    - application_method: Extract email addresses or URLs mentioned for applying.

    ROLE TITLE RULES:
    - Use the exact title mentioned when possible
    - If multiple seniority levels for same role, create separate entries: "Senior SWE" and "Staff SWE"
    - "SWE" = "Software Engineer", "MLE" = "Machine Learning Engineer"

    OUTPUT QUALITY:
    - If information is not mentioned, use null (not empty string)
    - Set extraction_confidence based on how clear the posting is
    - is_job_posting should be false for: comments asking questions, replies to other posts, meta-discussion

    ---

    Extract job data from this HN comment:

    {{ comment_content }}

    {{ ctx.output_format }}
  "#
}
```

### After

```baml
function ExtractJobData(comment_content: string) -> CommentExtraction {
  client GPT4oMini
  prompt #"
    Extract job posting data from this Hacker News "Who is hiring?" comment.

    IMPORTANT:
    - One comment may contain MULTIPLE job roles. Extract each distinct role separately.
    - If not a job posting (reply, question, off-topic), set is_job_posting=false with empty roles.

    {{ ctx.output_format }}

    Comment:
    {{ comment_content }}
  "#
}
```

### Why It Works

BAML's `{{ ctx.output_format }}` macro renders the full schema including all `@description` annotations. The extraction rules were redundant because they duplicated information already in `types.baml`:

| Prompt Instruction | Already in Schema |
|---|---|
| `$150k-$250k -> 150000, 250000` | `salary_min int? @description("Minimum salary as integer (e.g., 150000 for $150k)")` |
| `Default to USD for $ amounts` | `salary_currency string? @description("Currency code... Default to USD for $ amounts.")` |
| Employment type normalization | `@alias("Full-time")` on enum values |
| Company stage detection | `Bootstrapped @description("Self-funded, no external investment")` |

The only unique value the detailed prompt provided was behavioral instructions (multi-role extraction, non-job detection), which are retained in the simplified version.

### Results

| Model | Before (tokens) | After (tokens) | Reduction |
|-------|-----------------|----------------|-----------|
| gpt-4o-mini | 15,185 | 11,089 | **-27%** |
| gemini-2.0-flash-lite | 17,802 | 13,273 | **-25%** |
| gemini-2.5-flash-lite | 17,846 | 13,230 | **-26%** |

Pass rates remained equivalent (32/33 for both, with the single failure being LLM non-determinism on different test cases each run).

### Takeaway

When using BAML, prefer putting extraction guidance in `@description` annotations on your schema rather than in the prompt. This:

1. Keeps extraction rules co-located with the fields they describe
2. Reduces prompt token usage (~26% in this case)
3. Makes the schema self-documenting
4. Avoids duplication between prompt and schema
