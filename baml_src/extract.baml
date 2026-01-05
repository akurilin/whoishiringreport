// Job extraction function
// Extracts structured job data from HN "Who is hiring?" comments

function ExtractJobData(comment_content: string) -> CommentExtraction {
  client GPT4oMini  // Default client, can be overridden at runtime
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
