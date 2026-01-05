// Job extraction data types
// Mirrors the Pydantic models in extract_jobs.py

enum EmploymentType {
  FullTime @alias("Full-time") @description("Standard full-time employment")
  PartTime @alias("Part-time") @description("Part-time employment")
  Contract @description("Contract or consulting work")
  Internship @description("Internship position")
  Fractional @description("Fractional or part-time leadership role")
}

enum CompanyStage {
  PreSeed @alias("Pre-seed") @description("Pre-seed stage startup")
  Seed @description("Seed stage startup")
  SeriesA @alias("Series A") @description("Series A funded")
  SeriesB @alias("Series B") @description("Series B funded")
  SeriesC @alias("Series C") @description("Series C funded")
  SeriesDPlus @alias("Series D+") @description("Series D or later")
  Public @description("Publicly traded company")
  Bootstrapped @description("Self-funded, no external investment")
}

class ExtractedRole {
  // Role details
  role_title string? @description("The job title, e.g., 'Senior Backend Engineer', 'Engineering Manager'")
  locations string[] @description("Physical locations for this role, e.g., ['San Francisco', 'NYC', 'Remote']")
  is_remote bool? @description("Whether fully remote work is available")
  remote_regions string[] @description("Geographic restrictions for remote, e.g., ['US only', 'North America', 'EU']")
  employment_type EmploymentType? @description("Type of employment")

  // Compensation
  salary_min int? @description("Minimum salary as integer (e.g., 150000 for $150k)")
  salary_max int? @description("Maximum salary as integer (e.g., 250000 for $250k)")
  salary_currency string? @description("Currency code, e.g., 'USD', 'EUR', 'GBP'. Default to USD for $ amounts.")
  salary_raw string? @description("Original salary text as written, e.g., '$150k-$250k'")
  equity string? @description("Equity amount as percentage, e.g., '0.5%', '1-2%'. Null if not specified.")

  // Application
  application_method string? @description("How to apply: email address, URL, or instructions")

  // Company details (denormalized per role)
  company_name string? @description("Name of the hiring company")
  company_stage CompanyStage? @description("Funding stage of the company")
  company_url string? @description("Company website URL if mentioned")
}

class CommentExtraction {
  roles ExtractedRole[] @description("List of job roles extracted from this comment. One comment can have multiple roles.")
  is_job_posting bool @description("Whether this comment is actually a job posting (vs noise/replies/off-topic)")
  extraction_confidence "high" | "medium" | "low" @description("Confidence in the extraction quality")
}
