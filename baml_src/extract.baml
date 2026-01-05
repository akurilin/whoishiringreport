// Job extraction function
// Extracts structured job data from HN "Who is hiring?" comments

function ExtractJobData(comment_content: string) -> CommentExtraction {
  client GPT4oMini  // Default client, can be overridden at runtime
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
