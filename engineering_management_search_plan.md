# Engineering Management Role Search Plan

## Objective
Identify job postings for upper/middle management engineering positions (Head of Engineering, VP of Engineering, Director of Engineering, etc.) from comments in `comments.json`, accounting for various phrasings, capitalizations, abbreviations, and terminology.

## Challenges
1. **Capitalization variations**: Director, director, DIRECTOR, DER
2. **Abbreviations**: ENG, Eng, engineering, DER, VPE, etc.
3. **Terminology variations**: engineering, software development, programming, software engineering, tech
4. **Phrasing variations**: "Head of Engineering" vs "Engineering Head" vs "Head, Engineering"
5. **HTML entities**: Content may contain HTML-encoded characters (e.g., `&#x2F;`, `&amp;`)
6. **Context**: Need to distinguish actual job titles from mentions in other contexts

## Search Strategy

### Phase 1: Pattern-Based Search (Primary Approach)

#### 1.1 Title Pattern Matching
Use regex patterns to match common title structures:

**Pattern Group A: [Title] of [Engineering]**
- `(head|director|vp|vpe|v\.?p\.?|vice\s+president)\s+of\s+(eng|engineering|software\s+engineering|software\s+development|dev|development|tech|technology)`
- Case-insensitive matching
- Handles abbreviations and full words

**Pattern Group B: [Engineering] [Title]**
- `(eng|engineering|software\s+engineering|software\s+development|dev|development|tech|technology)\s+(head|director|vp|vpe|v\.?p\.?|vice\s+president|manager|lead)`
- Matches "Engineering Director", "Software Engineering Manager", etc.

**Pattern Group C: [Title], [Engineering]**
- `(head|director|vp|vpe|v\.?p\.?|vice\s+president|manager),\s*(eng|engineering|software\s+engineering|software\s+development|dev|development|tech|technology)`
- Matches "Director, Engineering" format

**Pattern Group D: Abbreviated Forms**
- `(head|dir|der|vp|vpe)\s+(of\s+)?(eng|eng'g|engr)`
- `(eng|eng'g|engr)\s+(head|dir|der|vp|vpe|mgr)`
- Handles very abbreviated forms

**Pattern Group E: Management-Specific Terms**
- `(engineering|software|dev|development)\s+(manager|director|head|lead|vp|vpe)`
- `(senior|principal)\s+(engineering|software|dev)\s+(manager|director|lead)`
- Catches "Senior Engineering Manager", "Principal Software Engineer" (if management track)

#### 1.2 Engineering Term Variations
Create a comprehensive list of engineering-related terms:
- `engineering|eng|eng'g|engr|engg`
- `software\s+engineering|software\s+development|software\s+dev`
- `development|dev`
- `programming`
- `tech|technology`
- `platform\s+engineering|infrastructure\s+engineering`
- `product\s+engineering`

#### 1.3 Title Term Variations
Create a comprehensive list of management title terms:
- `head|head\s+of`
- `director|dir|der`
- `vp|vpe|v\.?p\.?|vice\s+president`
- `manager|mgr`
- `lead|leader` (with context - only if clearly management)
- `chief\s+technology\s+officer|cto` (optional - may be too senior)

### Phase 2: HTML Entity Handling

Before pattern matching:
1. Decode HTML entities in comment content
2. Common entities to handle:
   - `&#x2F;` → `/`
   - `&amp;` → `&`
   - `&lt;` → `<`
   - `&gt;` → `>`
   - `&#x27;` → `'`
   - `&quot;` → `"`

### Phase 3: Context Filtering

To reduce false positives:
1. **Position-based filtering**: Titles often appear near the beginning of job postings
2. **Keyword context**: Look for job posting indicators:
   - "We're hiring", "Looking for", "Seeking", "Join us"
   - "Full-time", "Remote", "Onsite"
   - Salary ranges, equity mentions
3. **Exclude patterns**: 
   - Skip if title appears in "reporting to" context
   - Skip if it's clearly a non-management role (e.g., "Senior Engineer reporting to Director")

### Phase 4: Scoring and Ranking

Assign confidence scores based on:
- **High confidence**: Exact matches like "Head of Engineering", "VP of Engineering"
- **Medium confidence**: Abbreviated forms, alternative phrasings
- **Lower confidence**: Single-word matches that could be ambiguous

### Phase 5: Manual Review Candidates

Flag for manual review:
- Matches with low confidence scores
- Unusual phrasings that might be false positives
- Edge cases that don't fit standard patterns

## Implementation Approach

### Step 1: Create a Python script with:
1. HTML entity decoder
2. Multiple regex patterns (compiled for efficiency)
3. Pattern matching function that tries all patterns
4. Context analysis function
5. Scoring/ranking system
6. Output to JSON/CSV with matched text, confidence, and context

### Step 2: Test on sample data
- Run on a subset of comments
- Review results manually
- Refine patterns based on false positives/negatives

### Step 3: Full scan
- Process all comments
- Generate report with matches
- Include surrounding context for each match

### Step 4: Iterative refinement
- Review results
- Add patterns for missed variations
- Remove patterns causing too many false positives

## Pattern Examples (Python Regex)

```python
import re
import html

# Decode HTML entities
def decode_html(text):
    return html.unescape(text)

# Pattern Group A: [Title] of [Engineering]
pattern_a = re.compile(
    r'\b(head|director|dir|der|vp|vpe|v\.?p\.?|vice\s+president|manager|mgr)\s+of\s+(eng|eng\'?g|engr|engineering|software\s+engineering|software\s+development|dev|development|tech|technology|programming)\b',
    re.IGNORECASE
)

# Pattern Group B: [Engineering] [Title]
pattern_b = re.compile(
    r'\b(eng|eng\'?g|engr|engineering|software\s+engineering|software\s+development|dev|development|tech|technology|programming)\s+(head|director|dir|der|vp|vpe|v\.?p\.?|vice\s+president|manager|mgr|lead|leader)\b',
    re.IGNORECASE
)

# Pattern Group C: [Title], [Engineering]
pattern_c = re.compile(
    r'\b(head|director|dir|der|vp|vpe|v\.?p\.?|vice\s+president|manager|mgr),\s*(eng|eng\'?g|engr|engineering|software\s+engineering|software\s+development|dev|development|tech|technology)\b',
    re.IGNORECASE
)

# Pattern Group D: Abbreviated forms
pattern_d = re.compile(
    r'\b(head|dir|der|vp|vpe)\s+(of\s+)?(eng|eng\'?g|engr)\b',
    re.IGNORECASE
)

# Pattern Group E: Senior/Principal with management
pattern_e = re.compile(
    r'\b(senior|principal)\s+(engineering|software|dev|development)\s+(manager|director|head|lead|vp|vpe)\b',
    re.IGNORECASE
)
```

## Output Format

For each match, capture:
- Comment ID/index
- Post URL
- Commenter
- Date
- Matched text
- Pattern that matched
- Confidence score
- Surrounding context (50-100 chars before/after)
- Full comment content (for manual review)

## Success Metrics

- **Recall**: Catch as many true positives as possible (aim for >90%)
- **Precision**: Minimize false positives (aim for >80%)
- **Coverage**: Handle at least 95% of common phrasings

## Next Steps

1. Implement the search script
2. Test on sample data
3. Refine patterns based on results
4. Generate final report of matches


