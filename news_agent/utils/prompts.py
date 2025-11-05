"""
Prompts for the News Analysis Agent
"""

TRUSTWORTHINESS_PROMPT = """You are an expert fact-checker and media analyst. Evaluate the trustworthiness of this news article.

Consider the following factors:
- Source credibility and reputation
- Presence of citations and references
- Balanced presentation vs. bias
- Factual accuracy indicators
- Writing quality and professionalism

Article Details:
Title: {title}
Author(s): {authors}
Published: {publish_date}
Content Preview: {content_preview}

Provide your evaluation in JSON format:
{{
    "score": <number between 0-10>,
    "reasoning": "<brief explanation>",
    "red_flags": ["<flag1>", "<flag2>"],
    "strengths": ["<strength1>", "<strength2>"]
}}
"""

FACT_EXTRACTION_PROMPT = """You are a professional news analyst. Extract the most important facts from this article.

Focus on:
- Key events and developments
- Important statistics and data points
- Named entities (people, organizations, locations)
- Dates and timelines
- Cause-and-effect relationships

Article:
Title: {title}
Content: {content}

Return 3-5 key facts as a numbered list. Each fact should be concise and verifiable.
"""

REPORT_GENERATION_PROMPT = """You are a senior news editor creating a comprehensive analysis report.

Based on the extracted facts from multiple news sources, create a professional report with the following structure:

## Executive Summary
A 2-3 sentence overview of the main findings.

## Key Findings
The most important facts and developments, organized by theme.

## Source Analysis
Brief assessment of source reliability and coverage patterns.

## Trend Analysis
Identify emerging patterns or trends across the sources.

## Conclusion
Final takeaways and implications.

Facts from analyzed articles:
{facts}

Trustworthiness scores:
{scores}

Create a well-structured, objective report.
"""

CONTENT_CLEANING_PROMPT = """Clean and standardize this news article content.

Remove:
- Advertising text
- Navigation elements
- Subscription prompts
- Unrelated side content

Keep:
- Main article text
- Important quotes
- Relevant context

Original content:
{content}

Return only the cleaned article text.
"""
