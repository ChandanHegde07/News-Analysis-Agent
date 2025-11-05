import os
import re
import logging
from datetime import datetime
from typing import TypedDict, Optional
from dotenv import load_dotenv

# LangChain and LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# Web scraping
from bs4 import BeautifulSoup
import requests

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CONFIGURATION
class Config:
    """Configuration settings"""
    MAX_CONTENT_LENGTH = 5000
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 3
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# STATE DEFINITION
class NewsState(TypedDict):
    """State definition for the news analysis workflow"""
    urls: list
    scraped_content: dict
    analyses: list
    final_report: str
    errors: list

# PROMPTS
PROMPTS = {
    "analyze": """Analyze this news article and provide a structured analysis:

Article Content:
{content}

Please provide:
1. **Main Topic**: What is the primary subject of this article?
2. **Key Facts**: List 3-5 most important facts or findings
3. **Key Entities**: Identify important people, organizations, or technologies mentioned
4. **Summary**: Provide a concise 2-3 sentence summary

Format your response clearly with these sections.""",

    "report": """Create a comprehensive professional news analysis report based on these analyses:

=== Analysis 1 ===
{analysis1}

=== Analysis 2 ===
{analysis2}

Please create a cohesive report with:
1. Executive Summary (2-3 paragraphs)
2. Key Findings (bullet points)
3. Detailed Analysis (combining insights from both articles)
4. Implications and Trends
5. Conclusion

Use clear, professional language suitable for a business report."""
}

# ENVIRONMENT SETUP
def load_environment() -> str:
    """Load and validate environment variables"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please create a .env file with: GOOGLE_API_KEY=your_api_key"
        )
    logger.info("Environment loaded successfully")
    return api_key

# LLM INITIALIZATION
def initialize_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Initialize the language model"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=2048
        )
        logger.info("✓ LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

# TEXT CLEANING UTILITIES
def clean_text(text: str) -> str:
    """
    Remove markdown formatting and clean text
    
    Args:
        text: Raw text with potential markdown
        
    Returns:
        Cleaned text without markdown characters
    """
    if not text:
        return ""
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  
    text = re.sub(r'__(.+?)__', r'\1', text)      
    text = re.sub(r'\*(.+?)\*', r'\1', text)      
    text = re.sub(r'_(.+?)_', r'\1', text)        
    
    # Remove markdown lists
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Clean multiple spaces and newlines
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text.strip()


def extract_clean_content(html_content: str) -> str:
    """
    Extract and clean main content from HTML
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Cleaned text content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
        tag.decompose()
    
    main_content = None
    for selector in ['article', 'main', '[role="main"]', '.content', '.article-body']:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        text = soup.get_text(separator='\n', strip=True)
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    return text[:Config.MAX_CONTENT_LENGTH]

# WORKFLOW NODES
def scrape_node(state: NewsState) -> NewsState:
    """
    Node 1: Scrape content from URLs
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with scraped content
    """
    logger.info("[Node 1] Starting content scraping...")
    
    scraped = {}
    errors = state.get("errors", [])
    
    headers = {'User-Agent': Config.USER_AGENT}
    
    for url in state["urls"][:2]:  # Limit to first 2 URLs
        try:
            logger.info(f"Scraping: {url}")
            response = requests.get(
                url, 
                headers=headers, 
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            content = extract_clean_content(response.content)
            scraped[url] = content
            
            logger.info(f"Successfully scraped {len(content)} characters from {url[:50]}...")
            
        except requests.RequestException as e:
            error_msg = f"Request error for {url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            scraped[url] = ""
            
        except Exception as e:
            error_msg = f"Unexpected error scraping {url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            scraped[url] = ""
    
    return {**state, "scraped_content": scraped, "errors": errors}


def analyze_node(state: NewsState) -> NewsState:
    """
    Node 2: Analyze scraped content using LLM
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with analyses
    """
    logger.info("[Node 2] Starting content analysis...")
    
    analyses = []
    errors = state.get("errors", [])

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        llm = initialize_llm(api_key)
    except Exception as e:
        errors.append(f"LLM initialization failed: {str(e)}")
        return {**state, "analyses": analyses, "errors": errors}
    
    for url, content in state["scraped_content"].items():
        if not content:
            logger.warning(f"Skipping analysis for {url} (no content)")
            continue
        
        try:
            logger.info(f"Analyzing: {url[:50]}...")
            prompt = PROMPTS["analyze"].format(content=content)
            response = llm.invoke(prompt)
            analyses.append(response.content)
            logger.info(f"Analysis complete for {url[:50]}...")
            
        except Exception as e:
            error_msg = f"Analysis error for {url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

    logger.info(f"Completed {len(analyses)} analyses")
    return {**state, "analyses": analyses, "errors": errors}


def report_node(state: NewsState) -> NewsState:
    """
    Node 3: Generate final report from analyses
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with final report
    """
    logger.info("[Node 3] Generating final report...")
    
    errors = state.get("errors", [])
    
    if len(state["analyses"]) < 2:
        report = "Insufficient analyses to generate report. Need at least 2 successful analyses."
        logger.warning(report)
        return {**state, "final_report": report, "errors": errors}
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        llm = initialize_llm(api_key)
        
        prompt = PROMPTS["report"].format(
            analysis1=state["analyses"][0],
            analysis2=state["analyses"][1] if len(state["analyses"]) > 1 else state["analyses"][0]
        )
        
        response = llm.invoke(prompt)
        report = response.content
        logger.info("Report generated successfully")
        
    except Exception as e:
        error_msg = f"Report generation error: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        report = f"Error generating report: {str(e)}"
    
    return {**state, "final_report": report, "errors": errors}


# GRAPH CONSTRUCTION
def create_news_graph() -> StateGraph:
    """
    Build the LangGraph workflow
    
    Returns:
        Compiled workflow graph
    """
    # Log workflow construction start
    logger.info("Building workflow graph...")
    
    # Create state graph instance
    workflow = StateGraph(NewsState)
    
    # Add nodes to the workflow
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("report", report_node)
    
    # Define edges connecting the nodes
    workflow.add_edge(START, "scrape")
    workflow.add_edge("scrape", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    # Log successful graph construction
    logger.info("Workflow graph built successfully")
    return workflow.compile()

# PDF GENERATION
def parse_report_sections(report_text: str) -> list:
    """
    Parse report into structured sections
    
    Args:
        report_text: Raw report text
        
    Returns:
        List of dictionaries with header and content
    """
    sections = []
    current_section = None
    current_content = []
    
    for line in report_text.split('\n'):
        line = clean_text(line)
        
        if not line.strip():
            continue
        
        is_heading = (
            len(line) < 80 and 
            (line.endswith(':') or 
             line.isupper() or 
             any(word in line for word in [
                 'Executive Summary', 'Key Findings', 'Analysis', 
                 'Implications', 'Trends', 'Conclusion', 'Introduction'
             ]))
        )
        
        if is_heading:
            if current_section:
                sections.append({
                    'header': current_section,
                    'content': '\n\n'.join(current_content)
                })
            current_section = line
            current_content = []
        else:
            if current_section:
                current_content.append(line)
    
    if current_section:
        sections.append({
            'header': current_section,
            'content': '\n\n'.join(current_content)
        })
    
    return sections


def generate_pdf_report(report_content: str, urls: list, errors: list = None) -> Optional[str]:
    """
    Generate a professional PDF report
    
    Args:
        report_content: Report text content
        urls: List of source URLs
        errors: List of errors encountered (optional)
        
    Returns:
        Filename of generated PDF or None if failed
    """
    logger.info("[PDF] Generating professional PDF report...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"news_analysis_report_{timestamp}.pdf"
    
    try:
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        PRIMARY_COLOR = colors.HexColor('#1e3a8a')
        SECONDARY_COLOR = colors.HexColor('#3b82f6')
        ACCENT_COLOR = colors.HexColor('#60a5fa')
        DARK_TEXT = colors.HexColor('#1f2937')
        LIGHT_TEXT = colors.HexColor('#6b7280')
        ERROR_COLOR = colors.HexColor('#dc2626')
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=PRIMARY_COLOR,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=LIGHT_TEXT,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=PRIMARY_COLOR,
            spaceAfter=12,
            spaceBefore=16,
            fontName='Helvetica-Bold'
        )
        
        subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=styles['Heading3'],
            fontSize=13,
            textColor=SECONDARY_COLOR,
            spaceAfter=10,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            textColor=DARK_TEXT,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=15,
            fontName='Helvetica'
        )
        
        story.append(Paragraph("NEWS ANALYSIS REPORT", title_style))
        timestamp_str = datetime.now().strftime("%B %d, %Y at %H:%M")
        story.append(Paragraph(f"Generated on {timestamp_str}", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        
        meta_data = [
            ['Report ID:', timestamp],
            ['Sources Analyzed:', str(len(urls))],
            ['Analysis Type:', 'Comprehensive AI-Powered Analysis'],
            ['Model:', 'Google Gemini AI'],
        ]
        
        if errors:
            meta_data.append(['Errors Encountered:', str(len(errors))])
        
        meta_table = Table(meta_data, colWidths=[2*inch, 3.5*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), ACCENT_COLOR),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#f0f9ff')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
            ('TEXTCOLOR', (1, 0), (1, -1), DARK_TEXT),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, ACCENT_COLOR),
        ]))
        
        story.append(meta_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Sources section
        story.append(Paragraph("Sources Analyzed", header_style))
        
        source_data = [['#', 'Source URL']]
        for i, url in enumerate(urls, 1):
            source_data.append([str(i), url])
        
        source_table = Table(source_data, colWidths=[0.5*inch, 5.25*inch])
        source_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
            ('GRID', (0, 0), (-1, -1), 1, SECONDARY_COLOR),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [colors.white, colors.HexColor('#f8fafc')]),
        ]))
        
        story.append(source_table)
        story.append(Spacer(1, 0.3*inch))
        
        if errors:
            story.append(PageBreak())
            story.append(Paragraph("Errors and Warnings", header_style))
            
            error_data = [['Error Description']]
            for error in errors:
                error_data.append([error])
            
            error_table = Table(error_data, colWidths=[5.75*inch])
            error_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), ERROR_COLOR),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fee2e2')),
                ('GRID', (0, 0), (-1, -1), 1, ERROR_COLOR),
            ]))
            
            story.append(error_table)
            story.append(Spacer(1, 0.3*inch))
        
        story.append(PageBreak())
        story.append(Paragraph("Detailed Analysis", header_style))
        story.append(Spacer(1, 0.2*inch))
        
        sections = parse_report_sections(report_content)
        
        for section in sections:
            if section['header']:
                story.append(Paragraph(section['header'], subheader_style))
            
            paragraphs = section['content'].split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para:
                    clean_para = clean_text(para)
                    if clean_para:
                        story.append(Paragraph(clean_para, body_style))
            
            story.append(Spacer(1, 0.15*inch))
        
        story.append(Spacer(1, 0.4*inch))
        footer_text = (
            "Report generated by LangGraph News Analyzer | "
            "Powered by Google Gemini AI | "
            f"Copyright {datetime.now().year} | All rights reserved"
        )
        story.append(Paragraph(footer_text, subtitle_style))
        
        doc.build(story)
        logger.info(f"PDF Report saved: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

# MAIN EXECUTION
def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("NEWS ANALYZER POWERED BY LANGGRAPH & GOOGLE GEMINI AI")
    print("="*70 + "\n")
    
    try:
        api_key = load_environment()
        
        graph = create_news_graph()
        
        urls = [
            "https://www.theverge.com/ai-artificial-intelligence",
            "https://techcrunch.com/category/artificial-intelligence/",
        ]
        
        logger.info(f"Analyzing {len(urls)} sources...")
        
        initial_state = {
            "urls": urls,
            "scraped_content": {},
            "analyses": [],
            "final_report": "",
            "errors": []
        }
        
        logger.info("Starting workflow execution...")
        result = graph.invoke(initial_state)
        
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        print(result["final_report"])
        print("="*70 + "\n")
        
        if result.get("errors"):
            print("\nERRORS ENCOUNTERED:")
            for error in result["errors"]:
                print(f"  - {error}")
            print()
        
        pdf_file = generate_pdf_report(
            result["final_report"], 
            urls,
            result.get("errors")
        )
        
        if pdf_file:
            print(f"\nPDF Report Generated: {pdf_file}")
            print(f"Location: {os.path.abspath(pdf_file)}\n")
        else:
            print("\nPDF generation failed\n")
        
        print("="*70)
        print("Workflow completed successfully!")
        print(f"  - Sources analyzed: {len(result['scraped_content'])}")
        print(f"  - Successful analyses: {len(result['analyses'])}")
        print(f"  - Errors encountered: {len(result.get('errors', []))}")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())