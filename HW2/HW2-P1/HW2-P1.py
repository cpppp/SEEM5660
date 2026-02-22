import asyncio
import os
import json
import re
import datetime
from typing import Optional, List, Set, Tuple, Dict, Any
from pydantic import BaseModel
from markitdown import MarkItDown
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from rapidfuzz import fuzz

# Define the CVInfo model which is used to store the extracted information from the CV
class CVInfo(BaseModel):
    name: str
    title: Optional[str] = None
    location: Optional[str] = None
    hometown: Optional[str] = None
    current_company: Optional[str] = None
    education: List[dict] = []
    experience: List[dict] = []
    skills: List[str] = []

# Define the VerificationReport model which is used to store the verification results
class VerificationReport(BaseModel):
    cv_name: str
    cv_source: str
    linkedin_match: Optional[dict] = None
    facebook_match: Optional[dict] = None
    discrepancies: List[dict] = []
    verification_score: float = 0.0
    confidence: str = "low"
    match_details: dict = {}
    extracted_info: Optional[CVInfo] = None

# tokenize function used to tokenize the text
def tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return set(re.findall(r'\b[a-z0-9]+\b', text.lower()))

# jaccard_similarity calculate the Jaccard similarity between two strings
def jaccard_similarity(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    set1, set2 = tokenize(str1), tokenize(str2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# fuzzy_match_score calculate the fuzzy match score between two strings
def fuzzy_match_score(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    s1, s2 = str1.lower().strip(), str2.lower().strip()
    scores = [
        fuzz.ratio(s1, s2),
        fuzz.partial_ratio(s1, s2),
        fuzz.token_sort_ratio(s1, s2),
        fuzz.token_set_ratio(s1, s2),
    ]
    return max(scores) / 100.0

# combined_match calculate the combined match score between two strings
def combined_match(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    return 0.6 * fuzzy_match_score(str1, str2) + 0.4 * jaccard_similarity(str1, str2)

# parse_tool_result parse the tool results and return the JSON data
def parse_tool_result(results):
    try:
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and 'text' in first:
                try:
                    results = json.loads(first['text'])
                except:
                    pass
            elif isinstance(first, str):
                try:
                    results = json.loads(first)
                except:
                    pass
        if isinstance(results, str):
            results = json.loads(results)
    except:
        pass
    return results

# extract_cv_info extract the CV information using the LLM
async def extract_cv_info(llm, cv_text: str) -> CVInfo:
    prompt = f"""Extract information from this CV and return JSON only:

CV Text:
{cv_text}

Return JSON with: name, title, location, hometown, current_company, 
education (list with school/degree/field/year), experience (list with company/title/start_year/end_year), skills (list).

Important: location should be a single string, not a list.

JSON only, no other text."""

    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        content = resp.content
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        data = json.loads(content.strip())
        if isinstance(data.get('location'), list):
            data['location'] = ', '.join(data['location'])
        return CVInfo(**data)
    except Exception as e:
        print(f"    Warning: Failed to extract CV info: {e}")
        lines = cv_text.strip().split('\n')
        name = lines[0].strip() if lines else "Unknown"
        if len(name) < 2 or len(name) > 50:
            name = "Unknown"
        return CVInfo(name=name)

# detect_time_anomalies detect the time anomalies in the CV information
def detect_time_anomalies(cv_info: CVInfo) -> List[dict]:
    current_year = datetime.datetime.now().year
    anomalies = []
    
    for exp in cv_info.experience:
        start = exp.get('start_year')
        end = exp.get('end_year')
        
        if start:
            try:
                s = int(start) if str(start).isdigit() else None
                if s and s > current_year:
                    anomalies.append({
                        'type': 'future_start_date',
                        'severity': 'critical',
                        'description': f"Work starts in future: {start}"
                    })
            except:
                pass
        
        if end and end != 'Present':
            try:
                e = int(end) if str(end).isdigit() else None
                if e and e > current_year:
                    anomalies.append({
                        'type': 'future_end_date',
                        'severity': 'critical',
                        'description': f"Work ends in future: {end}"
                    })
            except:
                pass
    
    for i, exp1 in enumerate(cv_info.experience):
        for j, exp2 in enumerate(cv_info.experience):
            if i >= j:
                continue
            s1 = exp1.get('start_year')
            e1 = exp1.get('end_year')
            s2 = exp2.get('start_year')
            e2 = exp2.get('end_year')
            
            try:
                start1 = int(s1) if str(s1).isdigit() else None
                end1 = int(e1) if e1 and e1 != 'Present' and str(e1).isdigit() else current_year
                start2 = int(s2) if str(s2).isdigit() else None
                end2 = int(e2) if e2 and e2 != 'Present' and str(e2).isdigit() else current_year
                
                if start1 and start2:
                    if not (end1 < start2 or end2 < start1):
                        overlap = min(end1, end2) - max(start1, start2) + 1
                        if overlap > 1:
                            anomalies.append({
                                'type': 'overlapping_experience',
                                'severity': 'critical',
                                'description': f"Overlapping: {exp1.get('company')} ({s1}-{e1}) and {exp2.get('company')} ({s2}-{e2})"
                            })
            except:
                pass
    
    return anomalies

# search_linkedin search the LinkedIn people using the MCP tools
async def search_linkedin(mcp_tools, cv_info: CVInfo) -> Tuple[Optional[dict], dict]:
    search_tool = next((t for t in mcp_tools if t.name == 'search_linkedin_people'), None)
    if not search_tool:
        return None, {}
    
    results = parse_tool_result(await search_tool.ainvoke({'q': cv_info.name, 'limit': 20}))
    if not isinstance(results, list) or not results:
        return None, {}
    
    best_match, best_score, best_details = None, 0, {}
    
    for r in results:
        if not isinstance(r, dict) or 'id' not in r:
            continue
        
        name_score = combined_match(cv_info.name, r.get('name', ''))
        loc_score = combined_match(cv_info.location or '', r.get('location', ''))
        headline = r.get('headline', '')
        company = cv_info.current_company or ''
        if cv_info.experience:
            company = cv_info.experience[0].get('company', company)
        comp_score = combined_match(company, headline)
        
        total = name_score * 20 + loc_score * 8 + comp_score * 10
        
        if total > best_score:
            best_score = total
            best_match = r
            best_details = {'name': name_score, 'location': loc_score, 'company': comp_score, 'total': total}
    
    return best_match, best_details

# search_facebook search the Facebook users using the MCP tools
async def search_facebook(mcp_tools, cv_info: CVInfo) -> Tuple[Optional[dict], dict]:
    search_tool = next((t for t in mcp_tools if t.name == 'search_facebook_users'), None)
    if not search_tool:
        return None, {}
    
    results = parse_tool_result(await search_tool.ainvoke({'q': cv_info.name, 'limit': 20}))
    if not isinstance(results, list) or not results:
        return None, {}
    
    best_match, best_score, best_details = None, 0, {}
    
    for r in results:
        if not isinstance(r, dict) or 'id' not in r:
            continue
        
        fb_name = r.get('display_name', '') or r.get('original_name', '')
        name_score = combined_match(cv_info.name, fb_name)
        fb_loc = f"{r.get('city', '')} {r.get('country', '')}"
        loc_score = max(combined_match(cv_info.location or '', fb_loc), combined_match(cv_info.hometown or '', fb_loc))
        
        total = name_score * 20 + loc_score * 8
        
        if total > best_score:
            best_score = total
            best_match = r
            best_details = {'name': name_score, 'location': loc_score, 'total': total}
    
    return best_match, best_details

# get_profile get the profile using the MCP tools
async def get_profile(mcp_tools, tool_name, id_param, id_val):
    tool = next((t for t in mcp_tools if t.name == tool_name), None)
    if not tool:
        return None
    result = parse_tool_result(await tool.ainvoke({id_param: id_val}))
    if isinstance(result, list) and result:
        result = result[0]
    return result if isinstance(result, dict) and 'id' in result else None

# compare_profiles compare the CV information with the LinkedIn and Facebook profiles
def compare_profiles(cv_info: CVInfo, linkedin: Optional[dict], facebook: Optional[dict]) -> List[dict]:
    issues = []
    
    if linkedin:
        li_exp = linkedin.get('experience', [])
        li_current = [e for e in li_exp if isinstance(e, dict) and e.get('is_current')]
        
        cv_company = cv_info.current_company
        if cv_info.experience:
            cv_company = cv_info.experience[0].get('company', cv_company)
        
        if cv_company and li_current:
            li_company = li_current[0].get('company', '')
            if combined_match(cv_company, li_company) < 0.4:
                issues.append({
                    'type': 'company_mismatch_linkedin',
                    'severity': 'low',
                    'description': f"CV: {cv_company} vs LinkedIn: {li_company}"
                })
        
        li_edu = linkedin.get('education', [])
        cv_school = None
        if cv_info.education:
            cv_school = cv_info.education[0].get('school', '')
        
        if cv_school and li_edu:
            li_schools = [e.get('school', '') for e in li_edu if isinstance(e, dict)]
            match_found = any(combined_match(cv_school, s) > 0.4 for s in li_schools if s)
            if not match_found and li_schools:
                issues.append({
                    'type': 'education_mismatch_linkedin',
                    'severity': 'medium',
                    'description': f"CV school: {cv_school} not found in LinkedIn"
                })
    
    if facebook:
        cv_company = cv_info.current_company
        if cv_info.experience:
            cv_company = cv_info.experience[0].get('company', cv_company)
        
        fb_company = facebook.get('current_company', '')
        if cv_company and fb_company and combined_match(cv_company, fb_company) < 0.4:
            issues.append({
                'type': 'company_mismatch_facebook',
                'severity': 'medium',
                'description': f"CV: {cv_company} vs Facebook: {fb_company}"
            })
    
    return issues

# calculate_score calculate the verification score based on the discrepancies and profile details
def calculate_score(discrepancies: List[dict], li_details: dict, fb_details: dict) -> float:
    score = 0.5
    
    if li_details:
        li_total = li_details.get('total', 0)
        if li_total >= 30:
            score += 0.20
        elif li_total >= 25:
            score += 0.15
        elif li_total >= 20:
            score += 0.10
        elif li_total >= 15:
            score += 0.05
    
    if fb_details:
        fb_total = fb_details.get('total', 0)
        if fb_total >= 25:
            score += 0.15
        elif fb_total >= 20:
            score += 0.10
        elif fb_total >= 15:
            score += 0.05
    
    critical_count = sum(1 for d in discrepancies if d.get('severity') == 'critical')
    high_count = sum(1 for d in discrepancies if d.get('severity') == 'high')
    medium_count = sum(1 for d in discrepancies if d.get('severity') == 'medium')
    low_count = sum(1 for d in discrepancies if d.get('severity') == 'low')
    
    score -= critical_count * 0.20
    score -= high_count * 0.15
    score -= medium_count * 0.1
    score -= low_count * 0.05
    
    return max(0.05, min(1.0, score))

# verify_single_cv verify the single CV using the LLM and MCP tools
async def verify_single_cv(llm, mcp_tools, cv_text: str, cv_source: str) -> VerificationReport:
    cv_info = await extract_cv_info(llm, cv_text)
    print(f"  Extracted: {cv_info.name}")
    
    li_match, li_details = await search_linkedin(mcp_tools, cv_info)
    li_profile = None
    if li_match:
        li_id = li_match.get('id')
        if li_id:
            print(f"  LinkedIn: {li_match.get('name')} (score: {li_details.get('total', 0):.1f})")
            li_profile = await get_profile(mcp_tools, 'get_linkedin_profile', 'person_id', li_id)
    
    fb_match, fb_details = await search_facebook(mcp_tools, cv_info)
    fb_profile = None
    if fb_match:
        fb_id = fb_match.get('id')
        if fb_id:
            print(f"  Facebook: {fb_match.get('display_name')} (score: {fb_details.get('total', 0):.1f})")
            fb_profile = await get_profile(mcp_tools, 'get_facebook_profile', 'user_id', fb_id)
    
    discrepancies = detect_time_anomalies(cv_info)
    discrepancies.extend(compare_profiles(cv_info, li_profile, fb_profile))
    
    score = calculate_score(discrepancies, li_details, fb_details)
    confidence = "high" if score >= 0.6 else ("medium" if score >= 0.35 else "low")
    
    return VerificationReport(
        cv_name=cv_info.name,
        cv_source=cv_source,
        linkedin_match=li_profile,
        facebook_match=fb_profile,
        discrepancies=discrepancies,
        verification_score=score,
        confidence=confidence,
        match_details={'linkedin': li_details, 'facebook': fb_details},
        extracted_info=cv_info
    )

# verify_cv_list verify the list of CVs markdown using the LLM and MCP tools
async def verify_cv_list(llm, mcp_tools, cv_texts: List[str], sources: List[str] = None) -> Tuple[List[float], List[VerificationReport]]:
    if sources is None:
        sources = [f"input_{i+1}" for i in range(len(cv_texts))]
    
    scores, reports = [], []
    
    for cv_text, source in zip(cv_texts, sources):
        print(f"\n{'='*60}\nProcessing: {source}\n{'='*60}")
        report = await verify_single_cv(llm, mcp_tools, cv_text, source)
        reports.append(report)
        scores.append(report.verification_score)
        print(f"  Score: {report.verification_score:.2f} ({report.confidence})")
        for d in report.discrepancies:
            print(f"    ! {d['type']}: {d.get('description', '')}")
    
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Scores: {[round(s, 2) for s in scores]}")
    return scores, reports

# evaluate evaluate the verification scores using the groundtruth labels
def evaluate(scores, groundtruth, threshold=0.5):
    correct, decisions = 0, []
    for s, gt in zip(scores, groundtruth):
        pred = 1 if s > threshold else 0
        decisions.append(pred)
        if pred == gt:
            correct += 1
    return {"decisions": decisions, "correct": correct, "total": len(scores), "final_score": correct / len(scores)}

# define CVVerificationAgent class
class CVVerificationAgent:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = os.getenv("ARK_API_KEY")
        self.api_base = os.getenv("ARK_API_URL")
        self.model_name = os.getenv("ARK_API_MODEL", "deepseek-v3-2-251201")
        self.llm = None
        self.mcp_tools = None
        self._initialized = False

    async def initialize(self):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=self.model_name, api_key=self.api_key, base_url=self.api_base, temperature=0)
        client = MultiServerMCPClient({
            "social_graph": {"transport": "http", "url": "https://ftec5660.ngrok.app/mcp",
                            "headers": {"ngrok-skip-browser-warning": "true"}}
        })
        self.mcp_tools = await client.get_tools()
        self._initialized = True
        print("Agent initialized")

    async def verify_pdfs(self, cv_dir: str = "downloaded_cvs"):
        if not self._initialized:
            await self.initialize()
        
        md = MarkItDown(enable_plugins=False)
        pdfs = sorted([f for f in os.listdir(cv_dir) if f.endswith('.pdf')],
                     key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        texts, sources = [], []
        for pdf in pdfs:
            result = md.convert(os.path.join(cv_dir, pdf))
            texts.append(result.text_content)
            sources.append(pdf)
        
        return await verify_cv_list(self.llm, self.mcp_tools, texts, sources)

    async def verify_markdown_list(self, markdown_list: List[str], names: List[str] = None):
        """verify markdown list - main interface"""
        if not self._initialized:
            await self.initialize()
        return await verify_cv_list(self.llm, self.mcp_tools, markdown_list, names)

# main function
async def main():
    # generate the cvs markdown list
    from markitdown import MarkItDown
    md = MarkItDown(enable_plugins=False)
    cv_dir = "downloaded_cvs"
    pdfs = sorted([f for f in os.listdir(cv_dir) if f.endswith('.pdf')],
                key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    markdown_texts, pdf_names = [], []
    for pdf in pdfs:
        result = md.convert(os.path.join(cv_dir, pdf))
        markdown_texts.append(result.text_content)
        pdf_names.append(pdf)
    
    # create the CVVerificationAgent instance
    agent = CVVerificationAgent()
    
    # Here is the interface to verify the markdown texts
    scores, reports = await agent.verify_markdown_list(markdown_texts, pdf_names)
    
    # evaluate the verification scores
    print(f"\nFinal Scores: {[round(s, 2) for s in scores]}")
    result = evaluate(scores, [1, 1, 1, 0, 0])
    print(f"Evaluation: {result}")
    return scores, reports

# run the main function
if __name__ == "__main__":
    asyncio.run(main())
