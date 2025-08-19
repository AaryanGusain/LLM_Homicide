import os
from dotenv import load_dotenv
from typing import TypedDict, List, Set, Any
import json
import re
from urllib.parse import urljoin
import pytesseract
from PIL import Image
import io

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

import requests
from bs4 import BeautifulSoup
import fitz 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
from langchain import hub

load_dotenv()

'''using a supervisor router agent architecture to manage the generalist and specialist agents, 
specialist agent is used for the specific task of finding homicide statistics, while
 the generalist agent is used for everything else.'''

class SupervisorState(TypedDict):
    input: str
    messages: List[str]
    next: str


def create_generalist_agent():
    #Builds a general-purpose agent with a web search tool
    

    prompt = hub.pull("hwchase17/react")
    
    tools = [TavilySearchResults(max_results=3)]
    # I'm using a cheaper model for this project, since I dont have access the resources for the larger models, namely gemma-3-27b-it ( basically free )
    llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", temperature=0)
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)


def create_specialist_agent():

    class SpecialistState(TypedDict):
        objective: str
        cities: List[str]
        current_city: str
        data_schema: str
        structured_results: List[dict]
        urls_to_try: List[str]
        tried_urls: Set[str]
        url_to_scrape: str
        raw_content: Any
        cleaned_content: str
        scrape_count: int
        findings: dict
        new_data_found: bool 
        final_report: str

    specialist_llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", temperature=0)

    def search_node(state: SpecialistState):
        city = state['current_city']
        objective = state['objective']
        print(f"\n--- [Specialist] Researching '{objective}' in {city}... ---")
        query = f"official government portal or data source for {objective} in {city}"
        search_tool = TavilySearchResults(max_results=5, search_type="web")
        results = search_tool.invoke({"query": query})
        print(f"--- [Specialist] Search complete. Found {len(results)} sources. ---")
        return {"structured_results": results, "scrape_count": 0, "urls_to_try": [], "tried_urls": set()}

    def select_urls_node(state: SpecialistState):
        print("--- [Specialist] Selecting promising URLs... ---")
        if not state['structured_results']: return {"urls_to_try": []}
        all_urls = [res.get('url', '') for res in state['structured_results'] if res.get('url')]
        wiki_urls = [url for url in all_urls if 'wikipedia.org' in url.lower()]
        other_urls = [url for url in all_urls if 'wikipedia.org' not in url.lower()]
        sorted_urls = wiki_urls + other_urls #not sure about wiki urls, was unable to determine if they actually proivde useful data, but kept them as high priority for now
        # --- DEBUG LOGGING ---
        print(f"--- [Specialist] Selected {len(sorted_urls)} URLs. ---")
        return {"urls_to_try": sorted_urls[:5]}

    def scrape_node(state: SpecialistState):
        print(f"\n--- DEBUG at start of scrape: Findings so far ---\n{state.get('findings', {})}\n----------------------------------\n")
        if not state['urls_to_try']: return {"raw_content": None, "scrape_count": state.get('scrape_count', 0)}
        urls_to_try = state['urls_to_try']
        url_to_scrape = urls_to_try.pop(0)
        tried_urls = state['tried_urls']
        tried_urls.add(url_to_scrape)
        scrape_count = state.get('scrape_count', 0) + 1
        print(f"--- [Specialist]  Scraping URL ({scrape_count}): {url_to_scrape} ---")
        content = None
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            if url_to_scrape.lower().endswith(('.csv', '.xlsx')):
                if url_to_scrape.lower().endswith('.csv'): df = pd.read_csv(url_to_scrape, on_bad_lines='skip')
                else: df = pd.read_excel(url_to_scrape)
                content = df.head(30).to_string()
            elif url_to_scrape.lower().endswith('.pdf'):
                response = requests.get(url_to_scrape, timeout=20, headers=headers)
                response.raise_for_status()
                content = response.content
            else:
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                driver = webdriver.Chrome(options=chrome_options)
                driver.get(url_to_scrape)
                time.sleep(3)
                content = driver.page_source
                driver.quit()
        except Exception as e:
            print(f"--- [Specialist] Error scraping {url_to_scrape}: {e} ---")
        return {"urls_to_try": urls_to_try, "tried_urls": tried_urls, "raw_content": content, "url_to_scrape": url_to_scrape, "scrape_count": scrape_count}

    def distill_content_node(state: SpecialistState):
        """
        Cleans raw content, with OCR capabilities for scanned PDFs.
        """
        print("--- [Specialist]  distilling content... ---")
        raw_content = state['raw_content']
        
        cleaned_text = ""

        if isinstance(raw_content, bytes): 
            try:
                print("--- [Specialist] Attempting OCR... ---") #had issues with reading data from PDFs, so decided to use OCR
                cleaned_text = "" # Reset text
                with fitz.open(stream=raw_content, filetype="pdf") as doc:
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_bytes))
                        # Use pytesseract to do OCR on the image
                        cleaned_text += pytesseract.image_to_string(image) + "\n"
            except Exception as e:
                print(f"--- [Specialist] Error processing PDF: {e} ---")

        #content distiller, cleaning HTML or removing boilerplate tags, and truncates text to x amount of chars rn, also removes whitespaces
        elif isinstance(raw_content, str) and raw_content:
            if "DataFrame" in str(type(raw_content)) or "              " in raw_content:
                cleaned_text = raw_content
            else:
                soup = BeautifulSoup(raw_content, 'html.parser')
                for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
                    tag.decompose()
                main_content_tag, max_text_len = None, 0
                for tag in soup.find_all(['div', 'article', 'section', 'main']):
                    text = tag.get_text(" ", strip=True)
                    if len(text) > max_text_len:
                        max_text_len, main_content_tag = len(text), tag
                if main_content_tag: cleaned_text = main_content_tag.get_text(" ", strip=True)
                else: cleaned_text = soup.body.get_text(" ", strip=True) if soup.body else ""

        final_text = " ".join(cleaned_text.split())[:20000] #changing this has an adverse effect on performance, ive kept it at 20000 characters for now


    def analyze_node(state: SpecialistState):
        print("--- [Specialist] Analyzing content... ---")
        if not state['cleaned_content']: 
            return {"new_data_found": False}
            
        prompt = f"""You are a data extraction model. Objective: "{state['objective']}". From the text for {state['current_city']}, extract data matching the schema: {state['data_schema']}. Return ONLY a JSON object. If no data, return {{}}. Scraped Text: --- {state['cleaned_content']} --- JSON Output: """
        response = specialist_llm.invoke(prompt)
        
        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            new_data = json.loads(clean_response)
            
            # If no new data is found, report failure
            if not new_data:
                print("--- [Specialist]  Analysis complete, no new data found. ---")
                return {"new_data_found": False}
            
            current_findings = state.get('findings', {})
            city = state['current_city']
            if city not in current_findings: current_findings[city] = {}
            current_findings[city].update(new_data)
            
            print(f"--- [Specialist] Found new raw data: {new_data} ---")
            # Report success
            return {"findings": current_findings, "new_data_found": True}

        except (json.JSONDecodeError, ValueError): 
            print("--- [Specialist] LLM analysis failed to return valid JSON. ---")
            return {"new_data_found": False}


    def pry_deeper_node(state: SpecialistState):
        #have reason to believe that this makes a difference in performance, however not entirely sure, need more resources to test this
        print("--- Found a data scent!!!! Prying deeper for more links ---")
        

        raw_content = state['raw_content']
        url = state['url_to_scrape']
        
        if not isinstance(raw_content, str) or not raw_content:
            return {} 

        soup = BeautifulSoup(raw_content, 'html.parser')
        
        links = []
        for a in soup.find_all('a', href=True):
            link_text = a.get_text(" ", strip=True)
            link_href = urljoin(url, a['href'])
            links.append({"text": link_text, "url": link_href})
        
        if not links:
            return {}

        links_for_prompt = "\n".join([f"Text: \"{l['text']}\", URL: \"{l['url']}\"" for l in links[:30]])

        prompt = f"""
        You are a research assistant. You just found some relevant data for the objective "{state['objective']}" on the page "{url}".
        Now, look at this list of all the hyperlinks found on that same page.
        Your task is to identify the top 2-3 links that are most likely to contain MORE of the same type of data, especially for different time periods (e.g., historical data, archives, other reports).

        Links Found:
        ---
        {links_for_prompt}
        ---

        Return your answer ONLY as a JSON object with a single key "urls", containing a list of the most promising URLs. If no other links seem promising, return an empty list.
        Example: {{"urls": ["https://url1.gov/historical_data.csv", "https://url2.org/archives/2022_stats.pdf"]}}
        """
        response = specialist_llm.invoke(prompt)
        
        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            new_urls = json.loads(clean_response).get("urls", [])
            if new_urls:
                new_urls_to_try = [u for u in new_urls if u not in state['tried_urls'] and u not in state['urls_to_try']]
                # Prepend these new, high-priority URLs to the list
                state['urls_to_try'] = new_urls_to_try + state['urls_to_try']
                print(f"--- Prying deeper found {len(new_urls_to_try)} new high-priority URLs. ---")
                return {"urls_to_try": state['urls_to_try']}
        except (json.JSONDecodeError, AttributeError):
            print("---LLM failed to identify new URLs to pry into. ---")
        
        return {}

    def master_router(state: SpecialistState):
        """
        This intelligent router first checks for success to decide if it should pry deeper.
        """
        # If the last analysis found new data, it's a hot lead
        if state.get("new_data_found"):
            return "pry_deeper"
        
        # Otherwise, continue with the normal loop logic
        if state['urls_to_try'] and state.get('scrape_count', 0) < 5:
            return "continue_scraping"
        
        return "synthesis"


    def synthesis_node(state: SpecialistState):
  
        city = state['current_city']
        print(f"--- [Specialist] Synthesizing all collected data for {city}... ---")

        raw_findings = state.get('findings', {}).get(city, {})
        
        # The Guardrail to skip LLM calls on totally empty data
        if not raw_findings or all(value is None for value in raw_findings.values()):
            print(f"--- [Specialist] No concrete data found for {city}. Skipping synthesis. ---")
            schema_keys = json.loads(state['data_schema']).keys()
            clean_data = {key: None for key in schema_keys}
            current_findings = state['findings']
            current_findings[city] = clean_data
            return {"findings": current_findings}

        raw_data_str = json.dumps(raw_findings, indent=2)

        # --- THE FINAL PROMPT ---
        prompt = f"""
        You are a senior data analyst responsible for synthesizing a final, clean report. A junior agent has collected the following raw data points for {city} related to the objective: "{state['objective']}".

        Your goal is to populate this exact JSON schema:
        {state['data_schema']}

        Here is the raw data collected by the junior agent. It may be messy, conflicting, or incomplete.
        Raw Data Collected:
        ---
        {raw_data_str}
        ---

        Synthesize the raw data into a clean JSON object that perfectly matches the schema.
        Follow these rules STRICTLY:
        1.  **Populate Every Key:** Your final response MUST contain every key defined in the schema.
        2.  **Use `null` for Missing Data:** If, after reviewing all the raw data, you cannot find a credible number for a specific key in the schema, you MUST use the value `null` for that key.
        3.  **Be Critical:** Reject partial data (like 'Year-to-Date'). If you see multiple numbers for one year, choose the most likely official figure.
        4.  **Output Format:** Your entire response must be ONLY the final JSON object and nothing else. Do not add explanations or surrounding text.
        """
        
        response = specialist_llm.invoke(prompt)
        
        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            clean_data = json.loads(clean_response)
            
            current_findings = state['findings']
            current_findings[city] = clean_data
            
            print(f"--- [Specialist] Synthesis complete: {clean_data} ---")
            return {"findings": current_findings}

        except (json.JSONDecodeError, ValueError):
            print(f"--- [Specialist] LLM synthesis failed to return valid JSON. ---")
            # As a fallback, return an empty dict so the graph doesn't crash
            return {}


    def next_city_node(state: SpecialistState):
        """
        Updates to the next city and RESETS the temporary state for the new research loop.
        """
        processed_cities = set(state.get('findings', {}).keys())
        all_cities = state['cities']

        for city in all_cities:
            if city not in processed_cities:
                print(f"--- [Specialist] Moving to next city: {city} ---")
                # full, clean reset for the next loop.
                return {
                    "current_city": city,
                    "urls_to_try": [],
                    "tried_urls": set(),
                    "scrape_count": 0,
                }
        return {}

    def should_move_to_next_city(state: SpecialistState):
        if state['current_city'] not in state.get('findings', {}): state.setdefault('findings', {})[state['current_city']] = {}
        if len(state.get('findings', {})) < len(state['cities']): return "move_to_next_city"
        else: return "compile_report"

    def compile_report_node(state: SpecialistState):
        print("\n--- [Specialist] Compiling final report... ---")
        findings = state.get('findings', {})
        report = f"Final Report for Objective: {state['objective']}\n"
        report += "="*40 + "\n\n"
        all_headers = set()
        for city_data in findings.values(): all_headers.update(city_data.keys())
        sorted_headers = sorted(list(all_headers), reverse=True)
        if not sorted_headers: return {"final_report": "No data could be extracted."}
        header_line = f"| {'City':<15} |" + " | ".join([f" {h:<15} " for h in sorted_headers]) + " |\n"
        separator_line = f"|{'':-<17}|" + "|".join(["-"*(len(h)+2) for h in sorted_headers]) + "|\n"
        report += header_line + separator_line
        for city in state['cities']:
            report += f"| {city:<15} |"
            city_data = findings.get(city, {})
            for header in sorted_headers:
                value = city_data.get(header)
                value_str = str(value) if value is not None else "Not Found"
                report += f" {value_str:<15} |"
            report += "\n"
        return {"final_report": report}
    
    workflow = StateGraph(SpecialistState)
    workflow.add_node("search", search_node)
    workflow.add_node("select_urls", select_urls_node)
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("distill_content", distill_content_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("pry_deeper", pry_deeper_node)
    workflow.add_node("next_city", next_city_node)
    workflow.add_node("compile_report", compile_report_node)
    workflow.set_entry_point("search")
    workflow.add_edge("search", "select_urls")
    workflow.add_edge("select_urls", "scrape")
    workflow.add_edge("scrape", "distill_content")
    workflow.add_edge("distill_content", "analyze")
    workflow.add_edge("next_city", "search")
    workflow.add_edge("pry_deeper", "scrape")
    workflow.add_edge("compile_report", END)
    workflow.add_conditional_edges(
    "analyze",
    master_router,
    {
        "pry_deeper": "pry_deeper", #prying deeper for more links
        "continue_scraping": "scrape",
        "synthesis": "synthesis"
    }
)
    workflow.add_conditional_edges("synthesis", should_move_to_next_city, {"move_to_next_city": "next_city", "compile_report": "compile_report"})
    return workflow.compile()

# --- 5. Define the SUPERVISOR and the main graph nodes ---
generalist_agent = create_generalist_agent()
specialist_agent = create_specialist_agent()

supervisor_prompt_template = """You are a supervisor in a multi-agent system. Your role is to analyze the user's request and route it to the appropriate agent. You have two agents available:
1. 'specialist': An expert in finding and compiling specific data points about the pages for homicide statistics for the cities of New York, New Orleans, and Los Angeles for each year for the past 5 years.
2. 'generalist': A general conversational AI for answering questions, performing web searches, and handling non-specialized tasks.
Based on the user's query below, determine which agent is best suited. Respond with ONLY the name of the chosen agent ('specialist' or 'generalist').
User Query: "{input}" """
supervisor_chain = PromptTemplate.from_template(supervisor_prompt_template) | ChatGoogleGenerativeAI(model="gemma-3-27b-it", temperature=0)

def supervisor_node(state: SupervisorState):
    print("--- SUPERVISOR: Analyzing query... ---")
    response = supervisor_chain.invoke({"input": state['input']})
    print(f"--- SUPERVISOR: Routing to '{response.content.strip()}' agent. ---")
    return {"next": response.content.strip()}


def specialist_node(state: SupervisorState):
    #hardcoded specialist agent for homicide statistics, since the task is very specific
    print("---  SPECIALIST AGENT: Starting work. ---")
    objective = "homicide totals for the last 5 full calendar years"
    current_year = 2024 
    years = [str(y) for y in range(current_year - 4, current_year + 1)]
    data_schema = json.dumps({year: "integer or null for the total homicide count" for year in years})
    
    cities_match = re.findall(r'New York|New Orleans|Los Angeles', state['input'], re.IGNORECASE)
    cities_to_research = list(set([city.title() for city in cities_match])) if cities_match else ["New York", "New Orleans", "Los Angeles"]
    
    if not cities_to_research:
        return {"messages": ["Could not identify any target cities in the request."]}


    initial_state = {
        "objective": objective,
        "data_schema": data_schema,
        "cities": cities_to_research,
        "current_city": cities_to_research[0],
        "findings": {},
        "urls_to_try": [],
        "tried_urls": set(),
        "scrape_count": 0,
        "raw_content": None,
        "cleaned_content": "",
        "structured_results": [],
        "url_to_scrape": ""
    }
    
    final_state = specialist_agent.invoke(initial_state, {"recursion_limit": 100})
    final_report = final_state.get('final_report', "Specialist agent failed to produce a report.")
    return {"messages": [final_report]}

def generalist_node(state: SupervisorState):
    print("--- GENERALIST AGENT: Starting work. ---")
    result = generalist_agent.invoke({"input": state['input']})
    return {"messages": [result['output']]}


workflow = StateGraph(SupervisorState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("specialist", specialist_node)
workflow.add_node("generalist", generalist_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda state: state["next"], {"specialist": "specialist", "generalist": "generalist"})
workflow.add_edge("specialist", END)
workflow.add_edge("generalist", END)
app = workflow.compile()

if __name__ == '__main__':
    while True:
        user_input = input("Hello! How can I help you today?\n> ")
        if user_input.lower() in ["quit", "exit"]: break
        final_state = app.invoke({"input": user_input, "messages": []})
        print("\n--- Final Answer ---")
        print(final_state['messages'][-1])
        print("\n" + "="*40 + "\n")