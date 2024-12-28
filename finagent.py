import os
from typing import Dict, List, Optional, TypedDict, Literal, Union, Annotated
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import json
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import sqlite3
import re
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.tools import Tool
from functools import lru_cache
import hashlib
import pickle

# Initialize memory for state management
memory = {}  # Using a simple dictionary for in-memory storage

# Part 2: Type Definitions and Base Classes
class QueryType(Enum):
    DIRECT_SQL = "direct_sql"  # For direct SQL queries
    ANALYSIS = "analysis"      # For complex analysis requiring multiple queries

@dataclass
class QueryClassification:
    type: QueryType
    explanation: str
    raw_response: str
    confidence: float = 1.0

class AnalysisState(TypedDict):
    user_query: str              # The original user question
    query_classification: Dict    # How the query should be processed
    decomposed_questions: List[str]  # Breaking complex queries into parts
    sql_results: Dict            # Results from SQL queries
    analysis: str                # Analysis of the results
    final_output: Dict          # Final formatted output
    processing_time: float       # Time taken to process
    agent_states: Dict          # State tracking for agents
    raw_responses: Dict         # Raw responses from agents
    messages: List[AnyMessage]  # Conversation history

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

@dataclass
class Config:
    db_path: str = "final_working_database.db"
    sqlite_path: str = "sqlite:///final_working_database.db"
    model_name: str = "claude-3-sonnet-20240229"
    api_key: str = ""  # Anthropic API key
    cache_enabled: bool = True
    cache_dir: str = ".cache"
    cache_ttl: int = 86400  # Cache TTL in seconds (24 hours)

# Part 3: Prompt Templates
SYSTEM_SQL_PROMPT = """You MUST return ONLY a SQL query between ```sql tags. Nothing else.
The query should follow this format:

WITH metrics AS (
    SELECT 'Metric Name' as metric_name, calculated_value as value
    FROM relevant_tables
    -- Use appropriate joins if needed
)
SELECT metric_name, value FROM metrics;

Do not provide any explanations or analysis - just the SQL query."""

METRIC_CALCULATION_PROMPT = """Write a SQL query that calculates specific metrics for this question. 
Use CTEs and subqueries for complex calculations. 
Return ONLY calculated values, not raw data. 
Question: {question}"""

ANALYSIS_PROMPT = """Analyze these calculated metrics and provide insights:

Question: {question}
Calculated Metrics: {metrics}

Please provide:
1. Interpretation of each metric
2. Notable patterns or trends
3. Business implications
4. Potential areas for improvement"""

SYNTHESIS_PROMPT = """Synthesize the following analysis results into a coherent summary:
{results}

Provide:
1. Overall findings
2. Connections between different questions
3. Key insights from the combined analysis"""

# Part 4: Main DatabaseAnalyst Class
class DatabaseAnalyst:
    def __init__(self, config: Config):
        self.llm = ChatAnthropic(
            model=config.model_name,
            temperature=0,
            api_key=config.api_key
        )
        
        self.db_connection = sqlite3.connect(config.db_path)
        self.db = SQLDatabase.from_uri(
            config.sqlite_path,
            sample_rows_in_table_info=0,
            view_support=False,
            indexes_in_table_info=False
        )
        
        self.sql_toolkit = SQLDatabaseToolkit(
            db=self.db,
            llm=self.llm
        )
        
        self.sql_agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.sql_toolkit,
            verbose=False,
            agent_type="zero-shot-react-description",
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": SYSTEM_SQL_PROMPT
            }
        )
        
        # Initialize both query and prompt caches
        self.cache_file = ".query_cache.pkl"
        self.prompt_cache_file = ".prompt_cache.pkl"
        self.load_caches()

    def load_caches(self):
        """Load both caches from files"""
        try:
            with open(self.cache_file, 'rb') as f:
                self.query_cache = pickle.load(f)
        except FileNotFoundError:
            self.query_cache = {}
            
        try:
            with open(self.prompt_cache_file, 'rb') as f:
                self.prompt_cache = pickle.load(f)
        except FileNotFoundError:
            self.prompt_cache = {}

    def save_caches(self):
        """Save both caches to files"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.query_cache, f)
        with open(self.prompt_cache_file, 'wb') as f:
            pickle.dump(self.prompt_cache, f)

    def get_cache_key(self, text: str) -> str:
        """Generate cache key"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def get_cached_prompt_response(self, prompt: str) -> Optional[str]:
        """Get cached prompt response if exists"""
        cache_key = self.get_cache_key(prompt)
        return self.prompt_cache.get(cache_key)

    def cache_prompt_response(self, prompt: str, response: str):
        """Cache prompt response"""
        cache_key = self.get_cache_key(prompt)
        self.prompt_cache[cache_key] = response
        self.save_caches()

    def process_query(self, query: str) -> Dict:
        cache_key = self.get_cache_key(query)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        try:
            prompt = METRIC_CALCULATION_PROMPT.format(question=query)
            cached_response = self.get_cached_prompt_response(prompt)
            
            try:
                agent_response = self.sql_agent.invoke({
                    "input": prompt
                }, config={"handle_parsing_errors": False})
                response_text = str(agent_response)
            except Exception as e:
                error_text = str(e)
                # If this contains an analysis, capture it as a successful response
                if "Could not parse LLM output: `" in error_text:
                    analysis = error_text.split("Could not parse LLM output: `")[1].rsplit("`", 1)[0]
                    result = {
                        "success": True,
                        "metrics": {"analysis": analysis},
                        "type": "analysis"
                    }
                    self.query_cache[cache_key] = result
                    self.save_caches()
                    return result
                return {"success": False, "error": str(e)}

            # Continue with normal SQL processing if no exception
            sql_query = self._extract_sql(response_text)
            if not sql_query:
                return {"success": False, "error": "No valid SQL query was generated"}
            
            result = self._execute_sql_safely(sql_query)
            if result["success"]:
                self.query_cache[cache_key] = result
                self.save_caches()
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Query processing error: {str(e)}"}

    def _extract_sql(self, text: str) -> str:
        patterns = [
            r"```sql\n(.*?)\n```",     # Standard markdown SQL blocks
            r"```(.*?)```",            # Generic code blocks
            r"SELECT[\s\S]*?;",        # Direct SQL statements (multiline)
            r"WITH[\s\S]*?;",          # CTE-style queries (multiline)
            r"(?:SELECT|WITH).*?;"     # Any SQL starting with SELECT or WITH
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1).strip() if pattern.startswith('```') else match.group(0)
                # Basic validation that it's actually SQL
                if any(keyword in sql.upper() for keyword in ['SELECT', 'FROM']):
                    return sql
        return ""

    def _execute_sql_safely(self, query: str) -> Dict:
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query)
            
            metrics = {}
            for row in cursor.fetchall():
                metrics[str(row[0])] = row[1] if len(row) > 1 else row[0]
            
            return {"metrics": metrics, "success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze(self, query: str) -> Dict:
        if query in self.query_cache:
            return self.query_cache[query]
        
        state = AnalysisState(user_query=query)
        result = self.process_query(state)
        self.query_cache[query] = result
        return result

def format_output(results: Dict) -> str:
    output = []
    output.append("=== Database Analysis Results ===")
    output.append(f"\nQuery: {results.get('user_query', 'N/A')}")
    
    if "error" in results:
        output.append(f"\nError: {results['error']}")
        return "\n".join(output)
    
    if results.get('query_type') == 'direct_sql':
        output.append(f"\nSQL Query: {results.get('sql_query', 'N/A')}")
        output.append("\nResults:")
        if isinstance(results.get('results'), list):
            df = pd.DataFrame(results['results'])
            output.append(str(df))
        else:
            output.append(str(results.get('results', 'No results available')))
    else:
        output.append("\nAnalysis:")
        if isinstance(results.get('sql_results'), dict):
            output.append(f"\nSQL Results: {results['sql_results'].get('result', 'No results available')}")
        else:
            output.append(f"\nSQL Results: {str(results.get('sql_results', 'No results available'))}")
        
        output.append("\nDetailed Analysis:")
        output.append(str(results.get('analysis', 'No analysis available')))
    
    return "\n".join(output)

def analyze_query(query: str) -> str:
    try:
        config = Config()
        analyst = DatabaseAnalyst(config)
        results = analyst.analyze(query)
        
        if results and "error" not in results:
            formatted_output = format_output(results)
            
            # Save the analysis to a JSON file
            output_data = {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": formatted_output
            }
            
            # Create a safe filename from the query
            filename = f"{hashlib.md5(query.encode()).hexdigest()[:10]}_analysis.json"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
            except Exception as save_error:
                return f"{formatted_output}\n\nWarning: Could not save results to file: {str(save_error)}"
                
            return formatted_output + f"\n\nDetailed results saved to {filename}"
        else:
            return f"Error: {results.get('error', 'Unknown error occurred')}"
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Streamlit UI
def main():
    st.title("Database Analysis Assistant")
    
    # Initialize session state for results if it doesn't exist
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Enter your Anthropic API Key:", type="password")
    
    config = Config(api_key=api_key)
    
    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar to proceed.")
        return
        
    st.write("""
    Welcome to the Database Analysis Assistant! 
    Select the number of questions you'd like to ask and enter them in the boxes below.
    """)
    
    # Number selector for questions
    num_questions = st.number_input("How many questions would you like to ask?", 
                                  min_value=1, 
                                  max_value=10, 
                                  value=1)
    
    # Create list to store queries
    queries = []
    
    # Create text input boxes based on number selected
    for i in range(int(num_questions)):
        query = st.text_input(f"Question {i+1}:", key=f"query_{i}")
        if query:  # Only add non-empty queries
            queries.append(query)
    
    if st.button("Analyze"):
        if queries:
            # Clear previous results when running new analysis
            st.session_state.analysis_results = {}
            
            for query in queries:
                if query.strip():  # Process only non-empty queries
                    with st.spinner(f"Processing: {query}"):
                        analyst = DatabaseAnalyst(config)
                        result = analyst.process_query(query)
                        # Store result in session state
                        st.session_state.analysis_results[query] = result

    # Display results from session state
    for query, result in st.session_state.analysis_results.items():
        st.write(f"### Results for: {query}")
        if result["success"]:
            # Show Results
            st.write("#### Results:")
            st.write(result["metrics"])
            
            # Create unique key for download button
            download_key = f"download_{hashlib.md5(query.encode()).hexdigest()}"
            
            # Prepare download data
            json_str = json.dumps({
                "query": query,
                "results": result["metrics"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, indent=2)
            safe_filename = f"analysis_{hashlib.md5(query.encode()).hexdigest()[:8]}.json"
            
            # Download button with unique key
            st.download_button(
                label=f"Download Results for: {query[:30]}...",
                data=json_str,
                file_name=safe_filename,
                mime="application/json",
                key=download_key
            )
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()