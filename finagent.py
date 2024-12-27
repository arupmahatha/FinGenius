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

# Part 3: Prompt Templates
QUERY_CLASSIFIER_PROMPT = """Classify if this query needs:
1. direct_sql: Simple SQL query
2. analysis: Complex analysis with multiple queries
Return JSON: {"type": "direct_sql"|"analysis", "confidence": 0-1}"""

SQL_AGENT_PROMPT = """Create SQL queries for the given questions. 
Include: thought process, SQL query, and result interpretation."""

ANALYST_PROMPT = """Analyze the SQL results and provide key insights."""

# Part 4: Main DatabaseAnalyst Class
class DatabaseAnalyst:
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatAnthropic(
            model=config.model_name,
            temperature=0,
            api_key=config.api_key
        )
        
        # Create database and toolkit once
        self.db = SQLDatabase.from_uri(config.sqlite_path)
        self.sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Use the existing toolkit
        self.sql_agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.sql_toolkit,
            verbose=True,
            agent_type="zero-shot-react-description",
            handle_parsing_errors=True
        )
        
        self.workflow = self._setup_workflow()
        self.query_cache = {}

    def _setup_workflow(self) -> StateGraph:
        workflow = StateGraph(AnalysisState)

        # Use the existing toolkit instead of creating new ones
        def process_query(state: AnalysisState) -> AnalysisState:
            try:
                # Add explicit format instructions
                formatted_query = (
                    "Follow these steps:\n"
                    "1. Think about the query\n"
                    "2. Write the SQL query\n"
                    "3. Execute the query\n"
                    "4. Analyze the results\n\n"
                    f"Question: {state['user_query']}"
                )
                
                agent_response = self.sql_agent.invoke({
                    "input": formatted_query
                })
                
                # Handle both structured and unstructured responses
                if isinstance(agent_response, dict):
                    result = agent_response.get("output", str(agent_response))
                else:
                    result = str(agent_response)
                    
                state["sql_results"] = {
                    "query": state["user_query"],
                    "response": result,
                    "result": result
                }
                return state
            except Exception as e:
                state["sql_results"] = {
                    "query": state["user_query"],
                    "error": str(e),
                    "result": "Error occurred during query processing"
                }
                return state

        def analyze_results(state: AnalysisState) -> AnalysisState:
            try:
                # Add explicit format instructions
                analysis_prompt = (
                    "Analyze these results in a clear format:\n"
                    "1. Key findings\n"
                    "2. Important metrics\n"
                    "3. Recommendations\n\n"
                    f"Results to analyze: {state['sql_results']['result']}"
                )
                
                analysis_response = self.sql_agent.invoke({
                    "input": analysis_prompt
                })
                
                # Handle both structured and unstructured responses
                if isinstance(analysis_response, dict):
                    analysis = analysis_response.get("output", str(analysis_response))
                else:
                    analysis = str(analysis_response)
                    
                state["analysis"] = analysis
                return state
            except Exception as e:
                state["analysis"] = f"Error during analysis: {str(e)}"
                return state

        # Set up the workflow graph
        workflow.add_node("process_query", process_query)
        workflow.add_node("analyze_results", analyze_results)
        workflow.set_entry_point("process_query")
        workflow.add_edge("process_query", "analyze_results")
        workflow.add_edge("analyze_results", END)

        return workflow

    def analyze(self, query: str) -> Dict:
        if query in self.query_cache:
            return self.query_cache[query]
        
        start_time = time.time()
        initial_state = AnalysisState(
            user_query=query,
            query_classification={},
            decomposed_questions=[],
            sql_results={},
            analysis="",
            final_output={},
            processing_time=0,
            agent_states={},
            raw_responses={},
            messages=[]
        )
        
        try:
            # Compile the workflow first
            app = self.workflow.compile()
            # Then run it with the initial state
            final_state = app.invoke(initial_state)
            
            # Ensure the result is JSON serializable
            result = {
                "query": query,
                "results": str(final_state["sql_results"]),
                "analysis": str(final_state["analysis"]),
                "processing_time": time.time() - start_time
            }
            self.query_cache[query] = result
            return result
            
        except Exception as e:
            return {"error": str(e), "query": query}

    def _analyze_results(self, query: str, sql_results: Dict) -> str:
        # Filter only essential data
        filtered_results = {
            k: {
                'question': v.get('question'),
                'result': v.get('result')
            }
            for k, v in sql_results.items()
            if 'error' not in v
        }
        
        # Compact JSON representation
        results_context = json.dumps(filtered_results, separators=(',', ':'))
        
        response = self.llm.invoke([
            SystemMessage(content=ANALYST_PROMPT),
            HumanMessage(content=f"Q:{query} R:{results_context}")
        ])
        
        return response.content

    def _direct_sql_query(self, query: str) -> Dict:
        if query in self.query_cache:
            return self.query_cache[query]
        
        try:
            result = self.sql_agent.invoke({"input": f"SQL query only: {query}"})
            sql = self._extract_sql(result['output'])
            
            if sql:
                df = pd.read_sql_query(sql, self.conn)
                results = {
                    "query_type": "direct_sql",
                    "sql_query": sql,
                    "results": df.to_dict('records'),
                    "processing_time": time.time()
                }
                self.query_cache[query] = results
                return results
                
        except Exception as e:
            return {"error": str(e), "query": query}

    def _extract_thought(self, text: str) -> str:
        if "Thought:" in text:
            return text.split("Thought:")[1].split("SQL")[0].strip()
        return ""

    def _extract_sql(self, text: str) -> str:
        if "SQL:" in text:
            sql_part = text.split("SQL:")[1]
            if "SQLResult:" in sql_part:
                return sql_part.split("SQLResult:")[0].strip()
            if "Final Answer:" in sql_part:
                return sql_part.split("Final Answer:")[0].strip()
            return sql_part.strip()
        return ""

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
            filename = f"{query[:50].replace(' ', '_').lower()}_analysis.json"
            return formatted_output + f"\n\nDetailed results saved to {filename}"
        else:
            return f"Error: {results.get('error', 'Unknown error occurred')}"
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Streamlit UI
def main():
    st.title("Database Analysis Assistant")
    
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Enter your Anthropic API Key:", type="password")
    
    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar to proceed.")
        return
        
    st.write("""
    Welcome to the Database Analysis Assistant! 
    This tool helps analyze data using natural language queries.
    Ask any question about your data, and I'll help you analyze it.
    """)
    
    config = Config()
    config.api_key = api_key
    
    query = st.text_area("Enter your analysis question:", height=100)
    
    if st.button("Analyze"):
        if query:
            with st.spinner("Analyzing your query..."):
                try:
                    analyst = DatabaseAnalyst(config)
                    results = analyst.analyze(query)
                    
                    # Display results
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        # Display formatted output
                        st.write("### Analysis Results")
                        st.write(format_output(results))
                        
                        # Display JSON results in an expandable section
                        with st.expander("View Raw JSON Results"):
                            # Create tabs for different views
                            json_tab, pretty_tab = st.tabs(["JSON", "Pretty View"])
                            
                            with json_tab:
                                st.code(json.dumps(results, indent=2), language='json')
                            
                            with pretty_tab:
                                st.json(results)
                        
                        # Add download button for detailed results
                        filename = f"{query[:50].replace(' ', '_').lower()}_analysis.json"
                        if isinstance(results, dict):
                            st.download_button(
                                label="Download Detailed Results",
                                data=json.dumps(results, indent=2),
                                file_name=filename,
                                mime="application/json"
                            )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query first.")
    
    # Add sidebar with information
    st.sidebar.title("About")
    st.sidebar.info("""
    This tool uses advanced AI to analyze data from a database.
    You can ask questions about:
    - Data trends
    - Summary statistics
    - Data comparisons
    """)
    
    # Add example queries
    st.sidebar.title("Example Queries")
    st.sidebar.markdown("""
    - What are the average values for each category?
    - How do the results compare across different groups?
    - What are the key metrics for the last analysis?
    """)

if __name__ == "__main__":
    main()