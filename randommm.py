import streamlit as st
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
from dotenv import load_dotenv
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import sqlite3
import re
from langchain.memory import ConversationBufferMemory

# Load API keys from environment file
load_dotenv('api_key.env')

# Initialize memory for state management
memory = {}  # Using a simple dictionary for in-memory storage

# Type Definitions and Base Classes
class QueryType(Enum):
    DIRECT_SQL = "direct_sql" 
    ANALYSIS = "analysis"

@dataclass
class QueryClassification:
    type: QueryType
    explanation: str
    raw_response: str
    confidence: float = 1.0

class AnalysisState(TypedDict):
    user_query: str
    query_classification: Dict
    decomposed_questions: List[str]
    sql_results: Dict
    analysis: str
    final_output: Dict
    token_usage: Dict
    processing_time: float
    agent_states: Dict
    raw_responses: Dict
    messages: List[AnyMessage]

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

@dataclass
class Config:
    db_path: str = "final_working_database.db"
    sqlite_path: str = "sqlite:///final_working_database.db"
    model_name: str = "claude-3-sonnet-20240229"
    confidence_threshold: float = 0.85
    
    @property
    def api_key(self) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError("ANTHROPIC_API_KEY not found in api_key.env file")
        return api_key

# Prompt Templates
QUERY_CLASSIFIER_PROMPT = """You are a query classifier that determines if a stock market question:
1. Can be answered with a direct SQL query
2. Needs complex analysis

Respond in JSON format:
{
    "type": "direct_sql" | "analysis",
    "explanation": "brief explanation of classification",
    "confidence": <float between 0-1>,
    "needs_clarification": {
        "required": <boolean>,
        "details": "description of ambiguity or missing information",
        "suggested_questions": ["list of clarifying questions"]
    }
}"""

SQL_AGENT_PROMPT = """You are an expert financial database analyst. Your task is to:
1. Analyze stock market queries
2. Create appropriate SQL queries using the provided database schema
3. Provide clear results

Here is the database schema:
{schema}

If you encounter any ambiguity or data limitations:
1. Clearly explain the issue
2. Specify what clarification is needed
3. Suggest possible alternatives"""

ANALYST_PROMPT = """You are an expert financial analyst. Analyze the provided SQL results and provide insights.

If you encounter:
- Unclear patterns
- Multiple possible interpretations
- Need for additional context
- Insufficient data

Clearly state:
1. What additional information would help
2. Why it's needed
3. How it would improve the analysis

Focus on:
1. Price trends and patterns
2. Volume analysis
3. Technical indicators
4. Risk assessment
5. Notable patterns

Be specific and data-driven in your analysis."""

class StockAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.conn = sqlite3.connect(config.db_path)
        self.schema = self._get_database_schema()
        self.db = SQLDatabase.from_uri(config.sqlite_path)
        self.llm = ChatAnthropic(
            model=config.model_name,
            temperature=0,
            api_key=config.api_key
        )
        self.sql_agent = create_sql_agent(
            llm=self.llm,
            toolkit=SQLDatabaseToolkit(db=self.db, llm=self.llm),
            agent_type="zero-shot-react-description",
            verbose=True
        )
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.anthropic_client = Anthropic(api_key=config.api_key)

    def _get_database_schema(self) -> str:
        cursor = self.conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        schema = []
        for table in tables:
            table_name = table[0]
            columns = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
            schema.append(f"Table: {table_name}")
            schema.append("Columns:")
            for col in columns:
                schema.append(f"  - {col[1]} ({col[2]})")
            schema.append("")
        return "\n".join(schema)

    def analyze(self, query: str) -> Dict:
        start_time = time.time()
        
        # Classify query
        classification = self._classify_query(query)
        
        # Get SQL results
        sql_results = self._get_sql_results(query)
        
        # Get analysis
        analysis = self._get_analysis(query, sql_results)
        
        processing_time = time.time() - start_time
        
        return {
            "query": query,
            "classification": classification,
            "sql_results": sql_results,
            "analysis": analysis,
            "processing_time": processing_time,
            "token_usage": self.token_usage
        }

    def _classify_query(self, query: str) -> Dict:
        response = self.llm.invoke([
            SystemMessage(content=QUERY_CLASSIFIER_PROMPT),
            HumanMessage(content=f"Classify this question: {query}")
        ])
        self._update_token_usage(response)
        return json.loads(response.content)

    def _get_sql_results(self, query: str) -> Dict:
        result = self.sql_agent.invoke({"input": query})
        self._update_token_usage(result)
        
        sql = self._extract_sql(result['output'])
        if sql:
            try:
                df = pd.read_sql_query(sql, self.conn)
                return {
                    "sql": sql,
                    "results": df.to_dict('records'),
                    "error": None
                }
            except Exception as e:
                return {
                    "sql": sql,
                    "results": None,
                    "error": str(e)
                }
        return {
            "sql": None,
            "results": None,
            "error": "Could not extract SQL query"
        }

    def _get_analysis(self, query: str, sql_results: Dict) -> str:
        response = self.llm.invoke([
            SystemMessage(content=ANALYST_PROMPT),
            HumanMessage(content=f"""
            Original Question: {query}
            
            SQL Results:
            {json.dumps(sql_results, indent=2)}
            
            Provide a comprehensive analysis.""")
        ])
        self._update_token_usage(response)
        return response.content

    def _update_token_usage(self, response):
        if hasattr(response, 'usage'):
            usage = response.usage
            self.token_usage["prompt_tokens"] += getattr(usage, 'input_tokens', 0)
            self.token_usage["completion_tokens"] += getattr(usage, 'output_tokens', 0)

    def _extract_sql(self, text: str) -> Optional[str]:
        if "SQL:" in text:
            sql_part = text.split("SQL:")[1]
            if "SQLResult:" in sql_part:
                return sql_part.split("SQLResult:")[0].strip()
            if "Final Answer:" in sql_part:
                return sql_part.split("Final Answer:")[0].strip()
            return sql_part.strip()
        return None

def display_results(results: Dict):
    st.write("=== Analysis Results ===")
    
    # Display query and classification
    st.subheader("Query Information")
    st.write(f"Query: {results['query']}")
    st.write("Classification:", results['classification'])
    
    # Display SQL results
    st.subheader("SQL Results")
    if results['sql_results']['error']:
        st.error(f"SQL Error: {results['sql_results']['error']}")
    else:
        st.write("SQL Query:", results['sql_results']['sql'])
        if results['sql_results']['results']:
            st.dataframe(pd.DataFrame(results['sql_results']['results']))
    
    # Display analysis
    st.subheader("Expert Analysis")
    st.write(results['analysis'])
    
    # Display performance metrics
    st.subheader("Performance Metrics")
    st.write(f"Processing Time: {results['processing_time']:.2f} seconds")
    st.write("Token Usage:", results['token_usage'])

def main():
    st.title("Stock Market Analysis System")
    
    # Get number of questions
    num_questions = st.number_input("Enter number of questions:", min_value=1, value=1)
    
    # Create list to store questions
    questions = []
    for i in range(int(num_questions)):
        question = st.text_input(f"Question {i+1}:", key=f"q_{i}")
        if question:
            questions.append(question)
    
    if st.button("Analyze") and questions:
        # Initialize analyzer
        config = Config()
        analyzer = StockAnalyzer(config)
        
        # Process each question
        for i, query in enumerate(questions, 1):
            st.write(f"\n=== Processing Question {i} ===")
            
            # Get analysis results
            results = analyzer.analyze(query)
            
            # Display results
            display_results(results)
            
            # Save results to file
            filename = f"analysis_{i}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            st.write(f"\nResults saved to {filename}")
            
            st.write("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()