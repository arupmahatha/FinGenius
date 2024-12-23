# Part 1: Imports and Basic Setup
import streamlit as st
st.set_page_config(page_title="Stock Market Data Analyzer", layout="wide")

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

# Load API keys from environment file
load_dotenv('api_key.env')

# Initialize memory for state management
memory = {}  # Using a simple dictionary for in-memory storage

# Part 2: Type Definitions and Base Classes
class QueryType(Enum):
    DIRECT_SQL = "direct_sql"
    ANALYSIS = "analysis"

@dataclass
class QueryClassification:
    type: QueryType
    explanation: str
    raw_response: str

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
    db_path: str
    sqlite_path: str = "sqlite:///consumption.db"
    model_name: str = "claude-3-sonnet-20240229"
    human_in_the_loop: bool = False
    
    @property
    def api_key(self) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError("ANTHROPIC_API_KEY not found in environment variables")
        if not api_key.startswith('sk-'):
            raise ConfigError("Invalid API key format")
        return api_key

# Part 3: Prompt Templates
QUERY_CLASSIFIER_PROMPT = """You are a query classifier that determines if a stock market question needs complex analysis or can be answered with a direct SQL query.

Example 1:
Question: "Show me the last 5 days of stock prices"
Classification: direct_sql
Explanation: This is a straightforward data retrieval request.

Example 2:
Question: "What are the emerging trends in trading volume and their impact on price?"
Classification: analysis
Explanation: This requires complex analysis of relationships and patterns.

Respond in JSON format:
{
    "type": "direct_sql" or "analysis",
    "explanation": "brief explanation of classification"
}
"""

SQL_AGENT_PROMPT = """You are an expert financial database analyst. Your task is to:
1. Analyze stock market queries
2. Create appropriate SQL queries
3. Provide clear results

Example 1:
User: "What's the stock's performance last week?"
Thought: Need to analyze daily price changes and volume for the past week
SQL:
SELECT 
    date,
    ROUND(open, 2) as open_price,
    ROUND(close, 2) as close_price,
    ROUND(((close - open) / open * 100), 2) as daily_return,
    ROUND(high, 2) as high,
    ROUND(low, 2) as low,
    volume
FROM consumption
WHERE date >= date('now', '-7 days')
ORDER BY date DESC;

Example 2:
User: "Find volatile trading days"
Thought: Looking for days with large price ranges and high volume
SQL:
WITH metrics AS (
    SELECT AVG(volume) as avg_vol,
           AVG((high - low) / open * 100) as avg_range
    FROM consumption
)
SELECT 
    date,
    ROUND(open, 2) as open_price,
    ROUND(close, 2) as close_price,
    ROUND(((high - low) / open * 100), 2) as price_range_pct,
    volume,
    ROUND(volume / avg_vol, 2) as vol_ratio
FROM consumption, metrics
WHERE (high - low) / open * 100 > avg_range
AND volume > avg_vol
ORDER BY price_range_pct DESC
LIMIT 5;

Your responses should include:
1. Thought process
2. SQL query
3. Result interpretation"""

ANALYST_PROMPT = """You are an expert financial analyst. Analyze the provided SQL results and provide insights.

Focus on:
1. Price trends and patterns
2. Volume analysis
3. Technical indicators
4. Risk assessment
5. Notable patterns

Example Analysis Structure:
1. Key Findings
   - Main price trends
   - Volume patterns
   - Notable events

2. Technical Analysis
   - Support/resistance levels
   - Pattern recognition
   - Momentum indicators

3. Risk Assessment
   - Volatility measures
   - Liquidity analysis
   - Risk factors

4. Recommendations
   - Key levels to watch
   - Risk considerations
   - Potential scenarios

Be specific and data-driven in your analysis."""

# Part 4: Main StockAnalyzer Class
class StockAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.db = self._init_database()
        self.llm = self._init_llm()
        self.sql_agent = self._setup_sql_agent()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.anthropic_client = Anthropic(api_key=config.api_key)
        self.agent_states = {}
        self.raw_responses = {}
        self.conn = sqlite3.connect('consumption.db')

    def _init_database(self) -> SQLDatabase:
        return SQLDatabase.from_uri(self.config.sqlite_path)

    @staticmethod
    def initialize_database(csv_path: str) -> None:
        try:
            # Create SQLite database from CSV
            df = pd.read_csv(csv_path)
            conn = sqlite3.connect('consumption.db')
            df.to_sql('consumption', conn, if_exists='replace', index=False)
            conn.close()
        except Exception as e:
            raise ConfigError(f"Failed to initialize database: {str(e)}")

    def _init_llm(self) -> ChatAnthropic:
        return ChatAnthropic(
            model=self.config.model_name,
            temperature=0,
            api_key=self.config.api_key
        )

    def _setup_sql_agent(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        return create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            agent_type="zero-shot-react-description",
            verbose=True,
            prefix=SQL_AGENT_PROMPT
        )

    def _get_human_input(self, prompt: str, default_value=None) -> str:
        if not self.config.human_in_the_loop:
            return default_value
            
        # For Streamlit UI
        try:
            st.write("Human Review Required:")
            st.write(prompt)
            user_input = st.text_input("Enter your response or press Enter to accept default:", value=default_value)
            
            col1, col2 = st.columns(2)
            with col1:
                accept_default = st.button("Accept Default", key=f"accept_{prompt[:20]}")
            with col2:
                submit_modified = st.button("Submit Modified Value", key=f"submit_{prompt[:20]}")
            
            if accept_default:
                st.write("Using default value")
                return default_value
            elif submit_modified:
                st.write(f"Using modified value: {user_input}")
                return user_input
                
            # Removed st.stop() to prevent the app from halting
            return default_value
            
        except Exception as e:
            st.error(f"Error in human input: {str(e)}")
            # For command line
            response = input(f"\n{prompt}\nPress Enter to accept default or input your modification: ")
            return response if response.strip() else default_value

    def analyze(self, query: str) -> Dict:
        start_time = time.time()
        try:
            # Reset storages for new analysis
            self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
            self.agent_states = {}
            self.raw_responses = {}
            
            # First, classify the query
            classification = self._classify_query(query)
            
            # Show classification response
            st.write("### Query Classification")
            st.write(f"Type: {classification.type.value}")
            st.write(f"Explanation: {classification.explanation}")
            st.write("Raw Response:")
            st.code(classification.raw_response)
            
            # Human review of classification if enabled
            if self.config.human_in_the_loop:
                st.subheader("Query Classification Review")
                human_classification = self._get_human_input(
                    f"Query Classification: {classification.type.value}\nExplanation: {classification.explanation}\n"
                    "Enter 'direct_sql' or 'analysis' to modify, or press Enter to accept: ",
                    classification.type.value
                )
                if human_classification in ['direct_sql', 'analysis']:
                    classification.type = QueryType(human_classification)
                    st.write(f"Classification updated to: {human_classification}")
            
            # For direct SQL queries, use simplified processing
            if classification.type == QueryType.DIRECT_SQL:
                result = self._direct_sql_query(query)
                # Show the JSON output
                st.write("### Analysis Output (JSON)")
                st.json(result)
                return result
            
            # For analysis queries, use decomposition approach
            decomposed_questions = self._decompose_question(query)
            
            # Show decomposed questions
            st.write("### Decomposed Questions")
            for i, q in enumerate(decomposed_questions, 1):
                st.write(f"{i}. {q}")
            
            # Human review of decomposed questions if enabled
            if self.config.human_in_the_loop:
                st.subheader("Review Decomposed Questions")
                for i, q in enumerate(decomposed_questions):
                    modified_q = self._get_human_input(f"Question {i+1}: {q}", q)
                    decomposed_questions[i] = modified_q
                    st.write(f"Updated Question {i+1}: {modified_q}")
            
            sql_results = self._run_sql_analysis(decomposed_questions)
            
            # Show SQL results
            st.write("### SQL Analysis Results")
            for key, data in sql_results.items():
                st.write(f"\n**{key}:**")
                st.json(data)
            
            # Human review of SQL results if enabled
            if self.config.human_in_the_loop:
                st.subheader("Review SQL Results")
                for key, data in sql_results.items():
                    if 'sql' in data:
                        modified_sql = self._get_human_input(f"Review SQL for {key}:\n{data['sql']}", data['sql'])
                        if modified_sql != data['sql']:
                            try:
                                df = pd.read_sql_query(modified_sql, self.conn)
                                data['sql'] = modified_sql
                                data['result'] = df.to_dict('records')
                                st.write(f"Updated SQL for {key}")
                                st.code(modified_sql, language="sql")
                            except Exception as e:
                                st.error(f"Error with modified SQL: {str(e)}")
            
            analysis = self._analyze_results(query, sql_results)
            
            # Show analysis
            st.write("### Expert Analysis")
            st.write(analysis)
            
            # Human review of analysis if enabled
            if self.config.human_in_the_loop:
                st.subheader("Review Analysis")
                modified_analysis = self._get_human_input(
                    f"Review Analysis:\n{analysis}\nEnter modifications or press Enter to accept: ",
                    analysis
                )
                if modified_analysis != analysis:
                    analysis = modified_analysis
                    st.write("Updated Analysis:")
                    st.write(modified_analysis)
            
            processing_time = time.time() - start_time
            
            final_output = {
                "query_type": "analysis",
                "user_query": query,
                "query_classification": {
                    "type": classification.type.value,
                    "explanation": classification.explanation,
                    "raw_response": classification.raw_response
                },
                "sub_questions": decomposed_questions,
                "sql_analysis": sql_results,
                "expert_analysis": analysis,
                "timestamp": pd.Timestamp.now().isoformat(),
                "token_usage": self.token_usage,
                "processing_time": processing_time,
                "agent_states": self.agent_states,
                "raw_responses": self.raw_responses
            }
            
            # Save and show output
            filename = f"{query[:50].replace(' ', '_').lower()}_analysis.json"
            with open(filename, 'w') as f:
                json.dump(final_output, f, indent=2)
            
            st.write("### Final Analysis Output (JSON)")
            st.json(final_output)
            st.success(f"Analysis saved to {filename}")
                
            return final_output
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return {"error": str(e), "query": query}
        finally:
            self.conn.close()

    def _classify_query(self, query: str) -> QueryClassification:
        try:
            st.write("Classifying query...")
            response = self.llm.invoke([
                SystemMessage(content=QUERY_CLASSIFIER_PROMPT),
                HumanMessage(content=f"Classify this question: {query}")
            ])
            
            # Show raw response
            st.code(response.content)
            
            self._update_token_usage(response)
            classification = json.loads(response.content)
            
            self.raw_responses['classification'] = response.content
            
            return QueryClassification(
                type=QueryType(classification['type']),
                explanation=classification['explanation'],
                raw_response=response.content
            )
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return QueryClassification(
                type=QueryType.ANALYSIS,
                explanation="Classification failed, defaulting to analysis",
                raw_response=str(e)
            )

    def _direct_sql_query(self, query: str) -> Dict:
        start_time = time.time()
        try:
            st.write("Executing direct SQL query...")
            result = self.sql_agent.invoke({"input": query})
            
            # Show agent's thought process
            st.write("Agent's Response:")
            st.code(result['output'])
            
            self._update_token_usage(result)
            
            self.agent_states['direct_sql'] = result
            
            thought = self._extract_thought(result['output'])
            sql = self._extract_sql(result['output'])
            
            # Human review of SQL if enabled
            if self.config.human_in_the_loop:
                st.subheader("Review SQL Query")
                modified_sql = self._get_human_input(
                    f"Review SQL Query:\n{sql}\nEnter modifications or press Enter to accept: ",
                    sql
                )
                if modified_sql != sql:
                    sql = modified_sql
                    st.write("Using modified SQL:")
                    st.code(modified_sql, language="sql")
            
            try:
                if not sql:
                    sql_match = re.search(r'SELECT.*?;', result['output'], re.DOTALL | re.IGNORECASE)
                    if sql_match:
                        sql = sql_match.group(0)
                    else:
                        raise ValueError("Could not extract SQL query from agent output")
                
                sql = sql.split(';')[0] + ';'
                
                df = pd.read_sql_query(sql, self.conn)
                formatted_results = df.to_dict('records')
                
                # Show results
                st.write("Query Results:")
                st.dataframe(df)
                
            except Exception as e:
                formatted_results = f"Error executing SQL: {str(e)}"
                st.error(formatted_results)
            
            processing_time = time.time() - start_time
            
            output_data = {
                "query_type": "direct_sql",
                "user_query": query,
                "thought_process": thought if thought else "No thought process provided",
                "sql_query": sql,
                "results": formatted_results,
                "raw_agent_output": result['output'],
                "timestamp": pd.Timestamp.now().isoformat(),
                "token_usage": self.token_usage,
                "processing_time": processing_time,
                "agent_state": result
            }
            
            filename = f"{query[:50].replace(' ', '_').lower()}_analysis.json"
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            st.success(f"Analysis saved to {filename}")
                
            return output_data
            
        except Exception as e:
            st.error(f"Error in direct SQL query: {str(e)}")
            return {"error": str(e), "query": query}

    def _decompose_question(self, query: str) -> List[str]:
        st.write("Decomposing question...")
        response = self.llm.invoke([
            SystemMessage(content="Break down this stock analysis question into specific sub-questions that can be answered with SQL queries:"),
            HumanMessage(content=query)
        ])
        
        # Show decomposition response
        st.code(response.content)
        
        self._update_token_usage(response)
        self.raw_responses['decomposition'] = response.content
        
        questions = [
            q.strip().split(". ", 1)[1] if ". " in q else q.strip()
            for q in response.content.split("\n")
            if q.strip() and q[0].isdigit()
        ]
        
        return questions

    def _run_sql_analysis(self, questions: List[str]) -> Dict:
        results = {}
        agent_states = {}
        
        for i, question in enumerate(questions, 1):
            try:
                st.write(f"Analyzing question {i}: {question}")
                result = self.sql_agent.invoke({"input": question})
                
                # Show agent's response
                st.write(f"Agent's response for question {i}:")
                st.code(result['output'])
                
                self._update_token_usage(result)
                
                agent_states[f"question_{i}"] = result
                
                thought = self._extract_thought(result['output'])
                sql = self._extract_sql(result['output'])
                
                # Human review of SQL if enabled
                if self.config.human_in_the_loop:
                    st.subheader(f"Review SQL for Question {i}")
                    modified_sql = self._get_human_input(
                        f"Review SQL Query:\n{sql}\nEnter modifications or press Enter to accept: ",
                        sql
                    )
                    if modified_sql != sql:
                        sql = modified_sql
                        st.write("Using modified SQL:")
                        st.code(modified_sql, language="sql")
                
                try:
                    sql = sql.split(';')[0] + ';'
                    df = pd.read_sql_query(sql, self.conn)
                    parsed_result = df.to_dict('records')
                    
                    # Show results
                    st.write(f"Results for question {i}:")
                    st.dataframe(df)
                    
                except Exception as e:
                    parsed_result = f"Error executing SQL: {str(e)}"
                    st.error(parsed_result)
                
                results[f"question_{i}"] = {
                    "question": question,
                    "thought": thought if thought else "No thought process provided",
                    "sql": sql if sql else "No SQL query provided",
                    "result": parsed_result,
                    "raw_output": result['output']
                }
                    
            except Exception as e:
                st.error(f"Error analyzing question {i}: {str(e)}")
                results[f"question_{i}"] = {
                    "error": str(e),
                    "question": question
                }
        
        self.agent_states['sql_analysis'] = agent_states
        return results

    def _analyze_results(self, query: str, sql_results: Dict) -> str:
        st.write("Generating expert analysis...")
        results_context = json.dumps(sql_results, indent=2)
        response = self.llm.invoke([
            SystemMessage(content=ANALYST_PROMPT),
            HumanMessage(content=f"""
            Original Question: {query}
            
            Analysis Results:
            {results_context}
            
            Provide a comprehensive analysis.""")
        ])
        
        # Show analysis response
        st.code(response.content)
        
        self._update_token_usage(response)
        self.raw_responses['analysis'] = response.content
        
        # Human review of analysis if enabled
        if self.config.human_in_the_loop:
            st.subheader("Review Analysis")
            modified_analysis = self._get_human_input(
                f"Review Analysis:\n{response.content}\nEnter modifications or press Enter to accept: ",
                response.content
            )
            if modified_analysis != response.content:
                response.content = modified_analysis
                st.write("Updated Analysis:")
                st.write(modified_analysis)
        
        return response.content

    def _update_token_usage(self, response):
        try:
            if hasattr(response, '_raw_response') and 'usage' in response._raw_response:
                usage = response._raw_response['usage']
                self.token_usage["prompt_tokens"] += usage.get('input_tokens', 0)
                self.token_usage["completion_tokens"] += usage.get('output_tokens', 0)
            elif isinstance(response, dict) and 'usage' in response:
                usage = response['usage']
                self.token_usage["prompt_tokens"] += usage.get('input_tokens', 0)
                self.token_usage["completion_tokens"] += usage.get('output_tokens', 0)
            elif hasattr(response, 'usage'):
                usage = response.usage
                self.token_usage["prompt_tokens"] += usage.input_tokens if hasattr(usage, 'input_tokens') else 0
                self.token_usage["completion_tokens"] += usage.output_tokens if hasattr(usage, 'output_tokens') else 0
            else:
                message = response.content if hasattr(response, 'content') else str(response)
                result = self.anthropic_client.messages.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": message}],
                    max_tokens=1
                )
                if hasattr(result, 'usage'):
                    self.token_usage["prompt_tokens"] += result.usage.input_tokens
                    self.token_usage["completion_tokens"] += result.usage.output_tokens
        except Exception as e:
            st.warning(f"Error updating token usage: {str(e)}")

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
    output.append("=== Stock Analysis Results ===")
    output.append(f"\nQuery: {results.get('user_query', 'N/A')}")
    
    output.append(f"\nProcessing Time: {results.get('processing_time', 0):.2f} seconds")
    token_usage = results.get('token_usage', {})
    output.append(f"Token Usage:")
    output.append(f"  Prompt Tokens: {token_usage.get('prompt_tokens', 0)}")
    output.append(f"  Completion Tokens: {token_usage.get('completion_tokens', 0)}")
    output.append(f"  Total Tokens: {token_usage.get('prompt_tokens', 0) + token_usage.get('completion_tokens', 0)}")
    
    if "error" in results:
        output.append(f"\nError: {results['error']}")
        return "\n".join(output)
    
    if results.get('query_type') == 'direct_sql':
        output.append(f"\nThought Process: {results.get('thought_process', 'N/A')}")
        output.append(f"\nSQL Query: {results.get('sql_query', 'N/A')}")
        output.append("\nResults:")
        if isinstance(results.get('results'), list):
            df = pd.DataFrame(results['results'])
            output.append(str(df))
        else:
            output.append(str(results.get('results', 'No results available')))
    else:
        output.append("\nSub-Questions:")
        for i, q in enumerate(results.get('sub_questions', []), 1):
            output.append(f"{i}. {q}")
        
        output.append("\nSQL Analysis:")
        for key, data in results.get('sql_analysis', {}).items():
            output.append(f"\nQuestion: {data.get('question', 'N/A')}")
            if 'error' not in data:
                output.append(f"Thought Process: {data.get('thought', 'N/A')}")
                output.append(f"SQL Query: {data.get('sql', 'N/A')}")
                try:
                    if isinstance(data.get('result'), (list, dict)):
                        df = pd.DataFrame(data['result'])
                        output.append(str(df))
                    else:
                        output.append(f"Results: {data.get('result', 'No results available')}")
                except:
                    output.append(f"Results: {data.get('result', 'No results available')}")
            else:
                output.append(f"Error: {data['error']}")
        
        output.append("\nExpert Analysis:")
        output.append(results.get('expert_analysis', 'No analysis available'))
    
    return "\n".join(output)

def analyze_stock_query(query: str, csv_file: str, human_in_the_loop: bool = False) -> str:
    try:
        config = Config(db_path=csv_file, human_in_the_loop=human_in_the_loop)
        StockAnalyzer.initialize_database(csv_file)
        analyzer = StockAnalyzer(config)
        results = analyzer.analyze(query)
        
        if results and "error" not in results:
            formatted_output = format_output(results)
            filename = f"{query[:50].replace(' ', '_').lower()}_analysis.json"
            
            # Read and display saved JSON
            with open(filename, 'r') as f:
                saved_results = json.load(f)
            
            # Print output in Streamlit
            st.markdown("### Analysis Results")
            st.write(f"**Query:** {saved_results.get('user_query', 'N/A')}")
            
            st.write(f"**Processing Time:** {saved_results.get('processing_time', 0):.2f} seconds")
            token_usage = saved_results.get('token_usage', {})
            st.write("**Token Usage:**")
            st.write(f"- Prompt Tokens: {token_usage.get('prompt_tokens', 0)}")
            st.write(f"- Completion Tokens: {token_usage.get('completion_tokens', 0)}")
            st.write(f"- Total Tokens: {token_usage.get('prompt_tokens', 0) + token_usage.get('completion_tokens', 0)}")
            
            if saved_results.get('query_type') == 'direct_sql':
                st.write(f"**Thought Process:** {saved_results.get('thought_process', 'N/A')}")
                st.code(saved_results.get('sql_query', 'N/A'), language='sql')
                st.write("**Results:**")
                if isinstance(saved_results.get('results'), list):
                    st.dataframe(pd.DataFrame(saved_results['results']))
                else:
                    st.write(saved_results.get('results', 'No results available'))
            else:
                st.write("**Sub-Questions:**")
                for i, q in enumerate(saved_results.get('sub_questions', []), 1):
                    st.write(f"{i}. {q}")
                
                st.write("**SQL Analysis:**")
                for key, data in saved_results.get('sql_analysis', {}).items():
                    st.write(f"\nQuestion: {data.get('question', 'N/A')}")
                    if 'error' not in data:
                        st.write(f"Thought Process: {data.get('thought', 'N/A')}")
                        st.code(data.get('sql', 'N/A'), language='sql')
                        try:
                            if isinstance(data.get('result'), (list, dict)):
                                st.dataframe(pd.DataFrame(data['result']))
                            else:
                                st.write(f"Results: {data.get('result', 'No results available')}")
                        except:
                            st.write(f"Results: {data.get('result', 'No results available')}")
                    else:
                        st.error(f"Error: {data['error']}")
                
                st.write("**Expert Analysis:**")
                st.write(saved_results.get('expert_analysis', 'No analysis available'))
            
            st.success(f"Detailed results saved to {filename}")
            return formatted_output
        else:
            st.error(f"Error: {results.get('error', 'Unknown error occurred')}")
            return f"Error: {results.get('error', 'Unknown error occurred')}"
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return f"Error during analysis: {str(e)}"

# Save this file as stock_analyzer.py
if __name__ == "__main__":
    st.title("Stock Market Data Analyzer")
    st.write("Upload your stock market data CSV file and analyze it using natural language queries.")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Display sample data
        df = pd.read_csv("temp.csv")
        st.subheader("Preview of uploaded data")
        st.dataframe(df.head())
        
        # Enable human-in-the-loop option
        human_in_the_loop = st.checkbox("Enable Human-in-the-Loop Analysis", value=False)
        
        # Number of questions input
        num_questions = st.number_input("How many questions would you like to ask?", min_value=1, max_value=10, value=1)
        
        # Create text input fields for each question
        questions = []
        for i in range(int(num_questions)):
            question = st.text_input(f"Question {i+1}", key=f"question_{i}")
            questions.append(question)
        
        # Analysis button
        if st.button("Analyze Questions"):
            if all(questions):
                progress_bar = st.progress(0)
                for i, query in enumerate(questions, 1):
                    st.subheader(f"Analysis for Question {i}: {query}")
                    progress_text = st.empty()
                    progress_text.text("Analyzing...")
                    
                    try:
                        with st.spinner(f"Analyzing question {i}..."):
                            result = analyze_stock_query(query, "temp.csv", human_in_the_loop)
                            st.write(result)
                        progress_bar.progress(i/len(questions))
                    except Exception as e:
                        st.error(f"Error analyzing question {i}: {str(e)}")
                    
                    progress_text.empty()
            else:
                st.warning("Please fill in all question fields")

        # Cleanup
        if os.path.exists("temp.csv"):
            os.remove("temp.csv")
    else:
        st.info("Please upload a CSV file to begin analysis")