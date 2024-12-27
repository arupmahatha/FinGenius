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
    confidence_threshold: float = 0.85  # High confidence threshold for autonomous decisions
    api_key: str = ""  # Initialize empty API key

# Part 3: Prompt Templates
QUERY_CLASSIFIER_PROMPT = """You are a query classifier that determines if a database query:
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

SQL_AGENT_PROMPT = """You are an expert database analyst. Your task is to:
1. Analyze database queries
2. Create appropriate SQL queries using the provided database schema
3. Provide clear results

If you encounter any ambiguity or data limitations:
1. Clearly explain the issue
2. Specify what clarification is needed
3. Suggest possible alternatives

Your responses should include:
1. Confidence level (0-1)
2. Any clarification needed
3. Thought process
4. SQL query (if possible)
5. Result interpretation"""

ANALYST_PROMPT = """You are an expert analyst. Analyze the provided SQL results and provide insights.

If you encounter:
- Unclear patterns
- Multiple possible interpretations
- Need for additional context
- Insufficient data

Clearly state:
1. What additional information would help
2. Why it's needed
3. How it would improve the analysis

Be specific and data-driven in your analysis."""

# Part 4: Main StockAnalyzer Class
# Part 4: Main StockAnalyzer Class
class StockAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.conn = sqlite3.connect(config.db_path)
        self.schema = self._get_database_schema()
        self.db = self._init_database()
        self.llm = self._init_llm()
        
        self.classifier_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.sql_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.analyst_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.sql_agent = self._setup_sql_agent()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.anthropic_client = Anthropic(api_key=config.api_key)
        self.agent_states = {}
        self.raw_responses = {}
        self.query_cache = {}

    def _init_database(self) -> SQLDatabase:
        try:
            return SQLDatabase.from_uri(self.config.sqlite_path)
        except Exception as e:
            raise ConfigError(f"Database initialization failed: {str(e)}")

    def _get_database_schema(self) -> str:
        try:
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
        except Exception as e:
            raise ConfigError(f"Failed to get database schema: {str(e)}")

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
            memory=self.sql_memory,
            prefix=SQL_AGENT_PROMPT
        )

    # def _get_user_clarification(self, prompt: str) -> str:
    #     return input(f"\n{prompt}\nPlease provide clarification: ")

    def analyze(self, query: str) -> Dict:
        if query in self.query_cache:
            print("Using cached results...")
            return self.query_cache[query]
            
        start_time = time.time()
        try:
            self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
            self.agent_states = {}
            self.raw_responses = {}

            classification = self._classify_query(query)
            
            if classification.type == QueryType.DIRECT_SQL and classification.confidence >= self.config.confidence_threshold:
                result = self._direct_sql_query(query)
                self.query_cache[query] = result
                return result
            
            decomposed_questions = self._decompose_question(query)
            sql_results = self._run_sql_analysis(decomposed_questions)
            
            # # Only ask for clarification if there are errors
            # for result in sql_results.values():
            #     if isinstance(result.get('result'), str) and 'error' in result.get('result', '').lower():
            #         clarification = self._get_user_clarification(
            #             f"Error in SQL execution: {result['result']}\nHow would you like to proceed?"
            #         )
            #         if clarification:
            #             result = self._retry_sql_query(result['question'], clarification)
            #             if result:
            #                 sql_results[result['question']] = result
            
            analysis = self._analyze_results(query, sql_results)
            
            processing_time = time.time() - start_time
            
            final_output = {
                "query_type": classification.type.value,
                "user_query": query,
                "query_classification": {
                    "type": classification.type.value,
                    "explanation": classification.explanation,
                    "confidence": classification.confidence,
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
            
            filename = f"{query[:50].replace(' ', '_').lower()}_analysis.json"
            with open(filename, 'w') as f:
                json.dump(final_output, f, indent=2)
                
            self.query_cache[query] = final_output
            return final_output
            
        except Exception as e:
            return {"error": str(e), "query": query}
        finally:
            self.conn.close()

    def _classify_query(self, query: str) -> QueryClassification:
        chat_history = self.classifier_memory.load_memory_variables({})["chat_history"]
        for message in chat_history:
            if isinstance(message, HumanMessage) and query.lower() in message.content.lower():
                print("Using cached classification...")
                return self.query_cache.get(message.content, {}).get("query_classification")
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=QUERY_CLASSIFIER_PROMPT),
                HumanMessage(content=f"Classify this question: {query}")
            ])
            
            self.classifier_memory.save_context(
                {"input": query},
                {"output": response.content}
            )
            
            self._update_token_usage(response)
            classification = json.loads(response.content)
            
            self.raw_responses['classification'] = response.content
            
            # # Only ask for clarification if confidence is low
            # if classification.get('confidence', 1.0) < 0.5:
            #     details = classification.get('needs_clarification', {}).get('details', '')
            #     questions = classification.get('needs_clarification', {}).get('suggested_questions', [])
            #     clarification = self._get_user_clarification(
            #         f"Low confidence in classification. {details}\n\nSuggested questions:\n" + 
            #         "\n".join(f"- {q}" for q in questions)
            #     )
            #     return self._classify_query(f"{query} {clarification}")
            
            return QueryClassification(
                type=QueryType(classification['type']),
                explanation=classification['explanation'],
                confidence=classification.get('confidence', 1.0),
                raw_response=response.content
            )
        except Exception as e:
            return QueryClassification(
                type=QueryType.ANALYSIS,
                explanation=f"Classification failed: {str(e)}",
                confidence=0.0,
                raw_response=str(e)
            )

    def _direct_sql_query(self, query: str) -> Dict:
        chat_history = self.sql_memory.load_memory_variables({})["chat_history"]
        for message in chat_history:
            if isinstance(message, HumanMessage) and query.lower() in message.content.lower():
                print("Using cached SQL query results...")
                return self.query_cache.get(message.content, {})
        
        start_time = time.time()
        try:
            result = self.sql_agent.invoke({"input": query})
            self._update_token_usage(result)
            
            self.agent_states['direct_sql'] = result
            
            thought = self._extract_thought(result['output'])
            sql = self._extract_sql(result['output'])
            
            # # Only ask for clarification if SQL generation fails
            # if not sql:
            #     clarification = self._get_user_clarification(
            #         "Could not generate SQL query. Please provide guidance on what data you're looking for:"
            #     )
            #     result = self.sql_agent.invoke({"input": f"{query} {clarification}"})
            #     sql = self._extract_sql(result['output'])
            
            try:
                sql = sql.split(';')[0] + ';'
                df = pd.read_sql_query(sql, self.conn)
                formatted_results = df.to_dict('records')
            except Exception as e:
                # # Only ask for clarification if SQL execution fails
                # clarification = self._get_user_clarification(
                #     f"Error executing SQL: {str(e)}\nHow would you like to modify the query?"
                # )
                # try:
                #     df = pd.read_sql_query(clarification, self.conn)
                #     formatted_results = df.to_dict('records')
                # except Exception as e2:
                formatted_results = f"Error executing SQL: {str(e)}"
            
            processing_time = time.time() - start_time
            
            final_result = {
                "query_type": "direct_sql",
                "user_query": query,
                "thought_process": thought,
                "sql_query": sql,
                "results": formatted_results,
                "raw_agent_output": result['output'],
                "timestamp": pd.Timestamp.now().isoformat(),
                "token_usage": self.token_usage,
                "processing_time": processing_time,
                "agent_state": result
            }
            
            self.sql_memory.save_context(
                {"input": query},
                {"output": json.dumps(final_result)}
            )
            
            return final_result
            
        except Exception as e:
            return {"error": str(e), "query": query}

    def _decompose_question(self, query: str) -> List[str]:
        response = self.llm.invoke([
            SystemMessage(content="Break down this stock analysis question into specific sub-questions that can be answered with SQL queries:"),
            HumanMessage(content=query)
        ])
        
        self._update_token_usage(response)
        self.raw_responses['decomposition'] = response.content
        
        questions = [
            q.strip().split(". ", 1)[1] if ". " in q else q.strip()
            for q in response.content.split("\n")
            if q.strip() and q[0].isdigit()
        ]
        
        # # Only ask for clarification if no questions were generated
        # if not questions:
        #     clarification = self._get_user_clarification(
        #         "Could not break down the question. Please specify what aspects you want to analyze:"
        #     )
        #     return self._decompose_question(f"{query} {clarification}")
        
        return questions

    def _run_sql_analysis(self, questions: List[str]) -> Dict:
        results = {}
        agent_states = {}
        
        for i, question in enumerate(questions, 1):
            chat_history = self.sql_memory.load_memory_variables({})["chat_history"]
            cached_result = None
            for message in chat_history:
                if isinstance(message, HumanMessage) and question.lower() in message.content.lower():
                    print(f"Using cached results for sub-question {i}...")
                    cached_result = self.query_cache.get(message.content)
                    break
            
            if cached_result:
                results[f"question_{i}"] = cached_result
                continue
            try:
                result = self.sql_agent.invoke({"input": question})
                self._update_token_usage(result)
                
                agent_states[f"question_{i}"] = result
                
                thought = self._extract_thought(result['output'])
                sql = self._extract_sql(result['output'])
                
                if not sql:
                    clarification = self._get_user_clarification(
                        f"Could not generate SQL for: {question}\nPlease provide guidance:"
                    )
                    result = self.sql_agent.invoke({"input": f"{question} {clarification}"})
                    sql = self._extract_sql(result['output'])
                
                try:
                    sql = sql.split(';')[0] + ';'
                    df = pd.read_sql_query(sql, self.conn)
                    parsed_result = df.to_dict('records')
                except Exception as e:
                    clarification = self._get_user_clarification(
                        f"Error executing SQL for: {question}\n{str(e)}\nHow would you like to modify the query?"
                    )
                    try:
                        df = pd.read_sql_query(clarification, self.conn)
                        parsed_result = df.to_dict('records')
                        sql = clarification
                    except Exception as e2:
                        parsed_result = f"Error executing SQL even after clarification: {str(e2)}"
                
                results[f"question_{i}"] = {
                    "question": question,
                    "thought": thought if thought else "No thought process provided",
                    "sql": sql if sql else "No SQL query provided",
                    "result": parsed_result,
                    "raw_output": result['output']
                }
                    
            except Exception as e:
                results[f"question_{i}"] = {
                    "error": str(e),
                    "question": question
                }
        
        self.agent_states['sql_analysis'] = agent_states
        return results

    def _analyze_results(self, query: str, sql_results: Dict) -> str:
        results_context = json.dumps(sql_results, indent=2)
        response = self.llm.invoke([
            SystemMessage(content=ANALYST_PROMPT),
            HumanMessage(content=f"""
            Original Question: {query}
            
            Analysis Results:
            {results_context}
            
            Provide a comprehensive analysis.""")
        ])
        
        self._update_token_usage(response)
        self.raw_responses['analysis'] = response.content
        
        if len(response.content.strip()) < 100:  # If analysis is too short
            clarification = self._get_user_clarification(
                "Analysis seems incomplete. What specific aspects would you like to focus on?"
            )
            return self._analyze_results(f"{query} {clarification}", sql_results)
        
        return response.content

    def _retry_sql_query(self, question: str, clarification: str) -> Dict:
        try:
            result = self.sql_agent.invoke({"input": f"{question} {clarification}"})
            sql = self._extract_sql(result['output'])
            
            if sql:
                df = pd.read_sql_query(sql, self.conn)
                return {
                    "question": question,
                    "thought": self._extract_thought(result['output']),
                    "sql": sql,
                    "result": df.to_dict('records'),
                    "raw_output": result['output']
                }
        except Exception as e:
            return None

    def _update_token_usage(self, response):
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

# Initialize the database schema once at the start
config = Config()
analyzer = StockAnalyzer(config)
schema = analyzer.schema  # Store the schema for later use

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

def analyze_stock_query(query: str) -> str:
    try:
        config = Config()
        analyzer = StockAnalyzer(config)
        results = analyzer.analyze(query)
        
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
    
    # Add API key input in sidebar
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Enter your Anthropic API Key:", type="password")
    
    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar to proceed.")
        return
        
    st.write("""
    Welcome to the Database Analysis Assistant! 
    This tool helps analyze data using natural language queries.
    """)
    
    # Initialize config with API key
    config = Config()
    config.api_key = api_key
    
    # Query input
    query = st.text_area("Enter your analysis question:", height=100)
    
    if st.button("Analyze"):
        if query:
            with st.spinner("Analyzing your query..."):
                try:
                    # Create analyzer instance with the config containing API key
                    analyzer = StockAnalyzer(config)
                    results = analyzer.analyze(query)
                    
                    # Convert results to formatted output
                    result = format_output(results)
                    
                    # Display results in sections
                    st.subheader("Analysis Results")
                    
                    # Convert the string output to a more structured format
                    sections = result.split("\n\n")
                    
                    for section in sections:
                        if section.startswith("==="):
                            st.markdown("---")
                            st.markdown(f"**{section.strip('=')}**")
                        elif ":" in section:
                            title, content = section.split(":", 1)
                            st.markdown(f"**{title}:**")
                            st.write(content)
                        else:
                            st.write(section)
                            
                    # Add download button for JSON results
                    filename = f"{query[:50].replace(' ', '_').lower()}_analysis.json"
                    if os.path.exists(filename):
                        with open(filename, "r") as f:
                            json_str = f.read()
                        st.download_button(
                            label="Download Detailed Results",
                            data=json_str,
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