import os
from typing import Dict, List, Optional, TypedDict, Literal, Union, Annotated
from dataclasses import dataclass, field
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

# Add new state management for chat
@dataclass
class ChatState:
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_context: Dict = field(default_factory=dict)
    last_analysis: Optional[Dict] = None

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
        
        # Add conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chat_state = ChatState()

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

        retries = 0
        max_retries = 2
        last_error = None

        while retries <= max_retries:
            try:
                prompt = METRIC_CALCULATION_PROMPT.format(question=query)
                if retries > 0:
                    # Add error context for retry
                    prompt += f"\nPrevious attempt failed with error: {last_error}\nPlease correct the SQL query and try again."
                
                agent_response = self.sql_agent.invoke({
                    "input": prompt
                }, config={"handle_parsing_errors": False})
                response_text = str(agent_response)
                
                sql_query = self._extract_sql(response_text)
                if not sql_query:
                    return {"success": False, "error": "No valid SQL query was generated"}
                
                result = self._execute_sql_safely(sql_query)
                if result["success"]:
                    self.query_cache[cache_key] = result
                    self.save_caches()
                    return result
                else:
                    last_error = result["error"]
                    retries += 1
                    continue
                    
            except Exception as e:
                error_text = str(e)
                # Still handle analysis responses as before
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
                
                last_error = str(e)
                retries += 1
                continue

        # If we've exhausted retries, return the last error
        return {
            "success": False, 
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}"
        }

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

    def _execute_sql_safely(self, query: str, max_retries: int = 2) -> Dict:
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query)
            
            metrics = {}
            for row in cursor.fetchall():
                metrics[str(row[0])] = row[1] if len(row) > 1 else row[0]
            
            return {"metrics": metrics, "success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def analyze(self, query: str) -> Dict:
        if query in self.query_cache:
            return self.query_cache[query]
        
        state = AnalysisState(user_query=query)
        result = self.process_query(state)
        self.query_cache[query] = result
        return result

    def retry_with_different_approach(self, query: str, attempt: int = 1) -> Dict:
        """Retry query with different approaches based on attempt number"""
        
        # Different prompts for different retry attempts
        retry_prompts = {
            1: """Let's try this again. Please write a simple SQL query to answer this question. 
            Focus on the core metrics needed and use basic SQL operations:
            {query}""",
            
            2: """Break this down step by step:
            1. First, identify the main tables needed
            2. Then, write a basic SQL query using only essential joins
            3. Focus on calculating exactly what was asked
            Question: {query}""",
            
            3: """One final attempt. Write the simplest possible SQL query that could answer this:
            1. Use only basic SELECT, FROM, WHERE, GROUP BY
            2. Avoid complex subqueries if possible
            3. Focus on direct answers
            Question: {query}"""
        }
        
        try:
            # Get the appropriate prompt for this attempt
            prompt = retry_prompts.get(attempt, retry_prompts[1]).format(query=query)
            
            # Add attempt context to the state
            agent_response = self.sql_agent.invoke({
                "input": prompt,
                "attempt": attempt
            })
            
            response_text = str(agent_response)
            sql_query = self._extract_sql(response_text)
            
            if sql_query:
                result = self._execute_sql_safely(sql_query)
                if result["success"]:
                    return result
                
            return {
                "success": False,
                "error": f"Attempt {attempt}: Could not generate valid SQL",
                "attempt": attempt
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Attempt {attempt}: {str(e)}",
                "attempt": attempt
            }

    def process_chat_query(self, user_message: str) -> Dict:
        """Process a chat message with context and retry logic"""
        cache_key = self.get_cache_key(user_message)
        
        # Check if this is a follow-up question
        if self.chat_state.last_analysis:
            context = f"""Previous analysis context:
            {json.dumps(self.chat_state.last_analysis, indent=2)}
            
            Follow-up question: {user_message}"""
        else:
            context = user_message

        # Try up to 3 times with different approaches
        for attempt in range(1, 4):
            try:
                result = self.retry_with_different_approach(context, attempt)
                if result["success"]:
                    self.chat_state.conversation_history.append({
                        "user": user_message,
                        "assistant": result
                    })
                    self.chat_state.last_analysis = result
                    return result
                
            except Exception as e:
                continue
        
        # If all attempts fail, return the last error
        return {
            "success": False,
            "error": "Failed to generate valid SQL after multiple attempts. Please try rephrasing your question.",
            "suggestions": [
                "Be more specific about what you want to calculate",
                "Specify which tables or metrics you're interested in",
                "Break down complex questions into simpler parts"
            ]
        }

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

@dataclass
class Chat:
    id: str
    name: str
    messages: List[Dict[str, any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

# Streamlit UI
def main():
    st.title("Database Analysis Chatbot")
    
    # Initialize session state
    if 'chats' not in st.session_state:
        st.session_state.chats = [Chat(id="default", name="New Chat")]
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = "default"
    if 'analyst' not in st.session_state:
        st.session_state.analyst = None

    # Sidebar configuration
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Enter your Anthropic API Key:", type="password")
    
    # Chat management in sidebar
    st.sidebar.title("Chats")
    
    # Always show New Chat button
    if st.sidebar.button("New Chat", key="new_chat_btn"):
        new_chat_id = f"chat_{int(time.time())}"  # Use timestamp for unique IDs
        new_chat = Chat(id=new_chat_id, name="New Chat")
        st.session_state.chats.append(new_chat)
        st.session_state.current_chat_id = new_chat_id
        st.rerun()

    # Chat selector with direct ID mapping
    chat_options = {chat.id: chat.name for chat in st.session_state.chats}
    selected_chat_id = st.sidebar.selectbox(
        "Select Chat",
        options=list(chat_options.keys()),
        format_func=lambda x: chat_options[x],
        key="chat_selector",
        index=list(chat_options.keys()).index(st.session_state.current_chat_id)
    )

    # Update current chat immediately if changed
    if selected_chat_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat_id
        st.rerun()

    # Get current chat
    current_chat = next(
        chat for chat in st.session_state.chats 
        if chat.id == st.session_state.current_chat_id
    )

    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar to proceed.")
        return

    # Initialize analyst if not already done
    if not st.session_state.analyst:
        config = Config(api_key=api_key)
        st.session_state.analyst = DatabaseAnalyst(config)

    # Chat interface
    st.write("Chat with your database! Ask questions and get insights.")
    
    # Display chat history
    for idx, message in enumerate(current_chat.messages):
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("assistant"):
            if isinstance(message["assistant"], dict):
                if message["assistant"].get("success"):
                    st.write("Analysis Results:")
                    st.write(message["assistant"]["metrics"])
                    
                    # Add download buttons in columns
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        # JSON download
                        json_data = json.dumps({
                            "query": message["user"],
                            "results": message["assistant"],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }, indent=2)
                        st.download_button(
                            label="📥 JSON",
                            data=json_data,
                            file_name=f"analysis_{idx}.json",
                            mime="application/json",
                            key=f"download_json_{idx}"
                        )
                    
                    with col2:
                        # CSV download (if metrics are present)
                        if "metrics" in message["assistant"]:
                            df = pd.DataFrame(
                                message["assistant"]["metrics"].items(), 
                                columns=['Metric', 'Value']
                            )
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="📥 CSV",
                                data=csv_data,
                                file_name=f"analysis_{idx}.csv",
                                mime="text/csv",
                                key=f"download_csv_{idx}"
                            )
                else:
                    st.error(str(message["assistant"].get("error", "Unknown error")))
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("🔄 Retry", key=f"retry_{idx}"):
                            with st.spinner("Retrying analysis..."):
                                try:
                                    st.session_state.analyst.query_cache = {}
                                    response = st.session_state.analyst.process_chat_query(message["user"])
                                    
                                    if response.get("success"):
                                        st.success("Retry successful!")
                                        current_chat.messages[idx]["assistant"] = response
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error(f"Retry failed: {response.get('error', 'Unknown error')}")
                                        if "suggestions" in response:
                                            st.info("Suggestions for rephrasing:")
                                            for suggestion in response["suggestions"]:
                                                st.write(f"• {suggestion}")
                                except Exception as e:
                                    st.error(f"Retry error: {str(e)}")
            else:
                st.error(str(message["assistant"]))
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🔄 Retry", key=f"retry_str_{idx}"):
                        with st.spinner("Retrying analysis..."):
                            try:
                                st.session_state.analyst.query_cache = {}
                                response = st.session_state.analyst.process_chat_query(message["user"])
                                
                                if response.get("success"):
                                    st.success("Retry successful!")
                                    current_chat.messages[idx]["assistant"] = response
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error(f"Retry failed: {response.get('error', 'Unknown error')}")
                                    if "suggestions" in response:
                                        st.info("Suggestions for rephrasing:")
                                        for suggestion in response["suggestions"]:
                                            st.write(f"• {suggestion}")
                            except Exception as e:
                                st.error(f"Retry error: {str(e)}")

    # Chat input
    user_input = st.chat_input("Ask a question about your data...")
    
    if user_input:
        # Get current chat again as it might have changed
        current_chat = next(
            chat for chat in st.session_state.chats 
            if chat.id == st.session_state.current_chat_id
        )
        
        # Process query and update chat
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.analyst.process_chat_query(user_input)
                
                if response["success"]:
                    st.write("Analysis Results:")
                    st.write(response["metrics"])
                else:
                    # Show that we're retrying
                    if "attempt" in response:
                        st.info(f"Retrying... ({response['attempt']}/2)")
                    st.error(f"Error: {response.get('error', 'Unknown error')}")

        # Store in chat history
        current_chat.messages.append({
            "user": user_input,
            "assistant": response
        })

        # Update chat name if this is the first message
        if len(current_chat.messages) == 1:
            chat_name = user_input[:30] + ("..." if len(user_input) > 30 else "")
            current_chat.name = chat_name

if __name__ == "__main__":
    main()
