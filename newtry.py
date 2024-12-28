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
import uuid
from datetime import datetime

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
    conversation_history: List[Dict]  # Add this line

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
        
        # Add conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def load_caches(self):
        """Load both caches from files"""
        self.query_cache = {}
        self.prompt_cache = {}
        
        # Try to load query cache
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.query_cache = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load query cache: {e}")
        
        # Try to load prompt cache
        if os.path.exists(self.prompt_cache_file):
            try:
                with open(self.prompt_cache_file, 'rb') as f:
                    self.prompt_cache = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load prompt cache: {e}")

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
            # Improved prompt for better SQL generation
            enhanced_prompt = f"""Based on this question: {query}
            Generate a SQL query that will provide accurate metrics.
            Focus on calculating relevant business metrics and ensure all calculations are clear."""
            
            try:
                agent_response = self.sql_agent.invoke(
                    {"input": enhanced_prompt},
                    config={"handle_parsing_errors": True, "timeout": 30}
                )
                response_text = str(agent_response)
            except Exception as e:
                # Better error handling for agent responses
                if "Could not parse LLM output: `" in str(e):
                    analysis = str(e).split("Could not parse LLM output: `")[1].rsplit("`", 1)[0]
                    return {
                        "success": True,
                        "metrics": {"Analysis": analysis},
                        "type": "analysis"
                    }
                return {"success": False, "error": f"Query processing error: {str(e)}"}

            sql_query = self._extract_sql(response_text)
            if not sql_query:
                # Fallback to natural language response if no SQL found
                return {
                    "success": True,
                    "metrics": {"Analysis": response_text},
                    "type": "analysis"
                }
            
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

    def process_follow_up(self, follow_up_question: str, previous_context: Dict) -> Dict:
        # Add the context to the prompt
        context_prompt = f"""
        Previous context: {previous_context.get('metrics', {})}
        Follow-up question: {follow_up_question}
        
        Please answer the follow-up question using the context provided.
        """
        
        # Process the follow-up using the context
        result = self.process_query(context_prompt)
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

# Add new class for chat management
class ChatManager:
    def __init__(self):
        self.chats_dir = "saved_chats"
        os.makedirs(self.chats_dir, exist_ok=True)
        
    def save_chat(self, chat_id: str, messages: list):
        # Get chat title from first user message
        title = None
        first_query = None
        for message in messages:
            if message["role"] == "user":
                first_query = message["content"]
                # Clean and truncate the title
                clean_title = first_query.replace("\n", " ").strip()
                title = clean_title[:50] + "..." if len(clean_title) > 50 else clean_title
                break
        
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        chat_data = {
            "id": chat_id,
            "title": title,
            "first_query": first_query,
            "messages": messages,
            "timestamp": datetime.now().isoformat()
        }
        
        filename = os.path.join(self.chats_dir, f"{chat_id}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
            
    def load_chats(self) -> dict:
        chats = {}
        for filename in os.listdir(self.chats_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.chats_dir, filename), encoding='utf-8') as f:
                    chat_data = json.load(f)
                    chats[chat_data['id']] = chat_data
        return dict(sorted(chats.items(), key=lambda x: x[1]['timestamp'], reverse=True))
        
    def delete_chat(self, chat_id: str):
        filename = os.path.join(self.chats_dir, f"{chat_id}.json")
        if os.path.exists(filename):
            os.remove(filename)

# Streamlit UI
def main():
    st.set_page_config(
        page_title="SQL Database Analyst",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("SQL Database Analysis Assistant")
    
    # Initialize chat manager and session state
    chat_manager = ChatManager()
    
    # Improved session state initialization
    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'messages': [],
            'current_context': None,
            'current_chat_id': str(uuid.uuid4()),
            'chats': chat_manager.load_chats(),
            'last_query': None,
            'new_chat_clicked': False,
            'api_key_set': False
        })

    # Sidebar configuration with improved error handling
    with st.sidebar:
        st.title("Configuration Settings")
        api_key = st.text_input("Anthropic API Key:", type="password")
        
        if api_key:
            st.session_state.api_key_set = True
        
        st.title("Conversation Management")
        
        # New chat button with improved handling
        if st.button("Start New Analysis", type="primary", key="new_chat_button", disabled=not st.session_state.api_key_set):
            if not st.session_state.new_chat_clicked:
                st.session_state.new_chat_clicked = True
                
                # Save current chat if it exists
                if st.session_state.messages:
                    chat_manager.save_chat(
                        st.session_state.current_chat_id,
                        st.session_state.messages
                    )
                
                # Reset states with new chat
                new_chat_id = str(uuid.uuid4())
                st.session_state.update({
                    'current_chat_id': new_chat_id,
                    'messages': [],
                    'current_context': None,
                    'last_query': None
                })
                
                # Add welcome message
                welcome_message = {
                    "role": "assistant",
                    "content": """👋 Hello! I'm your SQL Database Analysis Assistant. 

I can help you analyze your database by:
- Running SQL queries
- Providing data insights
- Answering follow-up questions about the results

Please ask me any question about your database!"""
                }
                st.session_state.messages.append(welcome_message)
                st.rerun()
        
        # Display chat history
        st.title("Chat History")
        chats = chat_manager.load_chats()
        for chat_id, chat_data in chats.items():
            with st.expander(f"📝 {chat_data['title']}", expanded=False):
                st.write(f"Created: {datetime.fromisoformat(chat_data['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                if st.button("Load Chat", key=f"load_{chat_id}"):
                    # Save current chat before loading
                    if st.session_state.messages:
                        chat_manager.save_chat(
                            st.session_state.current_chat_id,
                            st.session_state.messages
                        )
                    # Load selected chat
                    st.session_state.messages = chat_data['messages']
                    st.session_state.current_chat_id = chat_id
                    st.rerun()

    # Reset new_chat_clicked flag
    if st.session_state.new_chat_clicked:
        st.session_state.new_chat_clicked = False

    # API key validation
    if not api_key:
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to proceed.")
        return

    # Initialize analyst with error handling
    try:
        config = Config(api_key=api_key)
        analyst = DatabaseAnalyst(config)
    except Exception as e:
        st.error(f"Failed to initialize database analyst: {str(e)}")
        return

    # Chat interface with improved message display
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add context for follow-up questions
            if message["role"] == "user":
                st.caption("Question")
            elif message["role"] == "assistant":
                st.caption("Response")
                if idx > 0 and st.session_state.messages[idx-1]["role"] == "user":
                    st.caption("Follow-up available ↓")

    # Chat input with automatic saving
    if prompt := st.chat_input("Ask me about your database...", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Process query with context
                    context = None
                    if len(st.session_state.messages) > 2:  # If there's previous conversation
                        context = st.session_state.messages[-3]["content"] if len(st.session_state.messages) >= 3 else None
                    
                    result = analyst.process_query(prompt)
                    
                    if result["success"]:
                        response = "🎯 Here are the results:\n\n"
                        for metric, value in result['metrics'].items():
                            response += f"**{metric}**: {value}\n"
                    else:
                        response = f"❌ Error: {result.get('error', 'Unknown error')}"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Save chat after each interaction
                    chat_manager.save_chat(
                        st.session_state.current_chat_id,
                        st.session_state.messages
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()