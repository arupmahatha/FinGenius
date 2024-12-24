import streamlit as st
import pandas as pd
import os
from typing import Dict, List, Optional, TypedDict, Literal, Union
from dataclasses import dataclass
from enum import Enum
import json
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from dotenv import load_dotenv
import time
from anthropic import Anthropic
import sqlite3
import numpy as np

# Load environment variables
load_dotenv('api_key.env')

# Define base classes and types
class QueryType(Enum):
    DIRECT_SQL = "direct_sql"
    ANALYSIS = "analysis"

@dataclass
class QueryClassification:
    type: QueryType
    explanation: str
    raw_response: str

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

@dataclass
class Config:
    db_path: str
    sqlite_path: str = "sqlite:///stock_data.db"
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

# Define prompts
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

The database has a table named 'stocks' with columns: date, open, high, low, close, volume.

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
FROM stocks
WHERE date >= date('now', '-7 days')
ORDER BY date DESC;

Keep responses clear and structured with:
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

Your analysis should include specific numbers, percentages, and actionable insights."""

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
        self.conn = sqlite3.connect('stock_data.db')

    def _init_database(self) -> SQLDatabase:
        return SQLDatabase.from_uri(self.config.sqlite_path)

    @staticmethod
    def initialize_database(csv_path: str) -> None:
        try:
            if not os.path.exists(csv_path):
                raise ConfigError(f"CSV file not found at path: {csv_path}")
                
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Ensure date column is in correct format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Create SQLite database
            conn = sqlite3.connect('stock_data.db')
            df.to_sql('stocks', conn, if_exists='replace', index=False)
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
            verbose=True
        )

    def _get_human_input(self, prompt: str, default_value=None, key=None) -> str:
        """Improved human input handling with state management"""
        if not self.config.human_in_the_loop:
            return default_value

        # Create container for this input
        container = st.container()
        
        with container:
            st.write("---")
            st.write("### Human Review Required")
            st.write(prompt)
            
            # Create unique keys for this input
            input_key = f"input_{key if key else hash(prompt)}"
            accept_key = f"accept_{hash(prompt)}"
            submit_key = f"submit_{hash(prompt)}"
            
            # Initialize session state for this input
            if input_key not in st.session_state:
                st.session_state[input_key] = default_value

            # Text input
            user_input = st.text_area(
                "Enter your response or accept default:",
                value=st.session_state[input_key],
                key=input_key,
                height=100
            )

            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                accept = st.button("✓ Accept Default", key=accept_key)
            with col2:
                submit = st.button("→ Submit Modified", key=submit_key)

            if accept:
                st.session_state[input_key] = default_value
                st.success("Using default value")
                return default_value
            elif submit:
                if user_input.strip():
                    st.session_state[input_key] = user_input
                    st.success("Using modified value")
                    return user_input
                else:
                    st.warning("Input cannot be empty. Using default value.")
                    st.session_state[input_key] = default_value
                    return default_value

            st.write("---")
            return st.session_state[input_key]

    def _classify_query(self, query: str) -> QueryClassification:
        try:
            st.write("### Query Classification")
            st.info("Classifying your query...")
            
            response = self.llm.invoke([
                SystemMessage(content=QUERY_CLASSIFIER_PROMPT),
                HumanMessage(content=f"Classify this question: {query}")
            ])
            
            classification = json.loads(response.content)
            
            # Display classification
            st.write(f"🔍 Classification: **{classification['type']}**")
            st.write(f"📝 Explanation: {classification['explanation']}")
            
            # Human review if enabled
            if self.config.human_in_the_loop:
                classification_type = self._get_human_input(
                    f"Current classification: {classification['type']}\nDo you want to modify this classification?",
                    classification['type'],
                    key=f"classify_{hash(query)}"
                )
                if classification_type != classification['type']:
                    classification['type'] = classification_type
                    st.success(f"Classification updated to: {classification_type}")
            
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

    def analyze(self, query: str) -> Dict:
        if 'analysis_step' not in st.session_state:
            st.session_state.analysis_step = 'start'
            st.session_state.results = {}

        try:
            if st.session_state.analysis_step == 'start':
                # Step 1: Classification
                classification = self._classify_query(query)
                st.session_state.classification = classification
                st.session_state.analysis_step = 'sql_generation'
                st.experimental_rerun()

            elif st.session_state.analysis_step == 'sql_generation':
                # Step 2: SQL Generation
                if st.session_state.classification.type == QueryType.DIRECT_SQL:
                    result = self._direct_sql_query(query)
                else:
                    result = self._complex_analysis(query)
                
                st.session_state.results = result
                st.session_state.analysis_step = 'complete'
                st.experimental_rerun()

            elif st.session_state.analysis_step == 'complete':
                # Show final results
                result = st.session_state.results
                
                st.write("### Final Analysis Results")
                
                if "error" not in result:
                    # Display results based on type
                    if result.get("type") == "direct_sql":
                        st.write("#### SQL Query")
                        st.code(result["sql"], language="sql")
                        
                        st.write("#### Results")
                        df = pd.DataFrame(result["results"])
                        st.dataframe(df)
                        
                        # Add visualization if appropriate
                        if len(df) > 0 and 'date' in df.columns:
                            st.write("#### Visualization")
                            st.line_chart(df.set_index('date'))
                    else:
                        st.write("#### Analysis")
                        st.write(result["analysis"])
                        
                        st.write("#### Supporting Data")
                        df = pd.DataFrame(result["results"])
                        st.dataframe(df)
                        
                        # Add visualizations
                        if len(df) > 0:
                            st.write("#### Visualizations")
                            if 'date' in df.columns:
                                st.line_chart(df.set_index('date'))
                            
                            # Add correlation heatmap if multiple numeric columns
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 1:
                                st.write("Correlation Heatmap")
                                correlation = df[numeric_cols].corr()
                                st.dataframe(correlation.style.background_gradient())
                
                if st.button("Start New Analysis"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.experimental_rerun()
                    return result

                return result

            return {"status": "in_progress"}

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return {"error": str(e)}

    def _direct_sql_query(self, query: str) -> Dict:
        try:
            st.write("### Direct SQL Query Generation")
            st.info("Generating SQL query...")
            
            result = self.sql_agent.invoke({"input": query})
            sql = self._extract_sql(result['output'])
            
            # Display original SQL
            st.write("#### Generated SQL:")
            st.code(sql, language='sql')
            
            if self.config.human_in_the_loop:
                sql = self._get_human_input(
                    f"Review the generated SQL query:",
                    sql,
                    key=f"sql_{hash(query)}"
                )
            
            # Execute query
            st.info("Executing SQL query...")
            df = pd.read_sql_query(sql, self.conn)
            
            # Show results
            st.write("#### Query Results:")
            st.dataframe(df)
            
            return {
                "type": "direct_sql",
                "query": query,
                "sql": sql,
                "results": df.to_dict('records')
            }
        except Exception as e:
            st.error(f"SQL query error: {str(e)}")
            return {"error": str(e)}

    def _complex_analysis(self, query: str) -> Dict:
        try:
            st.write("### Complex Analysis")
            st.info("Performing detailed analysis...")
            
            # Generate SQL for analysis
            result = self.sql_agent.invoke({"input": query})
            sql = self._extract_sql(result['output'])
            
            # Display original SQL
            st.write("#### Generated SQL:")
            st.code(sql, language='sql')
            
            if self.config.human_in_the_loop:
                sql = self._get_human_input(
                    f"Review the analysis SQL query:",
                    sql,
                    key=f"analysis_{hash(query)}"
                )
            
            # Execute SQL
            st.info("Executing analysis query...")
            df = pd.read_sql_query(sql, self.conn)
            
            # Show intermediate results
            st.write("#### Query Results:")
            st.dataframe(df)
            
            # Generate analysis
            st.info("Generating expert analysis...")
            analysis = self._generate_analysis(df, query)
            
            if self.config.human_in_the_loop:
                analysis = self._get_human_input(
                    f"Review the generated analysis:",
                    analysis,
                    key=f"review_{hash(query)}"
                )
            
            return {
                "type": "analysis",
                "query": query,
                "sql": sql,
                "results": df.to_dict('records'),
                "analysis": analysis
            }
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return {"error": str(e)}

    def _generate_analysis(self, df: pd.DataFrame, query: str) -> str:
        try:
            # Prepare data summary for the prompt
            data_summary = df.describe().to_string()
            
            prompt = f"""Analyze this stock market data in response to the query: {query}

Data Summary:
{data_summary}

Raw Data:
{df.head(10).to_string()}

Provide a clear, actionable analysis focusing on:
1. Key trends and patterns
2. Notable statistics
3. Potential insights
4. Actionable recommendations"""

            response = self.llm.invoke([
                SystemMessage(content=ANALYST_PROMPT),
                HumanMessage(content=prompt)
            ])
            
            return response.content
        except Exception as e:
            st.error(f"Analysis generation error: {str(e)}")
            return f"Error generating analysis: {str(e)}"

    @staticmethod
    def _extract_sql(text: str) -> str:
        """Extract SQL query from agent output"""
        if "SQL:" in text:
            sql_part = text.split("SQL:")[1]
            if "SQLResult:" in sql_part:
                return sql_part.split("SQLResult:")[0].strip()
            if "Human:" in sql_part:
                return sql_part.split("Human:")[0].strip()
            return sql_part.strip()
        return ""

def main():
    st.set_page_config(
        page_title="Stock Market Data Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Stock Market Data Analyzer")
    st.write("Upload your stock market data and analyze it using natural language queries.")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        if 'ANTHROPIC_API_KEY' not in os.environ:
            api_key = st.text_input("Enter Anthropic API Key:", type="password")
            if api_key:
                os.environ['ANTHROPIC_API_KEY'] = api_key
                st.success("API Key set successfully!")
        else:
            st.success("API Key is configured")
        
        # Instructions
        st.header("📝 Instructions")
        st.markdown("""
        1. Upload a CSV file with stock data
        2. Enable/disable human review
        3. Enter your query
        4. Review and adjust results
        """)
        
        # Example queries
        st.header("💡 Example Queries")
        st.markdown("""
        - Show the last 5 days of trading
        - What's the average daily volume?
        - Find days with unusual price movements
        - Analyze price trends and patterns
        """)

    # File upload section
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None and not st.session_state.file_uploaded:
        try:
            # Save and process uploaded file
            with open("temp_stock_data.csv", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Show preview
            df = pd.read_csv("temp_stock_data.csv")
            
            st.write("### Preview of uploaded data")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("First few rows:")
                st.dataframe(df.head())
            
            with col2:
                st.write("Data Summary:")
                st.dataframe(df.describe())
            
            # Initialize database after file is uploaded
            StockAnalyzer.initialize_database("temp_stock_data.csv")
            
            st.session_state.file_uploaded = True
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return

    if st.session_state.file_uploaded:
        # Analysis options
        st.write("### Analysis Options")
        
        col1, col2 = st.columns(2)
        with col1:
            human_in_loop = st.checkbox("Enable Human-in-the-Loop Analysis", 
                                      help="Enable this to review and modify intermediate results")
        
        # Query input
        st.write("### Enter Your Query")
        query = st.text_area("What would you like to analyze?", 
                            height=100,
                            help="Enter your question about the stock data")
        
        if st.button("🔍 Analyze", type="primary"):
            if query:
                if 'ANTHROPIC_API_KEY' not in os.environ:
                    st.error("Please configure your Anthropic API Key in the sidebar first.")
                    return
                
                try:
                    # Initialize analyzer
                    config = Config(
                        db_path="temp_stock_data.csv",
                        human_in_the_loop=human_in_loop
                    )
                    
                    # Create analyzer instance
                    analyzer = StockAnalyzer(config)
                    
                    # Run analysis
                    with st.spinner("🔄 Analyzing your query..."):
                        result = analyzer.analyze(query)
                        
                        if "error" not in result:
                            if result.get("status") != "in_progress":
                                st.success("✅ Analysis complete!")
                                
                                # Save results
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                filename = f"analysis_results_{timestamp}.json"
                                with open(filename, 'w') as f:
                                    json.dump(result, f, indent=2)
                                
                                st.download_button(
                                    label="📥 Download Results",
                                    data=json.dumps(result, indent=2),
                                    file_name=filename,
                                    mime="application/json"
                                )
                        else:
                            st.error(f"❌ Error: {result['error']}")
                            
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a query")

        # Reset button
        if st.button("🔄 Reset"):
            # Clean up
            if os.path.exists("temp_stock_data.csv"):
                os.remove("temp_stock_data.csv")
            if os.path.exists("stock_data.db"):
                os.remove("stock_data.db")
            
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.success("✨ Reset complete! Upload a new file to start again.")
            st.experimental_rerun()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        st.warning("Application terminated by user")
        # Cleanup
        if os.path.exists("temp_stock_data.csv"):
            os.remove("temp_stock_data.csv")
        if os.path.exists("stock_data.db"):
            os.remove("stock_data.db")
        # Clear session state
        if 'session_state' in st.__dict__:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
    except Exception as e:
        # Handle any other exceptions
        st.error(f"An unexpected error occurred: {str(e)}")
        if os.path.exists("temp_stock_data.csv"):
            os.remove("temp_stock_data.csv")
        if os.path.exists("stock_data.db"):
            os.remove("stock_data.db")
        # Log the error
        with open("error_log.txt", "a") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: {str(e)}")
    finally:
        # Always ensure database connection is closed if it exists
        if 'conn' in locals():
            conn.close()
        # Clean up any temporary files that might have been created
        for file in ['temp_stock_data.csv', 'stock_data.db']:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass
        # Print completion message
        print("Application terminated")