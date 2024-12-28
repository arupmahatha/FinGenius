import streamlit as st
import pandas as pd
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import sqlite3

# Page configuration
st.set_page_config(page_title="AI Database Analyst", layout="wide")

# Session state initialization
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None

def initialize_agent(api_key, model_choice, db_connection):
    if model_choice == "GPT-4":
        llm = ChatOpenAI(temperature=0, model_name="gpt-4", api_key=api_key)
    elif model_choice == "GPT-3.5":
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=api_key)
    elif model_choice == "Claude":
        llm = ChatAnthropic(
            temperature=0,
            model="claude-3-sonnet-20240229",
            api_key=api_key
        )
    
    return create_sql_agent(
        llm=llm,
        db=db_connection,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Enter API Key", type="password")
    model_choice = st.selectbox(
        "Select AI Model",
        ["GPT-4", "GPT-3.5", "Claude"]
    )
    
    uploaded_file = st.file_uploader("Upload Database (SQLite)", type=['db'])
    
    if uploaded_file is not None:
        with open("temp_db.db", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.db_connection = SQLDatabase.from_uri("sqlite:///temp_db.db")
        st.success("Database loaded successfully!")

# Main content
st.title("AI Database Analyst")

if not api_key:
    st.warning("Please enter your API key in the sidebar.")
elif st.session_state.db_connection is None:
    st.warning("Please upload a database file.")
else:
    # Initialize the AI agent
    agent = initialize_agent(api_key, model_choice, st.session_state.db_connection)
    
    # Query input
    user_input = st.text_area("Enter your analysis question:", height=100)
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            try:
                response = agent.run(user_input)
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "question": user_input,
                    "answer": response
                })
                
                # Display response in a more structured way
                st.success("Analysis Complete!")
                
                # Create an expander for the current analysis
                with st.expander("Current Analysis Result", expanded=True):
                    st.markdown("### Question")
                    st.info(user_input)
                    
                    st.markdown("### Analysis Result")
                    st.markdown(response)
                    
                    # If the response contains any SQL queries, try to extract and display them
                    if "SELECT" in response.upper() or "FROM" in response.upper():
                        st.markdown("### SQL Query Used")
                        # Extract and display SQL query if present
                        sql_start = response.upper().find("SELECT")
                        if sql_start != -1:
                            query = response[sql_start:].split("\n")[0]
                            st.code(query, language="sql")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Previous Analyses")
        for idx, interaction in enumerate(st.session_state.conversation_history):
            with st.expander(f"Q{idx + 1}: {interaction['question'][:50]}..."):
                st.write("Question:", interaction['question'])
                st.write("Answer:", interaction['answer'])

# Footer
st.markdown("---")
st.markdown("Powered by LangChain, OpenAI, and Anthropic")
