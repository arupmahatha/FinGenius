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
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None
if 'agent' not in st.session_state:
    st.session_state.agent = None

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
    
    agent = create_sql_agent(
        llm=llm,
        db=db_connection,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    st.session_state.agent = agent
    return agent

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
    # Initialize the AI agent if not already initialized
    if st.session_state.agent is None:
        agent = initialize_agent(api_key, model_choice, st.session_state.db_connection)
    else:
        agent = st.session_state.agent

    # Chat interface
    st.markdown("### Chat with your Database Analyst")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sql" in message:
                st.code(message["sql"], language="sql")

    # Chat input
    if prompt := st.chat_input("Ask me about your database..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent.run(prompt)
                    
                    # Extract SQL query if present
                    sql_query = None
                    if "SELECT" in response.upper() or "FROM" in response.upper():
                        sql_start = response.upper().find("SELECT")
                        if sql_start != -1:
                            sql_query = response[sql_start:].split("\n")[0]
                    
                    st.markdown(response)
                    if sql_query:
                        st.code(sql_query, language="sql")
                    
                    # Add assistant response to chat history
                    message_data = {
                        "role": "assistant",
                        "content": response
                    }
                    if sql_query:
                        message_data["sql"] = sql_query
                    st.session_state.messages.append(message_data)
                    
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Powered by LangChain, OpenAI, and Anthropic")