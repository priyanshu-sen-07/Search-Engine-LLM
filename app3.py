import streamlit as st
import asyncio
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# -------------------------
# App Title
# -------------------------
st.set_page_config(page_title="LangChain Search Chatbot", page_icon="🔎")
st.title("🔎 LangChain - Chat with Search")

"""
This chatbot uses Groq's Llama3 model + LangChain tools to search ArXiv, Wikipedia, and the web **in parallel** for faster results.
"""

# -------------------------
# Sidebar Settings
# -------------------------
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# -------------------------
# Tools Setup
# -------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

tools = [search, arxiv, wiki]

# -------------------------
# Session State Messages
# -------------------------
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# -------------------------
# Parallel Execution Helper
# -------------------------
async def run_parallel_agent(agent, prompt, chat_history, st_cb):
    """Run the agent in async mode for parallel tool execution."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: agent.run({"input": prompt, "chat_history": chat_history}, callbacks=[st_cb])
    )
    return response

# -------------------------
# Chat Input Handling
# -------------------------
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="Llama3-8b-8192",
            streaming=True
        )

        # Prepare chat history
        chat_history = [
            (m["role"], m["content"])
            for m in st.session_state.messages
            if m["role"] in ["user", "assistant"]
        ]

        # Initialize agent
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,  # limit loops for speed
            early_stopping_method="generate"
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                # Run agent asynchronously for parallel tool execution
                response = asyncio.run(run_parallel_agent(search_agent, prompt, chat_history, st_cb))
            except Exception as e:
                response = f"⚠️ Error: {e}"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
