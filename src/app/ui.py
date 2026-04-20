import streamlit as st
import requests

st.title("Enterprise Search Query Understanding")
st.write("Powered by Semantic Caching and Distilled LLMs")

query = st.text_input("Enter your search query (e.g., 'best coffee near central park'):")

if st.button("Parse Query"):
    if query:
        with st.spinner("Processing..."):
            try:
                # Target the FastAPI service
                res = requests.post(
                    "http://localhost:8000/parse_query", 
                    json={"query": query, "user_location": "local"}
                )
                data = res.json()
                st.success(f"Source: {data['source']}")
                st.json(data['extracted_entities'])
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")