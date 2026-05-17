"""Streamlit web application for KnowledgeNexus."""

import os
import streamlit as st
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv

load_dotenv()

from src.agents import KnowledgeNexusAgent
from src.vector_store import VectorStoreManager
from src.document_processor import DocumentProcessor

st.set_page_config(
    page_title="KnowledgeNexus",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    'primary': "#9BC0E7",
    'secondary': "#3F5C91",
    'accent': '#E0A458',
    'surface': "#355980",
    'text_primary': '#E0E1DD',
    'text_secondary': '#778DA9',
    'success': '#2DD4BF',
    'error': '#F87171'
}

# Load external CSS
css_path = Path(__file__).parent / "styles" / "main.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def init_agent():
    """Initialize the KnowledgeNexus agent."""
    vector_store = VectorStoreManager(persist_directory="./chroma_db")
    processor = DocumentProcessor(chunk_size=1000, overlap=200)
    return KnowledgeNexusAgent(vector_store, processor)


def main():
    agent = init_agent()

    with st.sidebar:
        st.markdown(f"""
            <div style="padding: 20px 0 16px 0; border-bottom: 1px solid {COLORS['surface']}; margin-bottom: 16px;">
                <h2 style="color: {COLORS['accent']}; margin: 0; font-size: 1.4rem;">📚 Document Library</h2>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("📤 Upload Documents", expanded=True):
            uploaded_files = st.file_uploader(
                "Drag & drop files here",
                type=['pdf', 'txt', 'csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Supported: PDF, TXT, CSV, Excel"
            )

            if uploaded_files:
                st.markdown(f"<p style='color: {COLORS['text_secondary']}; font-size: 0.85rem;'>{len(uploaded_files)} file(s) ready</p>", unsafe_allow_html=True)

                if st.button("⚡ Process Documents", type="primary", use_container_width=True):
                    with st.spinner("📄 Processing..."):
                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=Path(uploaded_file.name).suffix
                            ) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name

                            try:
                                result = agent.ingest_document(tmp_path, uploaded_file.name)
                            finally:
                                # Try to delete temp file, with retry logic for Windows
                                import time
                                for _ in range(3):
                                    try:
                                        os.unlink(tmp_path)
                                        break
                                    except PermissionError:
                                        time.sleep(0.1)

                            ext = Path(uploaded_file.name).suffix.lower().replace('.', '')
                            if result['status'] == 'success':
                                st.markdown(f"""
                                    <div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: rgba(45, 212, 191, 0.1); border-radius: 8px; margin: 8px 0;">
                                        <span style="color: {COLORS['success']};">✓</span>
                                        <span style="color: {COLORS['text_primary']}; font-size: 0.85rem;">{uploaded_file.name}</span>
                                        <span style="color: {COLORS['accent']}; font-size: 0.75rem; margin-left: auto;">{result['chunks']} chunks</span>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: rgba(248, 113, 113, 0.1); border-radius: 8px; margin: 8px 0;">
                                        <span style="color: {COLORS['error']};">✗</span>
                                        <span style="color: {COLORS['text_primary']}; font-size: 0.85rem;">{uploaded_file.name}</span>
                                    </div>
                                """, unsafe_allow_html=True)

        status = agent.get_status()
        st.markdown(f"""
            <div class="status-card">
                <h4>📊 System Status</h4>
                <p>Documents indexed: <strong>{status.get('documents_indexed', 0)}</strong></p>
                <p>Status: <strong style="color: {COLORS['success']};">{status.get('status', 'unknown').upper()}</strong></p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Knowledge Base", type="secondary", use_container_width=True):
            result = agent.clear_knowledge_base()
            if result['status'] == 'success':
                st.success("✅ Knowledge base cleared!")
                st.rerun()
            else:
                st.error(f"Error: {result.get('error')}")

        st.markdown("---")
        st.markdown(f"""
            <div style="padding: 16px; background: rgba(65, 90, 119, 0.2); border-radius: 12px;">
                <p style="color: {COLORS['accent']}; font-weight: 600; margin: 0 0 12px 0; font-size: 0.85rem;">💡 Tips</p>
                <ul style="color: {COLORS['text_secondary']}; font-size: 0.8rem; margin: 0; padding-left: 16px; line-height: 1.8;">
                    <li>Upload documents first</li>
                    <li>Use specific questions</li>
                    <li>Check citations for sources</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="main-header">
            <h1>🧠 KnowledgeNexus</h1>
            <p class="subtitle">Multi-agent AI knowledge platform with semantic retrieval</p>
        </div>
    """, unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'history' not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.messages:
        role = msg.get('role', 'assistant')
        content = msg.get('content', '')
        citations = msg.get('citations', [])

        with st.chat_message(role):
            if citations:
                text_content = content
                st.markdown(text_content, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="sources-container">
                        <span class="sources-label">Sources</span>
                        {''.join([f'<span class="citation-badge">📄 {c}</span>' for c in citations])}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(content)

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            with st.spinner("🔍 Searching & generating..."):
                result = agent.query(prompt, st.session_state.history)

                response = result.get('response', 'No response generated')
                citations = []

                if result.get('retrieved_docs'):
                    # Use set to deduplicate sources - same file should only appear once
                    seen_sources = set()
                    for doc in result['retrieved_docs'][:5]:
                        source = doc.get('metadata', {}).get('source', 'Unknown')
                        if source not in seen_sources:
                            seen_sources.add(source)
                            citations.append(source)

                st.markdown(response, unsafe_allow_html=True)

                if citations:
                    st.markdown(f"""
                        <div class="sources-container">
                            <span class="sources-label">Sources</span>
                            {''.join([f'<span class="citation-badge">📄 {c}</span>' for c in citations])}
                        </div>
                    """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response,
                    'citations': citations
                })

                st.session_state.history.extend([
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': response}
                ])

    # Data Visualization Section
    st.markdown("---")
    with st.expander("📈 Data Visualization & Analysis", expanded=False):
        analysis_file = st.file_uploader(
            "Upload CSV/Excel for analysis",
            type=['csv', 'xlsx', 'xls'],
            key="analysis_file"
        )

        if analysis_file:
            try:
                if analysis_file.name.endswith('.csv'):
                    df = pd.read_csv(analysis_file)
                else:
                    df = pd.read_excel(analysis_file)

                st.markdown(f"**File:** {analysis_file.name} | **Rows:** {len(df)} | **Columns:** {len(df.columns)}")

                # Show data preview (no expander to avoid nesting issue)
                st.markdown("### 📋 Data Preview")
                st.dataframe(df.head(20), use_container_width=True)

                # Column selector for charts
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                col1, col2 = st.columns(2)

                with col1:
                    chart_type = st.selectbox(
                        "Chart Type",
                        ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram"]
                    )

                with col2:
                    if chart_type in ["Bar Chart", "Line Chart", "Pie Chart"]:
                        x_col = st.selectbox("Category Column", df.columns.tolist())
                    else:
                        x_col = st.selectbox("X Column", df.columns.tolist())

                # Y column for appropriate charts
                y_col = None
                if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                    y_col = st.selectbox("Y Column (Numeric)", numeric_cols if numeric_cols else df.columns.tolist())

                # Generate chart
                if st.button("Generate Chart", type="primary"):
                    try:
                        if chart_type == "Bar Chart":
                            fig = px.bar(df, x=x_col, y=y_col, title=f"{chart_type}: {x_col} vs {y_col}", barmode='group')
                            fig.update_traces(texttemplate='%{y}', textposition='outside')
                        elif chart_type == "Line Chart":
                            fig = px.line(df, x=x_col, y=y_col, title=f"{chart_type}: {x_col} vs {y_col}", markers=True)
                            fig.update_traces(texttemplate='%{y}', textposition='top center')
                        elif chart_type == "Pie Chart":
                            fig = px.pie(df, names=x_col, title=f"{chart_type}: {x_col}")
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                        elif chart_type == "Scatter Plot":
                            fig = px.scatter(df, x=x_col, y=y_col, title=f"{chart_type}: {x_col} vs {y_col}", mode='markers+text')
                            fig.update_traces(texttemplate='%{y}', textposition='top center', textfont_size=10)
                        elif chart_type == "Histogram":
                            fig = px.histogram(df, x=x_col, title=f"{chart_type}: {x_col}")
                            fig.update_traces(marker_line_width=1, marker_line_color='white')

                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#E0E1DD",
                            title_font_color="#E0A458",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Show statistics
                        if numeric_cols:
                            st.markdown("### 📊 Statistics")
                            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

                    except Exception as e:
                        st.error(f"Error generating chart: {str(e)}")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    if st.session_state.messages:
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()


if __name__ == "__main__":
    main()