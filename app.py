#!/usr/bin/env python3
"""
THADS AI Agent - Streamlit Web Application
A web interface for testing Knowledge Base integration with PDF processing
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.database.knowledge_base import KnowledgeBase
    from src.database.models import DocumentType
    from src.database.integration import PDFProcessingIntegration
    from src.database.connection import get_database_connection
    from src.utils.pdf_processing.legal_document_processor import LegalDocumentProcessor
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def main():
    st.set_page_config(
        page_title="THADS AI Agent - Legal Document Analysis",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🏛️ THADS AI Agent - Legal Document Analysis")
    st.markdown("Upload and analyze Vietnamese legal documents with AI-powered extraction")
    
    # Initialize components
    try:
        db_connection = get_database_connection()
        kb = KnowledgeBase(db_connection)
        processor = LegalDocumentProcessor()
        
        st.success("✅ Database connection established successfully")
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {str(e)}")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📤 Upload & Process",
        "🔍 Search Documents", 
        "📊 Knowledge Base Stats",
        "🗃️ Document Library"
    ])
    
    if page == "📤 Upload & Process":
        upload_and_process_page(kb, processor)
    elif page == "🔍 Search Documents":
        search_documents_page(kb)
    elif page == "📊 Knowledge Base Stats":
        knowledge_base_stats_page(kb)
    elif page == "🗃️ Document Library":
        document_library_page(kb)

def upload_and_process_page(kb, processor):
    st.header("📤 Upload & Process Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=['pdf'],
        help="Upload Vietnamese legal documents (judgments, decisions, etc.)"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process the document
                    result = processor.process_document(tmp_path)
                    
                    if result and result.get('success'):
                        st.success("✅ Document processed successfully!")
                        
                        # Display processing results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📋 Document Info")
                            st.write(f"**Type:** {result.get('document_type', 'Unknown')}")
                            st.write(f"**Case Number:** {result.get('case_number', 'N/A')}")
                            st.write(f"**Court:** {result.get('court_name', 'N/A')}")
                            st.write(f"**Processing Time:** {result.get('processing_time', 0):.2f}s")
                        
                        with col2:
                            st.subheader("📊 Analysis Stats")
                            st.write(f"**Text Length:** {result.get('text_length', 0):,} chars")
                            st.write(f"**Word Count:** {result.get('word_count', 0):,}")
                            st.write(f"**Confidence:** {result.get('confidence_score', 0):.1%}")
                        
                        # Show extracted entities
                        if result.get('entities'):
                            st.subheader("🏷️ Extracted Entities")
                            entities_df = st.dataframe(result['entities'])
                        
                        # Show extracted text preview
                        if result.get('raw_text'):
                            st.subheader("📝 Extracted Text (Preview)")
                            st.text_area("", result['raw_text'][:1000] + "..." if len(result['raw_text']) > 1000 else result['raw_text'], height=200)
                    
                    else:
                        st.error("❌ Failed to process document")
                        if result and result.get('error'):
                            st.error(f"Error: {result['error']}")
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")

def search_documents_page(kb):
    st.header("🔍 Search Documents")
    
    search_query = st.text_input("Enter search query", placeholder="e.g., 'bản án hình sự', 'tòa án', 'bị cáo'")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        search_type = st.selectbox("Search Type", ["Full Text", "Entity", "Case Number"])
    with col2:
        limit = st.number_input("Max Results", min_value=1, max_value=100, value=10)
    with col3:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.1)
    
    if st.button("🔍 Search", type="primary") and search_query:
        with st.spinner("Searching..."):
            try:
                results = kb.search_documents(
                    query=search_query,
                    limit=limit,
                    min_confidence=min_confidence
                )
                
                if results:
                    st.success(f"✅ Found {len(results)} results")
                    
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"📄 {doc.get('filename', f'Document {i}')} (Score: {doc.get('relevance_score', 0):.2f})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Type:** {doc.get('document_type', 'Unknown')}")
                                st.write(f"**Case Number:** {doc.get('case_number', 'N/A')}")
                                st.write(f"**Court:** {doc.get('court_name', 'N/A')}")
                            
                            with col2:
                                st.write(f"**Date:** {doc.get('date_issued', 'N/A')}")
                                st.write(f"**Confidence:** {doc.get('confidence_score', 0):.1%}")
                                st.write(f"**Text Length:** {doc.get('text_length', 0):,}")
                            
                            if doc.get('raw_text'):
                                st.text_area("Preview", doc['raw_text'][:500] + "..." if len(doc['raw_text']) > 500 else doc['raw_text'], height=100, key=f"preview_{i}")
                else:
                    st.warning("🔍 No results found")
                    
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")

def knowledge_base_stats_page(kb):
    st.header("📊 Knowledge Base Statistics")
    
    try:
        stats = kb.get_statistics()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Documents", stats['documents']['total'])
        with col2:
            st.metric("🏷️ Total Entities", stats['entities']['total'])
        with col3:
            st.metric("⏱️ Avg Processing Time", f"{stats['documents'].get('avg_processing_time', 0):.2f}s")
        with col4:
            st.metric("🎯 Avg Confidence", f"{stats['documents'].get('avg_confidence_score', 0):.1%}")
        
        # Document types breakdown
        if stats['documents'].get('by_type'):
            st.subheader("📊 Documents by Type")
            doc_types = stats['documents']['by_type']
            st.bar_chart(doc_types)
        
        # Entity types breakdown
        if stats['entities'].get('by_type'):
            st.subheader("🏷️ Entities by Type")
            entity_types = stats['entities']['by_type']
            st.bar_chart(entity_types)
        
        # Database info
        st.subheader("🗄️ Database Information")
        db_info = stats.get('database_info', {})
        if 'error' not in db_info:
            for key, value in db_info.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.warning(f"Database info error: {db_info['error']}")
            
    except Exception as e:
        st.error(f"❌ Error loading statistics: {str(e)}")

def document_library_page(kb):
    st.header("🗃️ Document Library")
    
    try:
        # Get all documents
        documents = kb.get_all_documents()
        
        if documents:
            st.success(f"📚 Found {len(documents)} documents in library")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                doc_types = list(set([doc.get('document_type', 'Unknown') for doc in documents]))
                selected_type = st.selectbox("Filter by Type", ["All"] + doc_types)
            
            with col2:
                courts = list(set([doc.get('court_name', 'Unknown') for doc in documents if doc.get('court_name')]))
                selected_court = st.selectbox("Filter by Court", ["All"] + courts)
            
            with col3:
                sort_by = st.selectbox("Sort by", ["Date", "Filename", "Confidence", "Processing Time"])
            
            # Apply filters
            filtered_docs = documents
            if selected_type != "All":
                filtered_docs = [doc for doc in filtered_docs if doc.get('document_type') == selected_type]
            if selected_court != "All":
                filtered_docs = [doc for doc in filtered_docs if doc.get('court_name') == selected_court]
            
            # Display documents
            for i, doc in enumerate(filtered_docs, 1):
                with st.expander(f"📄 {doc.get('filename', f'Document {i}')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {doc.get('document_type', 'Unknown')}")
                        st.write(f"**Case Number:** {doc.get('case_number', 'N/A')}")
                        st.write(f"**Court:** {doc.get('court_name', 'N/A')}")
                        st.write(f"**Date Issued:** {doc.get('date_issued', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Processing Status:** {doc.get('processing_status', 'Unknown')}")
                        st.write(f"**Confidence:** {doc.get('confidence_score', 0):.1%}")
                        st.write(f"**Text Length:** {doc.get('text_length', 0):,}")
                        st.write(f"**Created:** {doc.get('created_at', 'N/A')}")
        else:
            st.info("📭 No documents found in library")
            
    except Exception as e:
        st.error(f"❌ Error loading document library: {str(e)}")

if __name__ == "__main__":
    main()