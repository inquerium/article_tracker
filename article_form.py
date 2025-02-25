import streamlit as st
import pandas as pd
from datetime import datetime
from article_tracker_simple import ArticleTracker
from config import DB_CONNECTION, EXPORT_DIR, DEPARTMENTS

def main():
    st.title("Article Submission Form")
    
    # Initialize tracker
    tracker = ArticleTracker(DB_CONNECTION, EXPORT_DIR)
    
    # Step 1: Enter article information
    st.header("Step 1: Enter Article Information")
    
    title = st.text_input("Article Title")
    content = st.text_area("Article Content", height=300)
    keywords = st.text_input("Keywords (comma-separated)")
    author = st.text_input("Author")
    department = st.selectbox("Department", DEPARTMENTS)
    publish_date = st.date_input("Publish Date", datetime.now().date())
    url = st.text_input("URL (optional)")
    
    # Step 2: Check for duplicates
    if st.button("Check for Duplicates"):
        if not title or not content or not keywords or not author:
            st.error("Please fill all required fields")
            return
        
        # Process keywords
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
        
        # Store data in session state
        st.session_state.article_data = {
            "title": title,
            "content": content,
            "keywords": keyword_list,
            "author": author,
            "department": department,
            "publish_date": publish_date,
            "url": url
        }
        
        # Check for duplicates
        duplicate_check = tracker.check_duplicates(title, content, keyword_list)
        st.session_state.duplicate_result = duplicate_check
        
        # Set the checked flag
        st.session_state.duplicates_checked = True
        
        # Refresh to show results
        st.experimental_rerun()
    
    # Step 3: Show duplicate check results and submit option
    if 'duplicates_checked' in st.session_state and st.session_state.duplicates_checked:
        st.header("Step 2: Duplicate Check Results")
        
        duplicate_result = st.session_state.duplicate_result
        
        if duplicate_result["found_duplicates"]:
            st.error("Potential duplicates found!")
            
            # Show duplicates
            if duplicate_result["title_duplicates"]:
                st.subheader("Similar Titles")
                for dup in duplicate_result["title_duplicates"]:
                    st.write(f"• {dup['title']} (ID: {dup['id']})")
            
            if duplicate_result["keyword_duplicates"]:
                st.subheader("Matching Keywords")
                for dup in duplicate_result["keyword_duplicates"]:
                    matching_kw = ", ".join(dup["matching_keywords"])
                    st.write(f"• {dup['title']} (ID: {dup['id']}) - Matching keywords: {matching_kw}")
            
            if duplicate_result["content_duplicates"]:
                st.subheader("Similar Content")
                for dup in duplicate_result["content_duplicates"]:
                    similarity = f"{dup['similarity']*100:.1f}%"
                    st.write(f"• {dup['title']} (ID: {dup['id']}) - Similarity: {similarity}")
                    
            st.warning("Article appears to be a duplicate. Continue only if you're sure it's new.")
        else:
            st.success("No duplicates found!")
        
        st.header("Step 3: Submit Article")
        
        # Submit button
        if st.button("Submit Article"):
            data = st.session_state.article_data
            
            result = tracker.add_article(
                data["title"],
                data["content"],
                data["author"],
                data["department"],
                data["publish_date"],
                data["keywords"],
                data["url"]
            )
            
            if result["success"]:
                st.balloons()
                st.success(f"Article submitted successfully! (ID: {result['article_id']})")
                
                # Clear session state
                if 'duplicates_checked' in st.session_state:
                    del st.session_state.duplicates_checked
                if 'duplicate_result' in st.session_state:
                    del st.session_state.duplicate_result
                if 'article_data' in st.session_state:
                    del st.session_state.article_data
                    
                st.button("Submit Another Article", on_click=lambda: st.experimental_rerun())
            else:
                st.error(f"Error submitting article: {result['message']}")

if __name__ == "__main__":
    main()