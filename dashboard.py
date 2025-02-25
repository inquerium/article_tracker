import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

from article_tracker_simple import ArticleTracker
from config import DB_CONNECTION, EXPORT_DIR, DEPARTMENTS

# Initialize the tracker
@st.cache_resource
def get_tracker():
    return ArticleTracker(DB_CONNECTION, EXPORT_DIR)

def main():
    st.set_page_config(
        page_title="Company Article Tracker",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    tracker = get_tracker()
    
    # Sidebar navigation
    st.sidebar.title("Article Tracking System")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Submit Article", "Recent Articles", "Search Articles", "Keyword Analysis", "Reports"]
    )
    
    # Reset submission flow when changing pages
    if page != "Submit Article" and "submission_step" in st.session_state:
        del st.session_state.submission_step
    
    # Page content
    if page == "Submit Article":
        display_article_submission_form(tracker)
    elif page == "Recent Articles":
        display_recent_articles(tracker)
    elif page == "Search Articles":
        display_search(tracker)
    elif page == "Keyword Analysis":
        display_keyword_analysis(tracker)
    elif page == "Reports":
        display_reports(tracker)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Company Article Tracking System")

def display_article_submission_form(tracker):
    st.title("Submit New Article")
    
    # Initialize submission step if not already set
    if "submission_step" not in st.session_state:
        st.session_state.submission_step = 1
    
    # Option to reset form
    #if st.sidebar.button("Reset Form"):
        #st.session_state.submission_step = 1
        #st.experimental_rerun()
    
    # Step 1: Article Information Form
    if st.session_state.submission_step == 1:
        st.write("Enter the article details below.")
        
        # Pre-fill logic - add this block
        prefill = {}
        if "article_data" in st.session_state:
            prefill = st.session_state.article_data
        
        with st.form("article_form"):
            title = st.text_input("Article Title", 
                                value=prefill.get("title", ""), 
                                help="Enter the full title of the article")
            
            content = st.text_area(
                "Article Content", 
                value=prefill.get("content", ""),
                height=300, 
                help="Copy and paste the full article content here"
            )
            
            keyword_input = st.text_input(
                "Keywords (comma-separated)",
                value=", ".join(prefill.get("keywords", [])) if "keywords" in prefill else "",
                help="Enter keywords separated by commas (e.g., AI, machine learning, data)"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                author = st.text_input("Author", help="Name of the article author")
                url = st.text_input("Article URL (optional)", help="Link to the published article")
            
            with col2:
                department = st.selectbox("Department", DEPARTMENTS)
                publish_date = st.date_input(
                    "Publication Date", 
                    datetime.now().date(),
                    help="When was this article published?"
                )
            
            check_button = st.form_submit_button("Check for Duplicates")
        
        if check_button:
            if not title or not content or not keyword_input or not author:
                st.error("Please fill in all required fields: title, content, keywords, and author.")
                return
            
            # Process keywords
            keyword_list = [k.strip() for k in keyword_input.split(",") if k.strip()]
            
            if not keyword_list:
                st.error("Please enter at least one keyword.")
                return
            
            # Store form data in session state
            st.session_state.article_data = {
                "title": title,
                "content": content,
                "author": author,
                "department": department,
                "publish_date": publish_date,
                "keywords": keyword_list,
                "url": url
            }
            
            # Check for duplicates
            with st.spinner("Checking for duplicates..."):
                duplicate_check = tracker.check_duplicates(title, content, keyword_list)
            
            # Store result and move to next step
            st.session_state.duplicate_result = duplicate_check
            st.session_state.submission_step = 2
            st.experimental_rerun()
    
    # Step 2: Duplicate Check Results
    elif st.session_state.submission_step == 2:
        st.write("Review duplicate check results.")
        
        duplicate_result = st.session_state.duplicate_result
        article_data = st.session_state.article_data
        
        # Display article info summary
        with st.expander("Article Information", expanded=False):
            st.write(f"**Title:** {article_data['title']}")
            st.write(f"**Author:** {article_data['author']}")
            st.write(f"**Department:** {article_data['department']}")
            st.write(f"**Keywords:** {', '.join(article_data['keywords'])}")
        
        # Show duplicate results
        if duplicate_result["found_duplicates"]:
            st.error("⚠️ Potential duplicate detected!")
            
            tab1, tab2, tab3 = st.tabs(["Title Duplicates", "Keyword Duplicates", "Content Duplicates"])
            
            with tab1:
                if duplicate_result["title_duplicates"]:
                    st.warning("The following articles have similar titles:")
                    for dup in duplicate_result["title_duplicates"]:
                        st.info(f"• {dup['title']} (ID: {dup['id']})")
                else:
                    st.success("No title duplicates found.")
            
            with tab2:
                if duplicate_result["keyword_duplicates"]:
                    st.warning("Articles with the same keywords exist:")
                    for dup in duplicate_result["keyword_duplicates"]:
                        matching_kw = ", ".join(dup["matching_keywords"])
                        st.info(f"• {dup['title']} (ID: {dup['id']}) - Matching keywords: {matching_kw}")
                else:
                    st.success("No keyword duplicates found.")
            
            with tab3:
                if duplicate_result["content_duplicates"]:
                    st.warning("Articles with similar content exist:")
                    for dup in duplicate_result["content_duplicates"]:
                        similarity = f"{dup['similarity']*100:.1f}%"
                        st.info(f"• {dup['title']} (ID: {dup['id']}) - Similarity: {similarity}")
                else:
                    st.success("No content duplicates found.")
            
            st.warning("The article appears to be a duplicate. Submit only if you're sure this is a new article.")
        else:
            st.success("No duplicates found! You can submit this article.")
        
        # Submission buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Back to Edit"):
                st.session_state.submission_step = 1
                st.experimental_rerun()
        
        with col2:
            if st.button("Submit Article", key="submit_actual"):
                st.session_state.submission_step = 3
                st.experimental_rerun()
    
    # Step 3: Submit Article
    elif st.session_state.submission_step == 3:
        st.write("Submitting article...")
        
        article_data = st.session_state.article_data
        
        with st.spinner("Submitting article to database..."):
            result = tracker.add_article(
                article_data["title"],
                article_data["content"],
                article_data["author"],
                article_data["department"],
                article_data["publish_date"],
                article_data["keywords"],
                article_data["url"]
            )
        
        if result["success"]:
            st.success(f"✅ Article submitted successfully (ID: {result['article_id']})")
            st.balloons()
            
            if st.button("Submit Another Article"):
                st.session_state.submission_step = 1
                st.experimental_rerun()
        else:
            st.error(f"Error: {result['message']}")
            
            if st.button("Try Again"):
                st.session_state.submission_step = 2
                st.experimental_rerun()

def display_recent_articles(tracker):
    st.title("Recent Articles")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        days = st.slider("Days", 1, 90, 30, help="Show articles from the last N days")
    with col2:
        department_filter = st.selectbox(
            "Department", 
            ["All"] + DEPARTMENTS,
            help="Filter by department"
        )
    with col3:
        search = st.text_input("Filter by title", help="Enter title keywords to filter")
    
    # Get recent articles
    try:
        df = tracker.get_recent_articles(days=days, department=department_filter if department_filter != "All" else None)
        
        # Apply title filter if provided
        if search and not df.empty:
            df = df[df['title'].str.contains(search, case=False)]
        
        # Format the dataframe
        if not df.empty:
            # Convert keywords list to comma-separated string
            df['keywords'] = df['keywords'].apply(lambda x: ', '.join(x) if x else '')
            
            # Format dates
            df['publish_date'] = pd.to_datetime(df['publish_date']).dt.strftime('%Y-%m-%d')
            
            # Show the articles
            st.dataframe(
                df[['id', 'title', 'author', 'department', 'publish_date', 'keywords']],
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "title": st.column_config.TextColumn("Title", width="large"),
                    "keywords": st.column_config.TextColumn("Keywords", width="medium"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                "recent_articles.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("No articles found with the current filters.")
    except Exception as e:
        st.error(f"Error retrieving articles: {e}")

def display_search(tracker):
    st.title("Search Articles")
    
    # Search form
    search_query = st.text_input("Enter search terms", help="Search articles by content, title, or keywords")
    
    if search_query:
        try:
            with st.spinner("Searching..."):
                results = tracker.search_articles(search_query.replace(' ', ' & '))
            
            if results:
                # Convert to DataFrame for display
                df = pd.DataFrame(results)
                df['publish_date'] = pd.to_datetime(df['publish_date']).dt.strftime('%Y-%m-%d')
                
                # Show results
                st.success(f"Found {len(results)} results")
                st.dataframe(
                    df[['id', 'title', 'author', 'publish_date', 'relevance']],
                    column_config={
                        "id": st.column_config.NumberColumn("ID", width="small"),
                        "title": st.column_config.TextColumn("Title", width="large"),
                        "relevance": st.column_config.ProgressColumn(
                            "Relevance", 
                            width="medium",
                            format="%.2f",
                            min_value=0,
                            max_value=1
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No articles found matching your search terms.")
        except Exception as e:
            st.error(f"Search error: {e}")

def display_keyword_analysis(tracker):
    st.title("Keyword Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        days = st.slider("Timeframe (days)", 7, 180, 30, help="Analyze keywords from the past N days")
        limit = st.slider("Number of keywords", 5, 50, 20, help="Show top N keywords")
    
    try:
        with st.spinner("Analyzing keywords..."):
            trending = tracker.get_trending_keywords(days=days, limit=limit)
        
        if trending:
            # Convert to DataFrame for plotting
            df = pd.DataFrame(trending)
            
            # Create bar chart with Plotly
            fig = px.bar(
                df, 
                x='keyword', 
                y='count', 
                title=f'Top {limit} Keywords (Past {days} Days)',
                labels={'count': 'Article Count', 'keyword': 'Keyword'},
                color='count',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show tabular data
            with st.expander("View data table"):
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No keyword data available for the selected timeframe.")
    except Exception as e:
        st.error(f"Error analyzing keywords: {e}")

def display_reports(tracker):
    st.title("Executive Reports")
    
    # Report types
    report_type = st.selectbox(
        "Report Type",
        ["Weekly Article Summary", "Department Activity", "Keyword Trends", "Export All Articles"]
    )
    
    if report_type == "Weekly Article Summary":
        # Generate on-demand
        if st.button("Generate Weekly Report"):
            with st.spinner("Generating report..."):
                report = tracker.generate_weekly_report()
            
            st.success(f"Report generated with {report['article_count']} articles")
            
            # Display summary
            st.subheader("Summary")
            st.write(f"Total articles: {report['article_count']}")
            
            if report['top_keywords']:
                st.write("Top keywords: " + ", ".join(report['top_keywords']))
            
            # Department summary
            if report['department_summary']:
                dept_df = pd.DataFrame(report['department_summary'])
                
                fig = px.pie(
                    dept_df, 
                    values='article_count', 
                    names='department',
                    title='Articles by Department'
                )
                st.plotly_chart(fig)
    
    elif report_type == "Export All Articles":
        format_option = st.selectbox("Export Format", ["Excel", "CSV", "JSON"])
        
        if st.button(f"Export as {format_option}"):
            with st.spinner(f"Exporting all articles as {format_option}..."):
                filepath = tracker.export_all_articles(format=format_option.lower())
            
            st.success(f"Exported to {filepath}")
            st.info("You can find this file in your export directory.")

if __name__ == "__main__":
    main()