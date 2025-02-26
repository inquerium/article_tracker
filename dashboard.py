import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import json
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import hashlib
import plotly.express as px

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

# ---- Configuration (directly in the script) ----
# Get configuration from Streamlit secrets
if 'postgres' in st.secrets:
    DB_HOST = st.secrets.postgres.host
    DB_PORT = st.secrets.postgres.port
    DB_NAME = st.secrets.postgres.dbname
    DB_USER = st.secrets.postgres.user
    DB_PASS = st.secrets.postgres.password
    DB_CONNECTION = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    # Fallback for local development
    DB_CONNECTION = "postgresql://localhost/company_articles"

EXPORT_DIR = "exports"
DEPARTMENTS = ["Marketing", "Sales", "Engineering", "Research", "Content", "Leadership", "Product", "Customer Success"]

# ---- Database Helper Functions ----
def generate_content_hash(content):
    """Generate a hash of article content for duplicate detection"""
    try:
        words = [w.lower() for w in word_tokenize(content) if w.isalnum() and w.lower() not in stop_words]
        normalized_content = " ".join(words)
        return hashlib.sha256(normalized_content.encode()).hexdigest()
    except:
        # Simple fallback if NLTK fails
        return hashlib.sha256(content.encode()).hexdigest()

def get_conn():
    """Get database connection"""
    return psycopg2.connect(DB_CONNECTION)

def check_duplicates(title, content, keywords):
    """Check for duplicate articles"""
    result = {
        "found_duplicates": False,
        "title_duplicates": [],
        "keyword_duplicates": [],
        "content_duplicates": []
    }
    
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # 1. Check for similar title
        cur.execute("""
        SELECT id, title
        FROM articles
        WHERE title = %s OR title ILIKE %s OR %s ILIKE CONCAT('%%', title, '%%')
        LIMIT 3
        """, (title, f"%{title}%", title))
        
        title_duplicates = cur.fetchall()
        if title_duplicates:
            result["found_duplicates"] = True
            result["title_duplicates"] = [
                {"id": row[0], "title": row[1]} for row in title_duplicates
            ]
        
        # 2. Check for keyword matches
        if keywords:
            placeholders = ', '.join(['%s'] * len(keywords))
            query = f"""
            SELECT id, title, keywords
            FROM articles
            WHERE keywords @> ARRAY[{placeholders}]::text[]
            LIMIT 5
            """
            cur.execute(query, keywords)
            
            keyword_duplicates = cur.fetchall()
            if keyword_duplicates:
                result["found_duplicates"] = True
                for row in keyword_duplicates:
                    matching_kw = [k for k in keywords if k in row[2]]
                    result["keyword_duplicates"].append({
                        "id": row[0],
                        "title": row[1],
                        "matching_keywords": matching_kw
                    })
        
        # 3. Check for content similarity
        content_hash = generate_content_hash(content)
        
        # Check for exact match
        cur.execute("""
        SELECT COUNT(*) FROM articles 
        WHERE content_hash = %s
        """, (content_hash,))
        
        has_exact_match = cur.fetchone()[0] > 0
        if has_exact_match:
            result["found_duplicates"] = True
            
            cur.execute("""
            SELECT id, title FROM articles 
            WHERE content_hash = %s
            """, (content_hash,))
            
            exact_matches = cur.fetchall()
            for row in exact_matches:
                result["content_duplicates"].append({
                    "id": row[0],
                    "title": row[1],
                    "similarity": 1.0
                })
        
        # Check similar content using keywords
        content_words = [w.lower() for w in word_tokenize(content) 
                         if w.isalnum() and w.lower() not in stop_words and len(w) > 3]
        top_words = content_words[:10]
        
        if top_words:
            tsquery_str = " | ".join(top_words)
            
            cur.execute("""
            SELECT id, title, ts_rank(document_vector, to_tsquery('english', %s)) AS rank
            FROM articles 
            WHERE document_vector @@ to_tsquery('english', %s)
            AND content_hash != %s
            ORDER BY rank DESC
            LIMIT 3
            """, (tsquery_str, tsquery_str, content_hash))
            
            similar_content = []
            for row in cur.fetchall():
                similarity_score = float(row[2])/10.0
                if similarity_score >= 0.25:
                    similar_content.append((row[0], row[1], similarity_score))
                    
            if similar_content:
                result["found_duplicates"] = True
                for row in similar_content:
                    result["content_duplicates"].append({
                        "id": row[0],
                        "title": row[1],
                        "similarity": row[2]
                    })
        
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error checking duplicates: {e}")
    
    return result

def add_article(title, content, author, department, publish_date, keywords, url=None):
    """Add a new article to the database"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Generate content hash
        content_hash = generate_content_hash(content)
        
        # Insert article
        cur.execute("""
        INSERT INTO articles 
        (title, content, author, department, publish_date, keywords, url, content_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """, (
            title, content, author, department, publish_date, 
            keywords, url, content_hash
        ))
        
        article_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            "success": True,
            "message": "Article added successfully",
            "article_id": article_id
        }
    except Exception as e:
        if conn:
            conn.rollback()
        return {
            "success": False,
            "message": f"Error adding article: {e}"
        }

def get_recent_articles(days=30, department=None, limit=100):
    """Get recent articles with optional department filter"""
    try:
        engine = create_engine(DB_CONNECTION)
        
        if department and department != "All":
            query = f"""
            SELECT id, title, author, department, publish_date, keywords
            FROM articles
            WHERE publish_date >= NOW() - INTERVAL '{days} days'
            AND department = '{department}'
            ORDER BY publish_date DESC 
            LIMIT {limit}
            """
        else:
            query = f"""
            SELECT id, title, author, department, publish_date, keywords
            FROM articles
            WHERE publish_date >= NOW() - INTERVAL '{days} days'
            ORDER BY publish_date DESC 
            LIMIT {limit}
            """
        
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def search_articles(query, limit=100):
    """Search articles using full-text search"""
    try:
        engine = create_engine(DB_CONNECTION)
        
        sql_query = f"""
        SELECT id, title, author, publish_date, ts_rank(document_vector, to_tsquery('english', %s)) AS relevance
        FROM articles
        WHERE document_vector @@ to_tsquery('english', %s)
        ORDER BY relevance DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql_query(sql_query, engine, params=(query, query))
        return df
    except Exception as e:
        st.error(f"Search error: {e}")
        return pd.DataFrame()

def get_trending_keywords(days=30, limit=20):
    """Get trending keywords from recent articles"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Fix the interval syntax here
        cur.execute("""
        SELECT keyword, COUNT(*) as count
        FROM articles, unnest(keywords) as keyword
        WHERE publish_date >= NOW() - INTERVAL '%s days'
        GROUP BY keyword
        ORDER BY count DESC
        LIMIT %s
        """, (days, limit))
        
        keywords = cur.fetchall()
        cur.close()
        conn.close()
        
        return [{"keyword": k[0], "count": k[1]} for k in keywords]
    except Exception as e:
        st.error(f"Error getting keywords: {e}")
        return []

# ----- Streamlit UI -----
def main():
    st.set_page_config(
        page_title="Company Article Tracker",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "submission_step" not in st.session_state:
        st.session_state.submission_step = 1
    
    # Sidebar navigation
    st.sidebar.title("Article Tracking System")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Submit Article", "Recent Articles", "Search Articles", "Keyword Analysis"]
    )
    
    # Reset submission flow when changing pages
    if page != "Submit Article" and "submission_step" in st.session_state:
        st.session_state.submission_step = 1
    
    # Test database connection
    try:
        conn = get_conn()
        conn.close()
        st.sidebar.success("✅ Connected to database")
    except Exception as e:
        st.sidebar.error(f"❌ Database connection error: {e}")
    
    # Page content
    if page == "Submit Article":
        display_article_submission_form()
    elif page == "Recent Articles":
        display_recent_articles()
    elif page == "Search Articles":
        display_search()
    elif page == "Keyword Analysis":
        display_keyword_analysis()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Company Article Tracking System")

def display_article_submission_form():
    st.title("Submit New Article")
    
    # Reset button
    if st.sidebar.button("New Article"):
        # Only set the session state if it hasn't been set already
        if st.session_state.get("reset_clicked") != True:
            # Clear form-related session state
            keys_to_delete = ['article_data', 'duplicate_result']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Set to step 1
            st.session_state.submission_step = 1
            
            # Mark that we've processed this reset
            st.session_state.reset_clicked = True
            
            # Rerun to refresh
            st.experimental_rerun()
    else:
        # Reset flag when button is not clicked
        if "reset_clicked" in st.session_state:
            del st.session_state.reset_clicked
        
    
    # Step 1: Article Information Form
    if st.session_state.submission_step == 1:
        st.write("Enter the article details below.")
        
        # Pre-fill logic
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
                author = st.text_input("Author", 
                                      value=prefill.get("author", ""),
                                      help="Name of the article author")
                url = st.text_input("Article URL (optional)", 
                                   value=prefill.get("url", ""),
                                   help="Link to the published article")
            
            with col2:
                department = st.selectbox("Department", DEPARTMENTS, 
                                         index=DEPARTMENTS.index(prefill.get("department", DEPARTMENTS[0])) if "department" in prefill else 0)
                publish_date = st.date_input(
                    "Publication Date", 
                    value=prefill.get("publish_date", datetime.now().date()),
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
                duplicate_check = check_duplicates(title, content, keyword_list)
            
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
            result = add_article(
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
                if "article_data" in st.session_state:
                    del st.session_state.article_data
                if "duplicate_result" in st.session_state:
                    del st.session_state.duplicate_result
                st.experimental_rerun()
        else:
            st.error(f"Error: {result['message']}")
            
            if st.button("Try Again"):
                st.session_state.submission_step = 2
                st.experimental_rerun()

def display_recent_articles():
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
        df = get_recent_articles(days=days, department=department_filter if department_filter != "All" else None)
        
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

def display_search():
    st.title("Search Articles")
    
    # Search form
    search_query = st.text_input("Enter search terms", help="Search articles by content, title, or keywords")
    
    if search_query:
        with st.spinner("Searching..."):
            df = search_articles(search_query.replace(' ', ' & '))
        
        if not df.empty:
            # Format dates
            df['publish_date'] = pd.to_datetime(df['publish_date']).dt.strftime('%Y-%m-%d')
            
            # Show results
            st.success(f"Found {len(df)} results")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No articles found matching your search terms.")

def display_keyword_analysis():
    st.title("Keyword Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        days = st.slider("Timeframe (days)", 7, 180, 30, help="Analyze keywords from the past N days")
        limit = st.slider("Number of keywords", 5, 50, 20, help="Show top N keywords")
    
    try:
        with st.spinner("Analyzing keywords..."):
            trending = get_trending_keywords(days=days, limit=limit)
        
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

if __name__ == "__main__":
    main()