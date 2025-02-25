import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine
from datetime import datetime
import hashlib
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("article_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('article_tracker')

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

class ArticleTracker:
    def __init__(self, db_connection_string, export_directory):
        """
        Initialize the Article Tracker system
        
        Args:
            db_connection_string (str): PostgreSQL connection string
            export_directory (str): Path to save reports and exports
        """
        self.db_conn_string = db_connection_string
        self.export_dir = Path(export_directory)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        try:
            self.conn = psycopg2.connect(db_connection_string)
            self.setup_database()
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
        # For pandas exports
        self.engine = create_engine(f'postgresql+psycopg2://{db_connection_string.split("://")[1]}')
    
    def setup_database(self):
        """Create necessary database tables if they don't exist"""
        cur = self.conn.cursor()
        
        # Articles table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            author TEXT NOT NULL,
            department TEXT,
            publish_date TIMESTAMP NOT NULL,
            keywords TEXT[],
            url TEXT,
            content_hash TEXT UNIQUE,
            document_vector TSVECTOR,
            metadata JSONB
        );
        
        CREATE INDEX IF NOT EXISTS article_text_idx ON articles USING GIN (document_vector);
        CREATE INDEX IF NOT EXISTS article_keywords_idx ON articles USING GIN (keywords);
        CREATE INDEX IF NOT EXISTS article_hash_idx ON articles (content_hash);
        """)
        
        # Make sure the trigram extension is available for text similarity
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        # Keyword tracking table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS keywords (
            id SERIAL PRIMARY KEY,
            keyword TEXT UNIQUE NOT NULL,
            article_count INTEGER DEFAULT 0,
            last_used TIMESTAMP
        );
        """)
        
        # Text search vector update trigger
        cur.execute("""
        CREATE OR REPLACE FUNCTION update_document_vector() RETURNS TRIGGER AS $$
        BEGIN
            NEW.document_vector = to_tsvector('english', NEW.title || ' ' || NEW.content);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        DROP TRIGGER IF EXISTS update_document_vector_trigger ON articles;
        CREATE TRIGGER update_document_vector_trigger
        BEFORE INSERT OR UPDATE ON articles
        FOR EACH ROW EXECUTE FUNCTION update_document_vector();
        """)
        
        # Update keyword counts trigger
        cur.execute("""
        CREATE OR REPLACE FUNCTION update_keyword_counts() RETURNS TRIGGER AS $$
        BEGIN
            -- Update counts for existing keywords
            UPDATE keywords 
            SET article_count = article_count + 1,
                last_used = NEW.publish_date
            WHERE keyword = ANY(NEW.keywords);
            
            -- Insert new keywords
            INSERT INTO keywords (keyword, article_count, last_used)
            SELECT keyword, 1, NEW.publish_date
            FROM unnest(NEW.keywords) AS keyword
            WHERE keyword NOT IN (SELECT keyword FROM keywords);
            
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        
        DROP TRIGGER IF EXISTS update_keyword_counts_trigger ON articles;
        CREATE TRIGGER update_keyword_counts_trigger
        AFTER INSERT ON articles
        FOR EACH ROW EXECUTE FUNCTION update_keyword_counts();
        """)
        
        self.conn.commit()
        cur.close()
    
    def _generate_content_hash(self, content):
        """Generate a hash of article content for duplicate detection"""
        # Normalize content: lowercase, remove punctuation, remove stop words
        words = [w.lower() for w in word_tokenize(content) if w.isalnum() and w.lower() not in stop_words]
        normalized_content = " ".join(words)
        return hashlib.sha256(normalized_content.encode()).hexdigest()
    
    def check_duplicates(self, title, content, keywords):
        """
        Comprehensive duplicate check across title, content, and keywords
        
        Args:
            title (str): Article title
            content (str): Article content
            keywords (list): List of keywords
            
        Returns:
            dict: Results of duplicate checks
        """
        result = {
            "found_duplicates": False,
            "title_duplicates": [],
            "keyword_duplicates": [],
            "content_duplicates": []
        }
        
        # 1. Check for duplicate/similar title
        cur = self.conn.cursor()
        cur.execute("""
        SELECT id, title
        FROM articles
        WHERE similarity(title, %s) > 0.7
        ORDER BY similarity(title, %s) DESC
        LIMIT 3
        """, (title, title))
        
        title_duplicates = cur.fetchall()
        if title_duplicates:
            result["found_duplicates"] = True
            result["title_duplicates"] = [
                {"id": row[0], "title": row[1]} for row in title_duplicates
            ]
        
        # 2. Check for keyword matches (articles with all the same keywords)
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
                    # Find which keywords match
                    matching_kw = [k for k in keywords if k in row[2]]
                    result["keyword_duplicates"].append({
                        "id": row[0],
                        "title": row[1],
                        "matching_keywords": matching_kw
                    })
        
        # 3. Check for content similarity
        # Normalize and hash content for comparison
        content_hash = self._generate_content_hash(content)
        
        # First check for exact content match
        cur.execute("""
        SELECT id, title FROM articles 
        WHERE content_hash = %s
        """, (content_hash,))
        
        exact_matches = cur.fetchall()
        if exact_matches:
            result["found_duplicates"] = True
            for row in exact_matches:
                result["content_duplicates"].append({
                    "id": row[0],
                    "title": row[1],
                    "similarity": 1.0  # Exact match
                })
        
        # Then check for similar content
        content_preview = " ".join(content.split()[:200])  # First 200 words
        cur.execute("""
        SELECT id, title, similarity(document_vector, to_tsvector('english', %s)) AS sim
        FROM articles 
        WHERE similarity(document_vector, to_tsvector('english', %s)) > 0.6
        AND content_hash != %s
        ORDER BY sim DESC
        LIMIT 3
        """, (content_preview, content_preview, content_hash))
        
        similar_content = cur.fetchall()
        if similar_content:
            result["found_duplicates"] = True
            for row in similar_content:
                result["content_duplicates"].append({
                    "id": row[0],
                    "title": row[1],
                    "similarity": float(row[2])/10.0  # Convert to similarity scale 0-1
                })
        
        cur.close()
        return result
    
    def add_article(self, title, content, author, department=None, 
                    publish_date=None, keywords=None, url=None, metadata=None):
        """
        Add a new article to the tracking system
        
        Args:
            title (str): Article title
            content (str): Full article content
            author (str): Author name
            department (str, optional): Department or team
            publish_date (datetime, optional): Publication date (defaults to now)
            keywords (list, optional): List of keywords
            url (str, optional): URL to the article
            metadata (dict, optional): Additional metadata
            
        Returns:
            dict: Result of the operation
        """
        # Set defaults
        if publish_date is None:
            publish_date = datetime.now()
        if metadata is None:
            metadata = {}
            
        # Generate content hash for duplicate detection
        content_hash = self._generate_content_hash(content)
            
        # Insert the article
        try:
            cur = self.conn.cursor()
            cur.execute("""
            INSERT INTO articles 
            (title, content, author, department, publish_date, keywords, url, content_hash, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """, (
                title, content, author, department, publish_date, 
                keywords, url, content_hash, json.dumps(metadata)
            ))
            
            article_id = cur.fetchone()[0]
            self.conn.commit()
            cur.close()
            
            logger.info(f"Added article: '{title}' (ID: {article_id})")
            return {
                "success": True,
                "message": "Article added successfully",
                "article_id": article_id
            }
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding article: {e}")
            return {
                "success": False,
                "message": f"Error adding article: {e}"
            }
    
    def get_article(self, article_id):
        """Get a single article by ID"""
        cur = self.conn.cursor()
        cur.execute("""
        SELECT id, title, content, author, department, publish_date, keywords, url, metadata
        FROM articles
        WHERE id = %s
        """, (article_id,))
        
        article = cur.fetchone()
        cur.close()
        
        if not article:
            return None
            
        return {
            "id": article[0],
            "title": article[1],
            "content": article[2],
            "author": article[3],
            "department": article[4],
            "publish_date": article[5],
            "keywords": article[6],
            "url": article[7],
            "metadata": article[8]
        }
    
    def get_articles_by_keyword(self, keyword, limit=100, offset=0):
        """Get articles that match a specific keyword"""
        cur = self.conn.cursor()
        cur.execute("""
        SELECT id, title, author, publish_date
        FROM articles
        WHERE %s = ANY(keywords)
        ORDER BY publish_date DESC
        LIMIT %s OFFSET %s
        """, (keyword, limit, offset))
        
        articles = cur.fetchall()
        cur.close()
        
        # Convert to list of dicts
        result = []
        for article in articles:
            result.append({
                "id": article[0],
                "title": article[1],
                "author": article[2],
                "publish_date": article[3]
            })
            
        return result
    
    def search_articles(self, query, limit=100, offset=0):
        """
        Search articles using full-text search
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            offset (int): Pagination offset
            
        Returns:
            list: Matching articles
        """
        cur = self.conn.cursor()
        cur.execute("""
        SELECT id, title, author, publish_date, ts_rank(document_vector, to_tsquery('english', %s)) AS rank
        FROM articles
        WHERE document_vector @@ to_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s OFFSET %s
        """, (query, query, limit, offset))
        
        articles = cur.fetchall()
        cur.close()
        
        # Convert to list of dicts
        result = []
        for article in articles:
            result.append({
                "id": article[0],
                "title": article[1],
                "author": article[2],
                "publish_date": article[3],
                "relevance": article[4]
            })
            
        return result
    
    def get_departments(self):
        """Get list of all departments with article counts"""
        cur = self.conn.cursor()
        cur.execute("""
        SELECT department, COUNT(*) as count
        FROM articles
        GROUP BY department
        ORDER BY count DESC
        """)
        
        departments = cur.fetchall()
        cur.close()
        
        return departments
    
    def get_trending_keywords(self, days=30, limit=20):
        """Get trending keywords from recent articles"""
        cur = self.conn.cursor()
        cur.execute("""
        SELECT keyword, article_count
        FROM keywords
        WHERE last_used >= NOW() - INTERVAL '%s days'
        ORDER BY article_count DESC
        LIMIT %s
        """, (days, limit))
        
        keywords = cur.fetchall()
        cur.close()
        
        return [{"keyword": k[0], "count": k[1]} for k in keywords]
    
    def generate_weekly_report(self):
        """Generate weekly article report for executives"""
        # Get articles from the past week
        df = pd.read_sql("""
        SELECT id, title, author, department, publish_date, keywords, url
        FROM articles
        WHERE publish_date >= NOW() - INTERVAL '7 days'
        ORDER BY publish_date DESC
        """, self.engine)
        
        # Get trending keywords
        trending_df = pd.read_sql("""
        SELECT keyword, article_count
        FROM keywords
        WHERE last_used >= NOW() - INTERVAL '7 days'
        ORDER BY article_count DESC
        LIMIT 20
        """, self.engine)
        
        # Department summary
        dept_summary = df.groupby('department').size().reset_index(name='article_count')
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d")
        df.to_excel(self.export_dir / f"weekly_articles_{timestamp}.xlsx", index=False)
        trending_df.to_excel(self.export_dir / f"trending_keywords_{timestamp}.xlsx", index=False)
        dept_summary.to_excel(self.export_dir / f"department_summary_{timestamp}.xlsx", index=False)
        
        logger.info(f"Generated weekly report: {self.export_dir}/weekly_articles_{timestamp}.xlsx")
        return {
            "article_count": len(df),
            "top_keywords": trending_df['keyword'].tolist()[:5],
            "department_summary": dept_summary.to_dict('records')
        }
    
    def export_all_articles(self, format="excel"):
        """Export all articles to file"""
        df = pd.read_sql("""
        SELECT id, title, author, department, publish_date, keywords, url
        FROM articles
        ORDER BY publish_date DESC
        """, self.engine)
        
        timestamp = datetime.now().strftime("%Y%m%d")
        if format.lower() == "excel":
            output_path = self.export_dir / f"all_articles_{timestamp}.xlsx"
            df.to_excel(output_path, index=False)
        elif format.lower() == "csv":
            output_path = self.export_dir / f"all_articles_{timestamp}.csv"
            df.to_csv(output_path, index=False)
        elif format.lower() == "json":
            output_path = self.export_dir / f"all_articles_{timestamp}.json"
            df.to_json(output_path, orient="records")
        
        logger.info(f"Exported all articles to: {output_path}")
        return str(output_path)
    
    def get_recent_articles(self, days=30, department=None, limit=100):
        """Get recent articles with optional department filter"""
        query = """
        SELECT id, title, author, department, publish_date, keywords
        FROM articles
        WHERE publish_date >= NOW() - INTERVAL '%s days'
        """
        
        params = [days]
        
        if department and department != "All":
            query += " AND department = %s"
            params.append(department)
            
        query += " ORDER BY publish_date DESC LIMIT %s"
        params.append(limit)
        
        df = pd.read_sql(query, self.engine, params=params)
        return df
    
    def clean_up(self):
        """Close database connections"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()