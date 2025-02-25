import schedule
import time
import logging
from article_tracker import ArticleTracker
from config import DB_CONNECTION, EXPORT_DIR, WEEKLY_REPORT_DAY, WEEKLY_REPORT_TIME, MONTHLY_EXPORT_INTERVAL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduled_jobs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('article_scheduler')

def run_weekly_report(tracker):
    """Generate weekly executive report"""
    logger.info("Generating weekly report")
    try:
        result = tracker.generate_weekly_report()
        logger.info(f"Weekly report generated with {result['article_count']} articles")
    except Exception as e:
        logger.error(f"Error generating weekly report: {e}")

def export_database(tracker):
    """Export full article database"""
    logger.info("Exporting full article database")
    try:
        filepath = tracker.export_all_articles(format="excel")
        logger.info(f"Database exported to: {filepath}")
    except Exception as e:
        logger.error(f"Error exporting database: {e}")

def main():
    logger.info("Starting scheduled job service")
    
    # Initialize tracker
    tracker = ArticleTracker(DB_CONNECTION, EXPORT_DIR)
    
    # Schedule weekly executive report
    getattr(schedule.every(), WEEKLY_REPORT_DAY).at(WEEKLY_REPORT_TIME).do(
        run_weekly_report, tracker
    )
    logger.info(f"Scheduled weekly report for {WEEKLY_REPORT_DAY} at {WEEKLY_REPORT_TIME}")
    
    # Schedule monthly database export
    schedule.every(MONTHLY_EXPORT_INTERVAL).days.do(export_database, tracker)
    logger.info(f"Scheduled monthly export every {MONTHLY_EXPORT_INTERVAL} days")
    
    logger.info("Scheduler started")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Scheduler shutting down")
        tracker.clean_up()

if __name__ == "__main__":
    main()