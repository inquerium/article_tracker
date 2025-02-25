# Database connection
# Format: "postgresql://username:password@host:port/database_name"
# Example: "postgresql://admin:password@localhost:5432/company_articles"
DB_CONNECTION = "postgresql://admin:password@localhost:5432/company_articles"

# Export directory for reports and data exports
EXPORT_DIR = "exports"

# Department list
DEPARTMENTS = [
    "Marketing",
    "Sales",
    "Engineering",
    "Research",
    "Content",
    "Leadership",
    "Product",
    "Customer Success"
]

# Duplicate detection thresholds
TITLE_SIMILARITY_THRESHOLD = 0.7  # Title similarity threshold (0-1)
CONTENT_SIMILARITY_THRESHOLD = 0.6  # Content similarity threshold (0-1)

# Scheduled job settings
WEEKLY_REPORT_DAY = "monday"
WEEKLY_REPORT_TIME = "08:00"
MONTHLY_EXPORT_INTERVAL = 30  # days