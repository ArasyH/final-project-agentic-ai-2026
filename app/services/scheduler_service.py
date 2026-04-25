# app/services/scheduler_service.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import subprocess
import os

TIMEZONE = pytz.timezone("Asia/Jakarta")

def run_etl_job():
    print("[SCHEDULER] Running ETL pipeline...")

    try:
        subprocess.run(
            ["python", "etl/run_etl.py"],
            cwd=os.getcwd(),
            check=True
        )
        print("[SCHEDULER] ETL completed successfully.")
    except Exception as e:
        print(f"[SCHEDULER] ETL failed: {e}")

def start_scheduler():
    scheduler = BackgroundScheduler(timezone=TIMEZONE)

    # Senin - Kamis jam 16:00 WIB
    scheduler.add_job(
        run_etl_job,
        CronTrigger(day_of_week="mon-thu", hour=16, minute=0),
        id="etl_weekday"
    )

    # Jumat jam 16:00 WIB
    scheduler.add_job(
        run_etl_job,
        CronTrigger(day_of_week="fri", hour=16, minute=0),
        id="etl_friday"
    )

    scheduler.start()
    print("[SCHEDULER] ETL scheduler started.")

    return scheduler