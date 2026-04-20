# run_etl.py — jalankan semua fase berurutan
import extract, transform, load

if __name__ == "__main__":
    print("=== Mulai ETL Pipeline ===")
    extract.extract_all()
    transform.transform_all()
    load.load_to_vector_db()
    print("=== ETL Pipeline selesai ===")

    # penjadwalan otomatis setiap hari jam 16.15 WIB bisa menggunakan cron job atau task scheduler sesuai OS
    #crontab -e 
    #"15 16 * * 1-5 /path/to/python /path/to/etl/run_etl.py >> /path/to/log/etl.log 2>&1"
    