from sqlalchemy import inspect, text
from database.database_15k import SessionLocal


def main():
    db = SessionLocal()

    try:
        engine = db.get_bind()
        columns = {
            column["name"]
            for column in inspect(engine).get_columns("articles")
        }

        if "published_at" in columns:
            print("Trường published_at đã tồn tại.")
            return

        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE articles "
                    "ADD COLUMN published_at DATETIME"
                )
            )

        print("Đã thêm trường published_at thành công.")

    finally:
        db.close()


if __name__ == "__main__":
    main()