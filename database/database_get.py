from sqlalchemy import create_engine,Column,Integer,String,Text
from sqlalchemy.orm import declarative_base,sessionmaker
engine = create_engine("sqlite:///data/news.db")

#base là cơ sở để để sqlalchemy biết lớp nào sẽ được dùng làm bảng trong database
Base = declarative_base()

#session quản lý các phiên làm việc với database (thêm , sửa, xoá , tìm bài báo thực hiện thông )
#qua session (python -> session -> database)
SessionLocal = sessionmaker(bind=engine)

#tao bang
class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)

    title = Column(String(500), nullable=False)

    sapo = Column(Text)

    content = Column(Text, nullable=False)

    category = Column(String(100))

    research_topic = Column(String(100))

    author = Column(String(200))

    published_at = Column(String(50))

    source = Column(String(200))

    url = Column(String(1000), unique=True)

    

Base.metadata.create_all(bind=engine)
