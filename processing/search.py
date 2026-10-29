from database.database_get import SessionLocal,Article
def search_article(keyword):
    db = SessionLocal()
    result = db.query(Article).filter(
        #tìm những bài có title chứa keyword
        (Article.title.contains(keyword)) |
        (Article.sapo.contains(keyword)) |
        (Article.content.contains(keyword))
    ).all()
    db.close()
    return result
def caculate_score(article,keyword):
    score = 0 
    if keyword.lower() in article.title.lower():
        score +=3
    if keyword.lower() in article.sapo.lower():
        score +=2
    if keyword.lower() in article.content.lower():
        score +=1
    return score
if __name__ == "__main__" :
    keyword = input("mời nhập từ khoá : ")
    result = search_article(keyword)
    if result :
        scored_result = []
        for article in result :
            score = caculate_score(article,keyword)
            scored_result.append((article,score))
    #sắp xếp theo điểm (phần tử x[1]) tăng dần  (x[0] là article)
    scored_result.sort(key = lambda x : x[1] , reverse=True)
    for article , score in scored_result :
        print("\n-------------------------")
        print("Điểm:", score)
        print("ID:", article.id)
        print("Tiêu đề:", article.title)
        print("Category:", article.category)
        print("Ngày đăng:", article.published_at)
        print("URL:", article.url)
    else : 
        print("không tim thấy bài báo ")