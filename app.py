from flask import Flask, render_template, request
from joblib import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# Load mô hình và vectorizer
clf = load('nlp_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Load dữ liệu để hiển thị thông tin phim
data = pd.read_csv('data.csv')

# Tạo bảng từ khóa với 2 cột
keyword_table = pd.DataFrame()
keyword_table['movie_title'] = data['movie_title']
keyword_table['keywords'] = data['director_name'] + ' ' + data['actor_1_name'] + ' ' + data['actor_2_name'] + ' ' + data['actor_3_name'] + ' ' + data['genres']

# Chuyển đổi từ khóa thành ma trận TF-IDF
tfidf_matrix = vectorizer.transform(keyword_table['keywords'])

# Tính ma trận linear kernel 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Hàm gợi ý với kiểm tra chỉ số hợp lệ
def recommend_movies_safe(movie_title, cosine_sim, threshold=0.2):
    movie_title = movie_title.strip()

    # Kiểm tra xem tên phim có trong bảng từ khóa không
    if movie_title not in keyword_table['movie_title'].values:
        print(f"Phim '{movie_title}' không có trong dữ liệu.")
        return []

    idx = keyword_table.index[keyword_table['movie_title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = [(i, score) for i, score in sim_scores if score > threshold]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:11]]

    # Kiểm tra xem chỉ số có hợp lệ không
    valid_indices = [i for i in movie_indices if i < len(data)]

    # Trả về danh sách các đối tượng phim với thông tin đầy đủ
    recommended_movies = []
    for i in valid_indices:
        movie_info = {
            'title': data['movie_title'].iloc[i],
            'director': data['director_name'].iloc[i],
            'genres': data['genres'].iloc[i]
        }
        recommended_movies.append(movie_info)

    return recommended_movies

# Trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Xử lý form tìm kiếm
@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        query = request.form['query'].lower()

        # Lấy thông tin của phim được tìm kiếm
        search_movie_info = get_movie_info(query)

        if not search_movie_info:
            return render_template('index.html', query=query, not_found=True)

        # Sử dụng mô hình để đề xuất phim
        recommended_movies = recommend_movies_safe(query, cosine_sim)

        return render_template('index.html', query=query, search_movie_info=search_movie_info, recommended_movies=recommended_movies)

# Hàm để lấy thông tin của phim được tìm kiếm
def get_movie_info(movie_title):
    movie_info = {}
    if movie_title in data['movie_title'].values:
        idx = data.index[data['movie_title'] == movie_title].tolist()[0]
        movie_info = {
            'title': data['movie_title'].iloc[idx],
            'director': data['director_name'].iloc[idx],
            'genres': data['genres'].iloc[idx],
            'poster_url': url_for('static', filename=f'hinhanh/phim.jpg')  
        }
    return movie_info

if __name__ == '__main__':
    app.run(debug=True)

