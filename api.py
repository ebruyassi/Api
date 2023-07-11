from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import seaborn as sns
from surprise import Reader, Dataset
from surprise import SVDpp
from collections import defaultdict
from surprise import SVDpp
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from waitress import serve

app = Flask(__name__)


@app.route("/api/userBased", methods=['GET'])
def getUBCfRec():
    # Read in data
    ratings = pd.read_csv(
        'C:/xx/userRating.csv', encoding="latin-1")

    products = pd.read_csv(
        'C:/xx/products.csv', encoding="latin-1")
    
    df = pd.merge(ratings, products, on='name', how='inner')
    
    agg_ratings = df.groupby('name').agg(mean_rating=('rating', 'mean'),
                                         number_of_ratings=('rating', 'count')).reset_index()
    agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings'] > 10]
    agg_ratings_GT100.sort_values(
        by='number_of_ratings', ascending=False).head()
    df_GT100 = pd.merge(
        df, agg_ratings_GT100[['name']], on='name', how='inner')

    matrix = df_GT100.pivot_table(
        index='uid', columns='name', values='rating')
    
    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis='rows')
    user_similarity = matrix_norm.T.corr(method='pearson')
    picked_userid = str(request.args['query'])
    user_similarity.drop(index=picked_userid, inplace=True)
    
    n = 3

    user_similarity_threshold = 0.8
    similar_users = user_similarity[user_similarity[picked_userid] >
                                    user_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]
    picked_userid_order = matrix_norm[matrix_norm.index == picked_userid].dropna(
        axis=1, how='all')
    picked_userid_order
    similar_user_products = matrix_norm[matrix_norm.index.isin(
        similar_users.index)].dropna(axis=1, how='all')

    similar_user_products.drop(
        picked_userid_order.columns, axis=1, inplace=True, errors='ignore')

    item_score = {}

    for i in similar_user_products.columns:
        product_rating = similar_user_products[i]
        total = 0
        count = 0
        for u in similar_users.index:
            if pd.isna(product_rating[u]) == False:
                score = similar_users[u] * product_rating[u]
                total += score
                count += 1
        item_score[i] = total / count
    item_score = pd.DataFrame(item_score.items(), columns=[
                              'product', 'product_score'])
    ranked_item_score = item_score.sort_values(
        by='product_score', ascending=False)
    d={}
    answer=ranked_item_score["product"].head(10).to_list()
    d["output"]=answer
    return answer


@app.route("/api/svdAlg", methods=['GET'])
def getSvdRec():
    df_ratings = pd.read_csv('C:/xx/userRating.csv')
    df_products = pd.read_csv('C:/xx/products.csv')

    minimum_rating = min(df_ratings['rating'].values)
    maximum_rating = max(df_ratings['rating'].values)

    df = pd.merge(df_ratings, df_products, on='name', how='inner')

    df_ratings = df.drop_duplicates(['uid', 'name'])
    ratings_mat = df_ratings.pivot(
        index='uid',
        columns='id',
        values='rating').fillna(0)
    ratings_mat.head()
    reader = Reader(rating_scale=(minimum_rating, maximum_rating))
    data = Dataset.load_from_df(df_ratings[['uid', 'id', 'rating']], reader)
    algo = SVDpp()
    algo.fit(data.build_full_trainset())

    picked_uid=str(request.args['query'])
    list_of_unrated_product = np.nonzero(
        ratings_mat.loc[picked_uid].to_numpy() == 0)[0]

    user_set = [[picked_uid, id, 0] for id in list_of_unrated_product]

    predictions = algo.test(user_set)
    top_n_recommendations = defaultdict(list)

    for picked_uid, id, _, est, _ in predictions:

        top_n_recommendations[picked_uid].append((id, est))

    for picked_uid, rating in top_n_recommendations.items():

        rating.sort(key=lambda x: x[1], reverse=True)

        top_n_recommendations[picked_uid] = rating[:20]

    count = 0

    print("Recommendations for user with id {}: ".format(picked_uid))
    ranked_item=[]
    for id, rating in top_n_recommendations[picked_uid]:
        count += 1
        print('{}. {}, predicted rating = {}'.format(
            count, df_products[df_products['id'] == id]['name'].iloc[0], round(rating, 3)))
        ranked_item.append(df_products[df_products['id'] == id]['name'].iloc[0])
    return ranked_item

@app.route("/api/knnAlg", methods=['GET'])
def getKnnRec():
    df_products = pd.read_csv('C:/xx/products.csv', 
                              encoding="latin-1",
                              usecols=['name', 'info','contents'],
                              dtype={'name': str, 'info': str, 'contents': str})
    df_ratings = pd.read_csv('C:/xx/userRating.csv',
                             encoding="latin-1",
                             usecols=['uid', 'name','rating'])
    df_products.head()
    df_ratings.head()
    df_ratings = df_ratings.drop_duplicates(['uid', 'name'])

    df_product_features = df_ratings.pivot(
        index='name',
        columns='uid',
        values='rating').fillna(0)
    
    df_product_features.head()
    
    mat_product_features = csr_matrix(df_product_features.values)
    df_product_features.head()
    model_knn=NearestNeighbors(metric="minkowski",algorithm="brute")
    model_knn.fit(mat_product_features)
    distances,indices=model_knn.kneighbors(df_product_features.loc[[str(request.args['query'])]],n_neighbors=10)
    ranked_items=[]
    for i in range(0,len(distances.flatten())):
            ranked_items.append(df_product_features.index[indices.flatten()[i]])
    return ranked_items
if __name__ == "__main__":
    serve(app, host="192.168.1.51", port=2530)
