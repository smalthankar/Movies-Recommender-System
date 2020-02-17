import pandas as pd
import numpy as np


def prepare_ratings_data():
    ratings_cols = ['user_id', 'movie_id', 'rating']
    ratings_df = pd.read_csv(r'ml-100k\u.data', sep='\t', names=ratings_cols, usecols=range(3))

    movies_cols = ['movie_id', 'title']
    movies_df = pd.read_csv(r'ml-100k\u.item', sep='|', names=movies_cols, usecols=range(2), encoding='latin-1')

    ratings_df = pd.merge(movies_df, ratings_df)

    return ratings_df


def prepare_my_system_ratings_data(ratings_df):
    user_ratings = ratings_df.pivot_table(index=['user_id'], columns=['title'], values='rating')

    corr_matrix = user_ratings.corr()

    corr_matrix = user_ratings.corr(method='pearson', min_periods=100)

    my_system_ratings = user_ratings.loc[1].dropna()
    my_system_ratings

    return corr_matrix, my_system_ratings


def get_similar_movies(corr_matrix, my_system_ratings):
    similar_movies = pd.Series()

    for i in range(0, len(my_system_ratings.index)):
        print('Adding similar movies for - ' + my_system_ratings.index[i] + '...')

        # Retrieve similar movies to the movies rated by the user
        similar = corr_matrix[my_system_ratings.index[i]].dropna()

        # Now scale its similarity that how well the user has rated this movie
        similar = similar.map(lambda x: x * my_system_ratings[i])

        # Add the score to the list of similar_movies
        similar_movies = similar_movies.append(similar)

    return similar_movies


def show_top_5_results(similar_movies):
    similar_movies.sort_values(inplace=True, ascending=False)
    print(similar_movies.head(5))


def main():
    ratings_df = prepare_ratings_data()

    corr_matrix, my_system_ratings = prepare_my_system_ratings_data(ratings_df)

    similar_movies = get_similar_movies(corr_matrix, my_system_ratings)

    show_top_5_results(similar_movies)


if __name__ == '__main__':
    main()