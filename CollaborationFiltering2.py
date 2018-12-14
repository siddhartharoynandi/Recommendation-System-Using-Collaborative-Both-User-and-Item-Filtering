import csv
import numpy as np
from scipy import spatial

def movieCF(rating_matrix,userID,movoieID,num_neighbours):
    trans_rating = np.transpose(rating_matrix)
    similarity_scores = []
    compare_vector = trans_rating[movoieID-1]
    #print compare_vector.shape
    for i in range(len(trans_rating)):
        #print trans_rating[i]
        #print compare_vector
        k = 1 - spatial.distance.cosine(trans_rating[i], compare_vector)
        if not np.isnan(k):
            similarity_scores.append(k)
        else:
            similarity_scores.append(0.0)
    #print similarity_scores[4505],similarity_scores[5451]
    top_movie_neighbours_idx = list(np.argsort(similarity_scores)[-(num_neighbours+1):])
    #print top_movie_neighbours_idx
    top_movie_neighbours_idx.remove(movoieID - 1)
    movieID_avg_rating = np.mean((trans_rating[movoieID - 1])[trans_rating[movoieID - 1] > 0])
    neighbour_movie_avg_rating = []
    for i in top_movie_neighbours_idx:
        neighbour_movie_avg_rating.append(np.mean(trans_rating[i][trans_rating[i]>0]))
        #print trans_rating[i][userID-1]
    #print rating_matrix[userID-1][4505],rating_matrix[userID-1][5451],rating_matrix[userID-1][5019],rating_matrix[userID-1][7564],rating_matrix[userID-1][5748]
    #print neighbour_movie_avg_rating,movoieID_avg_rating
    userID_movieID_Rating = 0.0
    sum_neighbour_similarity = 0.0
    for idx,i in enumerate(top_movie_neighbours_idx):
        sum_neighbour_similarity = sum_neighbour_similarity + similarity_scores[i]
        userID_movieID_Rating = userID_movieID_Rating + similarity_scores[i]* (rating_matrix[userID-1][i] - neighbour_movie_avg_rating[idx])
    return movieID_avg_rating+(userID_movieID_Rating/sum_neighbour_similarity)


def userCF(rating_matrix,userID,movoieID,num_neighbours):
    similarity_scores = []
    compare_vector = rating_matrix[userID-1]
    for i in range(len(rating_matrix)):
        similarity_scores.append(1 - spatial.distance.cosine(rating_matrix[i], compare_vector))
    top_num_neighbours_idx = list(np.argsort(similarity_scores)[-(num_neighbours+1):])
    #print top_num_neighbours_idx
    top_num_neighbours_idx.remove(userID-1)
    userID_avg_rating = np.mean((rating_matrix[userID-1])[rating_matrix[userID-1]>0])
    neighbour_avg_rating = []
    for i in top_num_neighbours_idx:
        neighbour_avg_rating.append(np.mean(rating_matrix[i][rating_matrix[i]>0]))
        #print rating_matrix[i][movoieID-1]
    #print neighbour_avg_rating,userID_avg_rating
    movieID_userID_Rating = 0.0
    sum_neighbour_similarity = 0.0
    for idx,i in enumerate(top_num_neighbours_idx):
        sum_neighbour_similarity = sum_neighbour_similarity + similarity_scores[i]
        movieID_userID_Rating = movieID_userID_Rating + similarity_scores[i]* (rating_matrix[i][movoieID-1] - neighbour_avg_rating[idx])
    return userID_avg_rating + (movieID_userID_Rating/sum_neighbour_similarity)

def CollaborativeFiltering(rating_matrix,userID,movoieID,num_neighbours):
    return userCF(rating_matrix,userID,movoieID,num_neighbours),movieCF(rating_matrix, userID, movoieID, num_neighbours)

if __name__ == '__main__':
    movie_list = []
    with open('/Users/siddhartharoynandi/Desktop/ml-latest-small/movies.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            movie_list.append(row[0])
    no_movies = len(movie_list)

    user_rating_dict = {}
    with open('/Users/siddhartharoynandi/Desktop/ml-latest-small/ratings.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            k = []
            k.append(row[1])
            k.append(row[2])

            if row[0] in user_rating_dict.keys():
                user_rating_dict[row[0]] = user_rating_dict[row[0]] + [k]
            else:
                user_rating_dict[row[0]] = [k]

    no_users = len(user_rating_dict)

    rating_matrix = np.zeros((no_users, no_movies)) # 610X9742
    for i in user_rating_dict.keys():
        for j in user_rating_dict[i]:
            movie_index = movie_list.index(j[0])
            rating_matrix[int(i)-1][movie_index] = j[1]
    ratings = CollaborativeFiltering(rating_matrix, 1, 1, 5)
    f = open('/Users/siddhartharoynandi/Desktop/ADM/CF_Out.txt', 'w')
    f.write('User Based Rating: '+str(ratings[0]))
    f.write('\n')
    f.write('Movie Based Rating: '+str(ratings[1]))
    f.close()
    exit(0)

