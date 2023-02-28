import sys
import os
import pandas as pd
import pickle
import collections
import sklearn



def main(args) -> None:
    """ Main function to be called when the script is run from the command line.
    This function will recommend songs based on the user's input and save the
    playlist to a csv file.

    Parameters
    ----------
    args: list
        list of arguments from the command line
    Returns
    -------
    None
    """
    arg_list = args[1:]
    if len(arg_list) == 0:
        print("Usage: python3 musicRecommender.py <csv file>")
        sys.exit()
    else:
        file_name = arg_list[0]
        if not os.path.isfile(file_name):
            print("File does not exist")
            sys.exit()
        else:
            userPreferences = pd.read_csv(file_name)

    # this code is just to check, delete later.
    print(userPreferences.head())

    # TODO:
    # 1. Use your train model to make recommendations for the user.
    # 2. Output the recommendations as 5 different playlists with
    #    the top 5 songs in each playlist. (5 playlists x 5 songs)
    # 2.1. Musics in a single playlist should be from the same cluster.
    # 2.2. Save playlists to a csv file.
    # 3. Output another single playlist recommendation with all top songs from all clusters.

    cluster = pickle.load(open(r"C:\Users\Admin\PycharmProjects\Recommender_System\model.pkl",'rb'))
    predict = cluster.predict(userPreferences)
    print(predict)
    dict = collections.Counter(predict)
    print(dict)
    print(len(predict))
    data = pd.read_csv(r"C:\Users\Admin\PycharmProjects\Recommender_System\AllMusic_dataframe.csv")
    arr = [15,8,12,17,3]
    count = 1
    for x in arr:
        print(count,'. ',x,' done')
        data[data['cluster']==x].sample(5).to_csv(r'C:\Users\Admin\PycharmProjects\Recommender_System\AllMusic_dataframe'+str(count)+r'.csv')
        count = count +1

    count = 1
    dataframe = pd.DataFrame()
    for k, v in dict.items():
        print(count, '. ', k, ' done')
        frames = [dataframe, data[data['cluster'] == k].sample(v)]
        dataframe = pd.concat(frames)
        count = count +1
    dataframe.to_csv(r'C:\Users\Admin\PycharmProjects\Recommender_System\TopSongs_dataframe.csv')


if __name__ == "__main__":
    # get arguments from command line
    args = sys.argv
    main(args)