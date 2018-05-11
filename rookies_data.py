import pandas as pd
from nba_py import player
import numpy as np
import requests
from sklearn.cross_validation import train_test_split


def best_EFF(rookie_data):
    '''
    finds the best EFF of given dataframe of players scrapped from nba.com
    :return:  
    '''
    rookie_data.index = range(len(rookie_data))
    rookie_ids = rookie_data['PLAYER_ID']
    row = 0
    for pid in rookie_ids:
        print row, pid, "row pid"
        rank_EFF_by_year = player.PlayerCareer(pid).regular_season_rankings()["RANK_PG_EFF"]
        if all(r is None for r in rank_EFF_by_year):
            best_pid_EFF=None
        else:
            best_pid_EFF = np.nanmin(rank_EFF_by_year.iloc[:].values)
        print "best eff "+str(best_pid_EFF)
        rookie_data.at[row,"BEST EFF"]=best_pid_EFF
        row += 1
    return rookie_data

################### credit to Eyal Shafran
def boxscoretraditionalv2(GameID,EndPeriod='10',EndRange='28800',RangeType='0',StartPeriod='1',StartRange='0'):
    url = 'http://stats.nba.com/stats/boxscoretraditionalv2?'
    api_param = {
        'EndPeriod' : EndPeriod,
        'EndRange' : EndRange,
        'GameID' : GameID,
        'RangeType' : RangeType,
        'StartPeriod' : StartPeriod,
        'StartRange' : StartRange,
        }
    u_a = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 Safari/537.36"
    response = requests.get(url,params=api_param,headers={"USER-AGENT":u_a})
    data = response.json()
    return pd.DataFrame(data['resultSets'][0]['rowSet'],columns=data['resultSets'][0]['headers'])

def commonallplayers(currentseason=0,leagueid='00',season='2015-16'):
    url = 'http://stats.nba.com/stats/commonallplayers?'
    api_param = {
        'IsOnlyCurrentSeason' : currentseason,
        'LeagueID' : leagueid,
        'Season' : season,
    }
    u_a = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 Safari/537.36"
    response = requests.get(url,params=api_param,headers={"USER-AGENT":u_a})
    data = response.json()
    return pd.DataFrame(data['resultSets'][0]['rowSet'],columns=data['resultSets'][0]['headers'])

def seasons_string(start,end):
    '''
    creates a list of NBA seasons from start-end
    '''
    years = np.arange(start,end+1)
    seasons = []
    for year in years:
        string1 = str(year)
        string2 = str(year+1)
        season = '{}-{}'.format(string1,string2[-2:])
        seasons.append(season)
    return seasons

def gamelog(counter = 1000,datefrom='',dateto='',direction='DESC',leagueid='00',
            playerorteam='T',season='2015-16',seasontype='Regular Season',sorter='PTS'):
    url = 'http://stats.nba.com/stats/leaguegamelog?'
    api_param = {
        'Counter' : counter,
        'DateFrom' :  datefrom,
        'DateTo' : dateto,
        'Direction' : direction,
        'LeagueID' : leagueid,
        'PlayerOrTeam' : playerorteam,
        'Season' : season,
        'SeasonType' : seasontype,
        'Sorter' : sorter,
    }
    u_a = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 Safari/537.36"
    response = requests.get(url,params=api_param,headers={"USER-AGENT":u_a})
    data = response.json()
    return pd.DataFrame(data['resultSets'][0]['rowSet'],columns=data['resultSets'][0]['headers'])


all_rookies = pd.DataFrame.from_csv('rookie_data_1997_2010.csv')

games = []
for season in seasons_string(1978,2017):
    df  = gamelog(season=season,seasontype='All Star')
    games.append(df)
games = pd.concat(games,ignore_index=True)

df = []
ugames = np.unique(games['GAME_ID'].values)
for gameid in ugames:
    temp = boxscoretraditionalv2(gameid)
    temp['SEASON'] = season
    df.append(temp)
df = pd.concat(df,ignore_index=True)

df['PLAYER'] = zip(df['PLAYER_NAME'],df['PLAYER_ID']) # create a unique player column from name + id

# Groupby to count how many times a player appeared in the games
g = df.groupby('PLAYER').size()
g = pd.DataFrame(g,index=g.index,columns=['AllStar_count'])
g['PLAYER_NAME'],g['PLAYER_ID'] = zip(*g.index)
g = g.sort_values(by='AllStar_count',ascending=False).reset_index(drop=True)
all_star = g[g['AllStar_count']>0]
all_star_id = set(all_star['PLAYER_ID'])
print all_star.head(20)
print all_star_id

all_rookies['ALL_STAR']=np.nan
for i in range(len(all_rookies)):
    id = all_rookies.iloc[i]['PLAYER_ID']
    print i
    if id in all_star_id:
        all_rookies.iloc[i,all_rookies.columns.get_loc('ALL_STAR')]=1
    else:
        all_rookies.iloc[i, all_rookies.columns.get_loc('ALL_STAR')]=0

all_rookies.to_csv(path_or_buf='rookies_allstar_labeled.csv')
#train, test = train_test_split(all_rookies, test_size=0.2)
#train.to_csv(path_or_buf='C:/Users/stav/Desktop/study/nba/rookies_allstar_train.csv')
#test.to_csv(path_or_buf='C:/Users/stav/Desktop/study/nba/rookies_allstar_test.csv')