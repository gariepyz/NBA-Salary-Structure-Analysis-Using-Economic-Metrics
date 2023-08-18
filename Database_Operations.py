from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints.playercareerstats import PlayerCareerStats
from nba_api.stats.static import players
from nba_api.stats.endpoints import teamyearbyyearstats
from nba_api.stats.static import teams
import pickle
import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt
import difflib
import sqlite3

#This class handles interfacing with APIs and data sourcing
class API_Handlers():
		def __init__(self, valid_years):
				'''
				inputs:
				valid_years (str) - list of str of years we want to study
				'''
				self.valid_years = valid_years

		def extract_player_seasons(self):
			'''
			extract player stats by season from nba api

			inputs:
				none
			
			outputs:
				valid_players (tuple) - player stats of interested seasons
				api_players (tuple) - all players
			'''
			valid_players = []
			all_players = players.get_players()

			#parse all players and check if they are in desired seasons
			for i in range(len(all_players)):
					sample_player = all_players[i]
					sample_stats = playercareerstats.PlayerCareerStats(player_id=sample_player['id'])					
					sample_df = sample_stats.get_data_frames()[0]
					szns_played = list(sample_df['SEASON_ID'])
					szns_played = [item[:4] for item in szns_played]
		
					valid = False
					for j in szns_played:
							if j in self.valid_years:
									valid = True			
					if valid:
							valid_players.append(sample_player)
					if i % 500 == 0:
							print(f'parsed {i} / {len(all_players)} players')
			#extract list of player names
			api_players = [player['full_name'] for player in valid_players]

			return valid_players, api_players     

		def generate_stats_pickle(self, valid_players):
			'''
			extract player stats into a pickle

			inputs:
				none
			
			outputs:
				none
			'''
			name = 'valid_players.pkl'
			self.pickle_valid = name
			with open(name, 'wb') as pickle_file:
					pickle.dump(valid_players, pickle_file)

		def remove_special_chars(self, value):
			'''
			self explanatory
			'''
				return unidecode(value)

		def import_hoopshype_data(self):
			'''
			extract player seasons and saved matched players to csv for later use

			inputs:
				none
			
			outputs:
				ref_players (tuple) - player salaries
				df_sal (df) - df of player salaries
			'''
			#read in csv (extracted from website API to make things easier as tutorial)
			df_sal = pd.read_csv('backup_datasets/salaries.csv')
			ref_players = list(df_sal['Player'].unique())
			valid_players,api_players = self.extract_player_seasons()

			not_in_api = [i for i in ref_players if i not in api_players]
							
			print(f'names not in the api: {len(not_in_api)}') 
			#track top string matching combinations to link missing names between 2 databases
			top_scores =[]
			p1=[]
			p2=[]

			for i in not_in_api:
					
					p1.append(i)
					best_match = 0
					name_match = None
					
					for j in api_players:
							
							match = difflib.SequenceMatcher(None, i,j).ratio()
							if match > best_match:
									best_match = match
									name_match = j
									
					top_scores.append(best_match)
					p2.append(name_match)    

			player_map = {}
			for i in range(len(not_in_api)):
					
					print(f'{p1[i]} | {p2[i]} | {str(top_scores[i])[:4]}')
					
					yn = input()
					if yn == 'y':
							player_map[p1[i]] = p2[i]
							
			filtered_df = df_sal[df_sal['Player'].isin(not_in_api)]

			drops=[]
			for i in range(filtered_df.shape[0]):
					if filtered_df['Salary'].iloc[i] < 2500000:
							 drops.append(filtered_df['Player'].iloc[i] )

			drops = list(set(drops))
			#filter the csv once again
			df_sal = pd.read_csv('backup_datasets/salaries.csv')
			df_sal = df_sal[df_sal['Season'] != '2023/24']
			df_sal = df_sal[~df_sal['Player'].isin(drops)]
			df_sal['Player'] = df_sal['Player'].replace(player_map, regex=False)
			df_sal = df_sal[~df_sal['Player'].isin(p1)]
			ref_players = list(df_sal['Player'].unique())
			df_sal.to_csv('Proper_salary_list.csv')
			return ref_players, df_sal

		def player_matching(self, ref_players,api_players,df_sal, plot=True, cutoff=2500000, pickler=True):
			'''
			match players between 2 databases

			inputs:
				ref_players/api_players (tuple) - stats of the 2 databases of players
				df_sal (df) - salaries of players
				cutoff (float) - salary cutoff if mismatch occurs 
				pickler (bool) - save pickle 
			
			outputs:
				filtered_api (tuple) - matchd db of stats and salaries
			'''

			not_in_api = []
			with open('valid_players.pkl', 'rb') as pickle_file:
				data = pickle.load(pickle_file)
				
			not_in_api = [i for i in ref_players if i not in api_players]

			print(f'names not in the api: {len(not_in_api)}')
			#perform same filtering but with the second database now
			top_scores = []
			p1 = []
			p2 = []

			for i in not_in_api:
					p1.append(i)
					best_match = 0
					name_match = None

					for j in api_players:
							match = difflib.SequenceMatcher(None, i, j).ratio()
							if match > best_match:
									best_match = match
									name_match = j

					top_scores.append(best_match)
					p2.append(name_match)

			plt.hist(top_scores, bins=25)
			plt.title('Name Matching Scores (0-1)')

			player_mapping = {}
			for i in range(len(p1)):
					print(f'{p1[i]} | {p2[i]} | {str(top_scores[i])[:4]}')

					yn = input()
					if yn == 'y':
							player_mapping[p1[i]] = p2[i]

			if plot:
					filtered_df = df_sal[df_sal['Player'].isin(not_in_api)]
					plt.hist(filtered_df['Salary'], bins=50)
					plt.locator_params(axis='x', nbins=20)

					drops = []
					for i in range(filtered_df.shape[0]):
							if filtered_df['Salary'].iloc[i] < cutoff:
									drops.append(filtered_df['Player'].iloc[i])

					drops = list(set(drops))

			df_sal = pd.read_csv('backup_datasets/Proper_salary_list.csv')
			sal_names = list(set(df_sal['Player']))

			filtered_api_data = []

			duplicate_check = []

			for i in range((len(data))):
				pname = data[i]['full_name']
				if pname not in duplicate_check:
					duplicate_check.append(pname)
					if pname in sal_names:
							filtered_api_data.append(data[i])

			if pickler:
				with open('Proper_api_player_list_ghub.pkl', 'wb') as pickle_file:
					pickle.dump(filtered_api_data, pickle_file)

			return filtered_api_data

		def generate_sqlite3_db(self, db_name):

			'''
			create sqlite3 db 

			inputs:
				db_name (str) - name of db
			
			outputs:
				none
			'''
			self.db_name = db_name
			connection = sqlite3.connect(self.db_name)
			cursor = connection.cursor()
			cursor.close()
			connection.close()

		def generate_player_stats_table(self,table_name,player_data):
			'''
			create player stats table in SQL db

			inputs:
				table_name (str) - table name
				player_data (tuple) - player stats to add to table
			
			outputs:
				none
			'''

			connection = sqlite3.connect(self.db_name)
			cursor = connection.cursor()

			create_table1_sql = f'''
			    CREATE TABLE IF NOT EXISTS {table_name} (
			        id INTEGER PRIMARY KEY,
			        Player_Name TEXT,
			        PLAYER_ID INTEGER,
			        SEASON_ID TEXT,
			        LEAGUE_ID TEXT,
			        TEAM_ID INTEGER,
			        TEAM_ABBREVIATION TEXT,
			        PLAYER_AGE INTEGER,
			        GP INTEGER,
			        GS INTEGER,
			        MIN INTEGER,
			        FGM INTEGER,
			        FGA INTEGER,
			        FG_PCT REAL,
			        FG3M INTEGER,
			        FG3A INTEGER,
			        FG3_PCT REAL,
			        FTM INTEGER,
			        FTA INTEGER,
			        FT_PCT REAL,
			        OREB INTEGER,
			        DREB INTEGER,
			        REB INTEGER,
			        AST INTEGER,
			        STL INTEGER,
			        BLK INTEGER,
			        TOV INTEGER,
			        PF INTEGER,
			        PTS INTEGER,
			        UNIQUE(Player_Name, SEASON_ID, TEAM_ABBREVIATION)
			    );
			'''

			cursor.execute(create_table1_sql)
			for i in range(len(player_data)):
				single_career = PlayerCareerStats(player_id=player_data[i]['id'])
				career_stats = single_career.get_dict()
				relevant_info = career_stats['resultSets'][0]
				for j in range(len(relevant_info['rowSet'])):
						stats = [player_data[i]['full_name']] + relevant_info['rowSet'][j]
						sql_injection = '''INSERT OR IGNORE INTO player_stats (
				Player_Name, PLAYER_ID, SEASON_ID, LEAGUE_ID, TEAM_ID, TEAM_ABBREVIATION, PLAYER_AGE,
				GP, GS, MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB,
				AST, STL, BLK, TOV, PF, PTS) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
						'''
						cursor.execute(sql_injection, stats)
				if i % 250 == 0:
						print(f'SQL table build progress: {i}/{len(player_data)}')

			connection.commit()
			cursor.close()
			connection.close()

		def close_connection(self):
			'''
			self explanatory
			'''
			cursor.close()
			connection.close()

		def generate_salary_table(self, table_name):
			'''
			create salary stats table in SQL db

			inputs:
				table_name (str) - table name
			
			outputs:
				none
			'''

			df_sal = pd.read_csv('backup_datasets/Matched_datasets/Proper_salary_list.csv')
			columns_to_drop = ['Unnamed: 0', 'Salary adjusted by inflation']
			df_sal = df_sal.drop(columns_to_drop, axis=1)
			sal_names = list(set(df_sal['Player']))

			create_table_sql = f'CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY,Player_Name TEXT,Salary INTEGER ,Season TEXT,UNIQUE(Player_Name,Season))'
			cursor.execute(create_table_sql)

			for i in range(df_sal.shape[0]):

					line = df_sal.iloc[i]
					injection = [line['Player'], int(line['Salary']), line['Season']]

					insert_sql = '''
					INSERT OR IGNORE INTO player_salaries (Player_Name, Salary, Season)
					VALUES (?, ?, ?)
					'''

					cursor.execute(insert_sql, injection)
					if i % 1000 == 0:
							print(f'SQL table build progress: {i}/{df_sal.shape[0]}')

			cursor.close()
			connection.close()

		def generate_team_stats_table(self, table_name):
			'''
			create player stats table in SQL db

			inputs:
				table_name (str) - table name
			
			outputs:
				none
			'''
			connection = sqlite3.connect(self.db_name)
			cursor = connection.cursor()

			create_table_query = f'''
			    CREATE TABLE {table_name} (
			        TEAM_ID TEXT,
			        TEAM_CITY TEXT,
			        TEAM_NAME TEXT,
			        YEAR TEXT,
			        GP INTEGER,
			        WINS INTEGER,
			        LOSSES INTEGER,
			        WIN_PCT REAL,
			        CONF_RANK INTEGER,
			        DIV_RANK INTEGER,
			        PO_WINS INTEGER,
			        PO_LOSSES INTEGER,
			        CONF_COUNT REAL,
			        DIV_COUNT INTEGER,
			        NBA_FINALS_APPEARANCE TEXT,
			        FGM INTEGER,
			        FGA INTEGER,
			        FG_PCT REAL,
			        FG3M INTEGER,
			        FG3A INTEGER,
			        FG3_PCT REAL,
			        FTM INTEGER,
			        FTA INTEGER,
			        FT_PCT REAL,
			        OREB INTEGER,
			        DREB INTEGER,
			        REB INTEGER,
			        AST INTEGER,
			        PF INTEGER,
			        STL INTEGER,
			        TOV INTEGER,
			        BLK INTEGER,
			        PTS INTEGER,
			        PTS_RANK INTEGER,
			        PRIMARY KEY (TEAM_NAME, YEAR)
			    );
			'''

			cursor.execute(create_table_query)
			connection.commit()

			all_teams = teams.get_teams()
			sample_team = teamyearbyyearstats.TeamYearByYearStats(team_id=all_teams[0]['id'])
			sample_df = sample_team.get_data_frames()[0]
			valid_years = list(sample_df['YEAR'].unique())[-24:]

			for j in range(len(all_teams)):
				sample_team = teamyearbyyearstats.TeamYearByYearStats(team_id=all_teams[j]['id'])
				sample_df = sample_team.get_data_frames()[0]
				sample_df = sample_df.fillna('N/A')

				relevant_years = sample_df[sample_df['YEAR'].isin(valid_years)]

				for i in range(relevant_years.shape[0]):
					l = relevant_years.iloc[i]
					teams_stats = [
							int(l['TEAM_ID']),
							str(l['TEAM_CITY']),
							str(l['TEAM_NAME']),
							str(l['YEAR']),
							int(l['GP']),
							int(l['WINS']),
							int(l['LOSSES']),
							float(l['WIN_PCT']),
							int(l['CONF_RANK']),
							int(l['DIV_RANK']),
							int(l['PO_WINS']),
							int(l['PO_LOSSES']),
							int(l['CONF_COUNT']),
							int(l['DIV_COUNT']),
							str(l['NBA_FINALS_APPEARANCE']),
							int(l['FGM']),
							int(l['FGA']),
							float(l['FG_PCT']),
							int(l['FG3M']),
							int(l['FG3A']),
							float(l['FG3_PCT']),
							int(l['FTM']),
							int(l['FTA']),
							float(l['FT_PCT']),
							int(l['OREB']),
							int(l['DREB']),
							int(l['REB']),
							int(l['AST']),
							int(l['PF']),
							int(l['STL']),
							int(l['TOV']),
							int(l['BLK']),
							int(l['PTS']),
							int(l['PTS_RANK']),
					]

					insert_query = '''
					INSERT OR REPLACE INTO team_stats (
							TEAM_ID, TEAM_CITY, TEAM_NAME, YEAR, GP, WINS, LOSSES, WIN_PCT, CONF_RANK,
							DIV_RANK, PO_WINS, PO_LOSSES, CONF_COUNT, DIV_COUNT, NBA_FINALS_APPEARANCE,
							FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB,
							AST, PF, STL, TOV, BLK, PTS, PTS_RANK
					) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			'''
					cursor.execute(insert_query, teams_stats)

					if j % 2 == 0:
						print(f'SQL table build progress: {j}/{len(all_teams)}')

			cursor.close()
			connection.close()
		def generate_salary_team_player_table(self,table_name):
			'''
			create team stats table in SQL db

			inputs:
				table_name (str) - table name
			
			outputs:
				none
			'''

			name=[]
			sal=[]
			team=[]
			year=[]

			for j in range(len(file_names)):
			    df = pd.read_csv('backup_datasets/years/'+file_names[j])
			    
			    for i in range(len(list(df['name']))):
			        line=df.iloc[i]
			        name.append(line['name'])
			        sal.append(line['sal'])
			        team.append(line['team'])
			        year.append(file_names[j][:4])
			sal_df= pd.DataFrame({'Name':name,'Salary':sal,'Team_abbrev':team,'Year':year})
			sal_df = sal_df.drop_duplicates()

			create_table_query = f'''
			    CREATE TABLE IF NOT EXISTS {table_name} (
			        id INTEGER PRIMARY KEY,
			        Name TEXT,
			        Salary REAL,
			        Team_abbrev TEXT,
			        Year TEXT,
			        UNIQUE (Name, Team_abbrev, Year)
			    );
			'''


			cursor.execute(create_table_query)
			for i in range(sal_df.shape[0]):

			    line = sal_df.iloc[i]
			    injection = [line['Name'], int(line['Salary']),line['Team_abbrev'],line['Year']]
			    
			    insert_sql = '''
			    INSERT INTO Salaries_Teams_Players (Name, Salary, Team_abbrev,Year)
			    VALUES (?, ?, ?,?)
			    '''
			    
			    cursor.execute(insert_sql, injection)
			    if i%500 == 0:
			        print(f'SQL table build progress: {i}/{sal_df.shape[0]}')
		def generate_team_caps_table(self,table_name):
			'''
			create team caps table in SQL db

			inputs:
				table_name (str) - table name
			
			outputs:
				none
			'''
			df_team_cap = pd.read_csv('backup_datasets/teamcaps - Sheet1.csv')
			df_team_cap['Cap'] = df_team_cap['Cap'].str.replace(',', '').astype(float)
			df_team_cap['Team'] = df_team_cap['Team'].map(nba_teams_abbreviations)

			create_table_query = f'''
			    CREATE TABLE IF NOT EXISTS {table_name} (
			        id INTEGER PRIMARY KEY,
			        Team TEXT,
			        Cap REAL,
			        Year TEXT,
			        UNIQUE (Team, Year)
			    );
			'''


			cursor.execute(create_table_query)

			connection.commit()
			
			#for i in range(len(player_data)):
			for i in range(df_team_cap.shape[0]):

			    line = df_team_cap.iloc[i]
			    injection = [line['Team'], int(line['Cap']),str(line['Year'])]
			    
			    insert_query = '''
			INSERT INTO team_caps (Team, Cap, Year)
			VALUES (?, ?, ?);
			'''
			    
			    cursor.execute(insert_query, injection)
			    if i%50 == 0:
			        print(f'SQL table build progress: {i}/{df_team_cap.shape[0]}')