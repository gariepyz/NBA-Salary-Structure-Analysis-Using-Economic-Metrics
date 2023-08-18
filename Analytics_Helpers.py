import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from  math import log
import matplotlib.cm as cm

import warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

with warnings.catch_warnings():
	warnings.simplefilter("ignore", category=RuntimeWarning)

#This class handles interfacing with the SQL db, data analysis, economic metrics and algs.
class Data_Interfacer():
		def __init__(self,db_name):
			'''
			inputs:
				db_name (str) - name of the SQLite DB

			built-in data:
				teams_list (list of str) - All 30 NBA team abbreviations
				years (list of str) - seasons of interest in this SQL db
			'''
			self.db_name = db_name
			self.teams_list = teams_list = [
	"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", 
	"HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", 
	"OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
			self.years = [str(year) for year in range(2011, 2022)]

		def SQL_connect(self,):
			'''
			Establish connection/cursor to SQL db
			'''
			connection = sqlite3.connect(self.db_name)
			cursor = connection.cursor()
			return connection,cursor

		def SQL_disconnect(self,connection,cursor):
			'''
			End connection/cursor to SQL db
			'''
			cursor.close()	
			connection.close()

		def team_totals_from_SQL(self,year,team):
			'''
			Calculate totals/stats for a given year/team
			
			inputs: 
				year (str) - year of interest
				team (str) - team abbrev of interest

			outputs:
				rows (tuple) - SQL value total salaries
			'''
			#SQL query to retrieve data given year/team
			select_query = '''
			SELECT Name, Salary, Team_abbrev,Year
			FROM Salaries_Teams_Players
			WHERE Year = ? AND Team_abbrev = ?;
			'''
			connection,cursor = self.SQL_connect()
			cursor.execute(select_query, (year, team))

			# Fetch all the rows
			rows = cursor.fetchall()
			self.SQL_disconnect(connection,cursor)
			return rows

		def salary_sum(self,rows):
			'''
			Calculate sum			
			
			inputs: 
				rows (df series) - row of salaries

			outputs:
				sum (float) - total salary based on DB values
			'''
			return(sum([row[1] for row in rows ]))

		def get_real_team_cap(self,team,year):
			'''
			Calculate real team cap from HoopsHype.com/salaries
			
			inputs: 
				year (str) - year of interest
				team (str) - Team abbrev of interest

			outputs:
				rows (tuple) - extracted team caps
			'''
		
			# Define SQL query
			select_query = '''
			SELECT Cap
			FROM team_caps
			WHERE Team = ? AND Year = ?;
			'''
			connection,cursor = self.SQL_connect()
			# Execute query
			cursor.execute(select_query, (team, year))
			rows = cursor.fetchall()
			rows = rows[0][0]
			self.SQL_disconnect(connection,cursor)
			return rows

		def get_calculated_team_cap(self,team='GSW',year='2021'):
			'''
			Calculate team cap
			
			inputs: 
				year (str) - year of interest
				team (str) - team abbrev of interest

			outputs:
				rows (tuple) - extracted team caps
			'''

			# Define SQL query
			select_query = '''
			SELECT Salary
			FROM Salaries_Teams_Players
			WHERE Year = ? AND Team_abbrev = ?;
			'''
			connection,cursor = self.SQL_connect()
			
			# Execute query
			cursor.execute(select_query, (year, team))
			rows = cursor.fetchall() 
			rows  = [float(row[0]) for row in rows]

			self.SQL_disconnect(connection,cursor)
			return sum(rows)

		def percent_difference(self,value1, value2):
			"""
			Calculate percent difference between two values.
			
			inputs:
				value1 (float) - the first value.
				value2 (float) - the second value.
				
			outputs:
				diff (float) - the percent difference between the two values.
			"""
			if value1 == 0:
				raise ValueError("first value is 0")
			
			difference = abs(value2 - value1)
			percent_diff = (difference / value1) * 100
			
			return percent_diff  

		def get_db_error(self,plot=True):
			"""
			Calculate % error between db and actual salaries to 
			test reliability of data
			
			inputs: 
				year (str) - year of interest
				team (str) - team abbrev of interest
				
			outputs:
				diff (df) - % errors in a dataframe with name and years
			"""
			diff = []
			team = []
			year=[]
			for j in range(len(self.years)):
				for i in range(30): #30 = number of teams in the NBA
					x1 = self.get_real_team_cap(self.teams_list[i],self.years[j])
					x2 = self.get_calculated_team_cap(self.teams_list[i],self.years[j])
					
					d = self.percent_difference(x1,x2)
					if d>50:
						#data for some teams are missing
						diff.append(0)
						
					else:    
						diff.append(d)
					team.append(self.teams_list[i])
					year.append(self.years[j])
			df_diff = pd.DataFrame({'pdif':diff,'team':team,'year':year})
			df_diff = df_diff.sort_values(by='pdif', ascending=False)

			if plot:
				err= np.mean( df_diff['pdif'] )
				print(f'Mean % error: {err:.2f}%')
				sns.histplot(data=df_diff,x='pdif',bins=25,kde=True)
				plt.xlabel('Percent Difference (%)')
				plt.title('Salary Error Distribution of Combined Databases')
			return df_diff
		
		def calculate_team_mean(self,team,year):
			"""
			Calculate means
			
			inputs:
				value1 (float) - the first value.
				value2 (float) - the second value.
				
			outputs:
				diff (float) - team salary mean
			"""
			q = '''Select Salary from Salaries_Teams_Players 
			Where Team_abbrev = ? AND Year = ?;'''
			connection,cursor = self.SQL_connect()
			cursor.execute(q,(team,year))
			rows = cursor.fetchall()
			rows = [row[0] for row in rows]
			if len(rows) <5:
				raise ValueError("Query likely pulled nothing.")
			self.SQL_disconnect(connection,cursor)
			return sum(rows)

		def calculate_year_means(self,year):
			"""
			Calculate mean of years
			
			inputs:
				year (str) - year of interest

			outputs:
				df (df) - mean of all teams in a given year
			"""
			sums=[]
			team=[]
			years=[]
			for i in self.teams_list:
				try:
					x = self.calculate_team_mean(i,year)
					sums.append(x)
					team.append(i)
					years.append(year)
				except:
					continue

			df = pd.DataFrame({'Total Salary':sums,'Team':team,'Year':years})
			return df

		def calculate_all_means(self,):
			"""
			Calculate means
			
			inputs:
				none
				
			outputs:
				totals (list) - all means
			"""
			
			totals = [self.calculate_year_means(i) for i in self.years]

			return totals

		def visualize_salary_progression(self,save=True):
			"""
			visualize salary distributions
			
			inputs:
				save (bool) - save plot or not
				
			outputs:
				none
			"""
			t = self.calculate_all_means()
			
			ridge = pd.concat(t, ignore_index=True)

			sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
			pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
			g = sns.FacetGrid(ridge, row="Year", hue="Year", aspect=15, height=.5, palette=pal)
			g.map(sns.kdeplot, "Total Salary",
				  bw_adjust=.5, clip_on=False,
				  fill=True, alpha=1, linewidth=1.5)
			g.map(sns.kdeplot, "Total Salary", clip_on=False, color="w", lw=2, bw_adjust=.5)
			g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
			#function to label the plot in axes coordinates
			def label(x, color, label):
				ax = plt.gca()
				ax.text(0, .2, label, fontweight="bold", color=color,
						ha="left", va="center", transform=ax.transAxes)
			g.map(label, "Total Salary")
			g.figure.subplots_adjust(hspace=-.25)
			# Remove axes details that don't play well with overlap
			g.set_titles("")
			g.set(yticks=[], ylabel="")
			g.despine(bottom=True, left=True)
			g.fig.suptitle("Total Salaries Highlighting Distribution", y=0.97)
			# Annotate the furthest data point to the right on the '2021' row
			for ax, year in zip(g.axes.flat, ridge['Year'].unique()):
				if year == 2018:
					max_index = ridge.loc[ridge['Year'] == year, 'Total Salary'].idxmax()
					max_value = ridge.loc[max_index, 'Total Salary']
					ax.scatter(max_value, 0, marker='*', color='red', s=100)
					ax.annotate('Heat (Actual Surprise)', (max_value, 0),
								textcoords="offset points", xytext=(0, 10),
								ha='center', fontsize=10, color='red', fontweight='bold')
			for ax, year in zip(g.axes.flat, ridge['Year'].unique()):
				if year == 2021:
					max_index = ridge.loc[ridge['Year'] == year, 'Total Salary'].idxmax()
					max_value = ridge.loc[max_index, 'Total Salary']
					ax.scatter(max_value, 0, marker='*', color='red', s=100)
					ax.annotate('Wars (No Surprise x2)', (max_value, 0),
								textcoords="offset points", xytext=(0, 10),
								ha='center', fontsize=10, color='red', fontweight='bold')
			for ax, year in zip(g.axes.flat, ridge['Year'].unique()):
				if year == 2020:
					max_index = ridge.loc[ridge['Year'] == year, 'Total Salary'].idxmax()
					max_value = ridge.loc[max_index, 'Total Salary']
					ax.scatter(max_value, 0, marker='*', color='red', s=100)
					ax.annotate('Wars (No Surprise)', (max_value, 0),
								textcoords="offset points", xytext=(0, 10),
								ha='center', fontsize=10, color='red', fontweight='bold')
			if save:
				plt.savefig('Ridge_plot_label_ghub.png')
			return ridge

		def visualize_salary_distribution(self,ridge,save=True):
			"""
			visualize salary distributions in another way
			
			inputs:
				ridge (df) - data to visualize
				save (bool) - save plot or not
				
			outputs:
				none
			"""

			sns.boxplot(x="Year", y="Total Salary",
						palette=["m"],
						data=ridge)
			sns.despine(offset=10, trim=True)
			plt.title('Total Salaries Highlighting Max/Min')
			if save:
				plt.savefig('Boxplot_ghub.png')
		
		def salary_as_a_percent(self,salaries,total):
			"""
			get salaries as percentage of team
			
			inputs:
				salaries (list) - self explanatory
				total (float) - total salary
				
			outputs:
				percents (list) - salaries as a percentage
				number (list) - number of sorted players (1-15)
			"""
			percents = [100 * (i / total) for i in salaries]
			percents = sorted(percents)
			number = list(range(1, 16))

			return percents,number

		def calculate_team_cumulative_salary(self,team,year):
			"""
			calculate cumulative salary distributions
			
			inputs: 
				year (str) - year of interest
				team (str) - team abbrev of interest
				
			outputs:
				rows (tuple) - salary data
				sums (float) - summation of salary
			"""

			connection,cursor = self.SQL_connect()
			q = '''Select Salary from Salaries_Teams_Players 
			Where Team_abbrev = ? AND Year = ?;'''
			cursor.execute(q,(team,year))
			rows = cursor.fetchall()
			rows = [row[0] for row in rows]
			
			if len(rows) <5:
				raise ValueError("Query likely pulled nothing.")
			while len(rows) < 15:
				rows.append(1)	
			rows = sorted(rows )	
			while len(rows) > 15:
				rows = rows[:-1]
			self.SQL_disconnect(connection,cursor)		        
			return rows,sum(rows)

		def plot_cumulative(self,p,team,plot=False):
			"""
			calculate cumulative salary distributions
			
			inputs: 
				year (str) - year of interest
				team (str) - team abbrev of interest
				
			outputs:
				cumulative salary (list) - cumulative salaries
				number (list) - player number (1-15)
			"""
			cumulative_salary = [sum(p[:i+1]) for i in range(len(p))]
			number = list(range(len(p)))
	
			if plot:
				plt.plot(number,cumulative_salary,label=team)
			return cumulative_salary,number
			
		def get_all_percents(self,):
			"""
			self explanatory
			
			inputs: 
				none
				
			outputs:
				d1 (df) - all percent by year and team
			"""
			all_percents=[]
			for i in self.teams_list:
				for j in self.years:
					try:
						row,total = self.calculate_team_cumulative_salary(team=i,year=j)
						p,numbers = self.salary_as_a_percent(row,total)
						team = [i for k in range(1, 16)]
						year1 =[j for k in range(1, 16)]

						df = pd.DataFrame({'Salary_percent': p,'Number':numbers,'Team':team,'Year':year1})
						all_percents.append(df)
					except:
						continue  
			d1 = pd.concat(all_percents, ignore_index=True)
			return d1 #all_percents

		def get_all_lorenz(self,):
			"""
			generate each lorenz curve and append to df
			
			inputs: 
				none
				
			outputs:
				percents (df) - df of all lorenz curves
			"""
			all_lorenz=[]
			for i in self.teams_list:
				for j in self.years:
					try:
						row,total = self.calculate_team_cumulative_salary(team=i,year=j)
						p,numbers = self.salary_as_a_percent(row,total)
						x1,x2 = self.plot_cumulative(p,i)

						team = [i for k in range(1, 16)]
						year1 =[j for k in range(1, 16)]

						df = pd.DataFrame({'Salary_percent': x1,'Number':numbers,'Team':team,'Year':year1})
						all_lorenz.append(df)
					except:
						continue
			percents = pd.concat(all_lorenz, ignore_index=True)
			return percents

		def gini(self,x): #1D np array
			"""
			calculate gini coefficient 			
			
			inputs: 
				none
				
			outputs:
				gini (float) - gini coefficient (0-1)
			"""
			total = 0
			for i, xi in enumerate(x[:-1], 1):
				total += np.sum(np.abs(xi - x[i:]))
			return total / (len(x)**2 * np.mean(x))
		
		def Extract_gini(self,df,team,year):
			"""
			get gini of a specific year/team	

			inputs: 
				year (str) - year of interest
				team (str) - team abbrev of interest
				
			outputs:
				df_extraction (df) - df of gini coef.
			"""
			df_extraction = df[(df['Team'] == team) & (df['Year'] == year)]
			return df_extraction

		def gini_distribution(self,plot=True):
			"""
			generate gini distribution and createa a df
			
			inputs: 
				plot (bool) - plot distribution
				
			outputs:
				df (df) - gini df of all teams/years
			"""
			all_gini=[] 
			team=[]
			year1=[]
			for i in self.teams_list:
				for j in self.years:
					try:
						row,total = self.calculate_team_cumulative_salary(team=i,year=j)                        
						all_gini.append(self.gini( np.array( row ) ))
						team.append(i)
						year1.append(j)
					except:
						continue
										
			df = pd.DataFrame({'Gini Coef': all_gini, 'Team':team,'Year':year1})
						#print(df.shape)
			df['Year'] = df['Year'].astype(float)

			if plot:
				plt.hist(df['Gini Coef'],bins=20)
				plt.title('Gini Coefficient distribution')
			self.gini_dist_df = df
			return df

		def gini_violin(self,):
			"""
			generate gini violin plot
			
			inputs: 
				none
				
			outputs:
				none
			"""
			sns.violinplot(data=self.gini_dist_df,x='Year',y='Gini Coef' ,palette="Set3", bw=.2, cut=1, linewidth=1)

		def calculate_gini_avg(self,):
			"""
			calculate average gini of a year
			
			inputs: 
				none
				
			outputs:
				df_avg (df) - annual gini values
			"""
			avg_gini = [np.mean(self.gini_dist_df[self.gini_dist_df['Year'] == float(i)]['Gini Coef']) for i in self.years]
			year_avg = self.years.copy()

			df_avg = pd.DataFrame({'Average Gini Coef': avg_gini, 'Year': year_avg})
			df_avg = df_avg.drop(5, axis=0)

			self.df_avg = df_avg
			return df_avg

		def gini_dist_summary(self,save=True):
			"""
			plot gini results
			
			inputs: 
				none
				
			outputs:
				none
			"""			
			fig, axes = plt.subplots(1, 3, figsize=(18, 5))

			axes[0].hist(self.gini_dist_df['Gini Coef'], bins=20)
			axes[0].set_title('Gini Coefficient distribution')

			sns.violinplot(data=self.gini_dist_df, x='Year', y='Gini Coef', palette="Set3", bw=.2, cut=1, linewidth=1, ax=axes[1])
			axes[1].set_title('Gini Coefficient by Year')

			axes[2].plot(self.df_avg['Year'], self.df_avg['Average Gini Coef'])
			axes[2].set_title('Average Gini Coefficient over Years')

			plt.tight_layout()
			if save:
				plt.savefig('Gini_stats_summary.png')
			plt.show()
	
		def thiel_T(self,N): #Sensitive to top earners
			"""
			calculate t index Thiel coefficient
			
			inputs: 
				N (list) - salaries
				
			outputs:
				T_t (float) - Thiel_t index
			"""
			u = np.mean(N)
			sums=[]
			for i in range(len(N)):
				sums.append( (N[i]/u)*log( N[i] / u ) )
			return (1/len(N))*sum(sums)
		
		def thiel_L(self,N): #Sensitive to bottom earners
			"""
			calculate L index Thiel coefficient
			
			inputs: 
				N (list) - salaries
				
			outputs:
				T_l (float) - Thiel_L index
			"""

			u = np.mean(N)
			sums=[]
			for i in range(len(N)):
				sums.append( log( N[i] / u ) )
			return (1/len(N))*sum(sums)

		def generate_wins_gini(self,plot=True):
			"""
			generate df of gini and wins 
			
			inputs: 
				plot (bool) - True
				
			outputs:
				merged_df (df) - df of gini coef and win %
			"""

			connection,cursor = self.SQL_connect()

			q = '''
			SELECT TEAM_NAME, YEAR,WIN_PCT FROM team_stats
			'''
			cursor.execute(q)
			results = cursor.fetchall()
			columns = ['TEAM', 'Season', 'Win PCT']
			# Create a DataFrame from the list of tuples
			df_wins_raw = pd.DataFrame(results, columns=columns)
			exclusions = ['1999-00',
			 '2000-01',
			 '2001-02',
			 '2002-03',
			 '2003-04',
			 '2004-05',
			 '2005-06',
			 '2006-07',
			 '2007-08',
			 '2008-09',
			 '2009-10',
			 '2010-11',
			 '2022-23']
			team_name_mapping = {
				'Hawks': 'ATL',
				'Celtics': 'BOS',
				'Cavaliers': 'CLE',
				'Hornets': 'CHA',
				'Pelicans': 'NOP',
				'Bulls': 'CHI',
				'Mavericks': 'DAL',
				'Nuggets': 'DEN',
				'Warriors': 'GSW',
				'Rockets': 'HOU',
				'Clippers': 'LAC',
				'Lakers': 'LAL',
				'Heat': 'MIA',
				'Bucks': 'MIL',
				'Timberwolves': 'MIN',
				'Nets': 'BKN',
				'Knicks': 'NYK',
				'Magic': 'ORL',
				'Pacers': 'IND',
				'76ers': 'PHI',
				'Suns': 'PHX',
				'Trail Blazers': 'POR',
				'Kings': 'SAC',
				'Spurs': 'SAS',
				'Thunder': 'OKC',
				'Raptors': 'TOR',
				'Jazz': 'UTA',
				'Grizzlies': 'MEM',
				'Wizards': 'WAS',
				'Pistons': 'DET',
				'Bobcats': 'CHA'
			}
			df_wins_raw = df_wins_raw[~df_wins_raw['Season'].isin(exclusions)]
			df_wins_raw['Season'] = df_wins_raw['Season'].apply(lambda x: float(x[:4]))
			#df_wins_raw['Season'] = df_wins_raw['Season'].astype(float)
			df_wins_raw['TEAM'] = df_wins_raw['TEAM'].map(team_name_mapping)
			df_wins_raw.rename(columns={'TEAM': 'Team'}, inplace=True)
			df_wins_raw.rename(columns={'Season': 'Year'}, inplace=True)
			merged_df = df_wins_raw.merge(self.gini_dist_df, on=['Team', 'Year'])
			self.SQL_disconnect(connection,cursor)

			self.mdf = merged_df
			return merged_df        
		def generate_thiel_df(self,):
			"""
			combine thiel index and win %
			
			inputs: 
				none
				
			outputs:
				df_thiel (df) - df of thiel and win %
				df_full (df) - df of thiel, gini and win%
			"""
			all_Thiel_L=[]
			all_Thiel_T=[]

			team=[]
			year1=[]
			for i in self.teams_list:
				for j in self.years:
					try:
						row,total = self.calculate_team_cumulative_salary(team=i,year=j)
						
						all_Thiel_L.append(self.thiel_L(  row  ))
						all_Thiel_T.append(self.thiel_T(  row  ))

						team.append(i)
						year1.append(j)
					except:
						continue
			df_t = pd.DataFrame({'Thiel_top Coef': all_Thiel_T,'Thiel_bot Coef':all_Thiel_L, 'Team':team,'Year':year1})
			df_t['Year'] = df_t['Year'].astype(float)
			df_tmerge = self.mdf.merge(df_t, on=['Team', 'Year'])
			self.full_df = df_tmerge
	
			avg_T = [np.mean(df_tmerge[df_tmerge['Year'] == float(i)]['Thiel_top Coef']) for i in self.years]
			avg_L = [np.mean(df_tmerge[df_tmerge['Year'] == float(i)]['Thiel_bot Coef']) for i in self.years]
			year_avg = [float(i) for i in self.years]

			df_thiel = pd.DataFrame({'Average Thiel_top Coef': avg_T, 'Average Thiel_bot Coef': avg_L, 'Year': year_avg})
			df_thiel = df_thiel.drop(5)

			self.df_thiel = df_thiel
			return df_thiel,df_tmerge

		def plot_thiels(self,save=True):
			"""
			plot thiel index results
			
			inputs: 
				none
				
			outputs:
				none
			"""
			sns.set_theme()

			fig, axs = plt.subplots(1, 3, figsize=(15, 5))

			regression_coefficients = np.polyfit(self.mdf['Gini Coef'], self.mdf['Win PCT'], 1)
			regression_line = np.polyval(regression_coefficients, self.mdf['Gini Coef'])
			slope, intercept = regression_coefficients

			correlation1 = self.mdf['Gini Coef'].corr(self.mdf['Win PCT'])
			axs[0].scatter(self.mdf['Gini Coef'], self.mdf['Win PCT'])
			axs[0].plot(self.mdf['Gini Coef'], regression_line, color='red', label='Fitted Line')
			axs[0].set_xlabel('Gini')
			axs[0].set_ylabel('Win PCT')
			axs[0].set_title(f'Equation: y = {slope:.2f}x + {intercept:.2f} | Correlation: {correlation1:.2f}')
			axs[0].legend()

			# Second plot
			X_top = self.df_thiel['Year'].values.reshape(-1, 1)
			y_top = self.df_thiel['Average Thiel_top Coef'].values

			regressor_top = LinearRegression()
			regressor_top.fit(X_top, y_top)
			y_pred_top = regressor_top.predict(X_top)
			r2_top = r2_score(y_top, y_pred_top)

			correlation2 = self.df_thiel['Year'].corr(self.df_thiel['Average Thiel_top Coef'])
			axs[1].scatter(X_top, y_top, label='Data Points')
			axs[1].plot(X_top, y_pred_top, color='red', label='Linear Regression Line')
			axs[1].set_xlabel('Year')
			axs[1].set_ylabel('Average Thiel_top Coef')
			axs[1].set_title(f'Equation: y = {regressor_top.coef_[0]:.4f} * x + {regressor_top.intercept_:.2f} | Correlation : {correlation2:.2f}')
			axs[1].legend()

			# Third plot
			X = self.df_thiel['Year'].values.reshape(-1, 1)
			y = self.df_thiel['Average Thiel_bot Coef'].values

			regressor = LinearRegression()
			regressor.fit(X, y)

			y_pred = regressor.predict(X)
			r2 = r2_score(y, y_pred)
			correlation3 = self.df_thiel['Year'].corr(self.df_thiel['Average Thiel_bot Coef'])
			axs[2].scatter(self.df_thiel['Year'], self.df_thiel['Average Thiel_bot Coef'])
			axs[2].plot(X, y_pred, color='red', label='Linear Regression Line')
			axs[2].set_xlabel('Year')
			axs[2].set_ylabel('Average Thiel_bot Coef')
			axs[2].set_title(f'Line: y = {regressor.coef_[0]:.2f} * x + {regressor.intercept_:.2f} | Correlation: {correlation3:.2f}')
			axs[2].legend()

			second_titles = ["Gino Coef. Vs. Win PCT", "Avg. Thiel_T Over the Years", "Avg. Thiel_L Over the Years"]
			for i, title in enumerate(second_titles):
				axs[i].text(0.5, 1.15, title, ha='center', va='center', color='blue', transform=axs[i].transAxes)

			plt.tight_layout()
			if save:
				plt.savefig('Econ_metrics.png')
			plt.show()

		def champion_filter(self,save=True):
			"""
			extract data from champion teams
			
			inputs: 
				none
				
			outputs:
				conf_champs (df) - conference champions data 
			"""
			sns.set_theme()

			champions = []
			teams_list2 = [
			"MIA",
			"MIA",
			"SAS",
			"GSW",
			"CLE",
			"GSW",
			"GSW",
			"TOR",
			"LAL",
			"MIL",
			"GSW"
			]

			champs = pd.concat(
				[self.full_df[(self.full_df['Year'] == float(year)) & (self.full_df['Team'] == team)] for year, team in zip(self.years, teams_list2)],
				ignore_index=True
			)

			conf_champions = []
			WCF = [
				"OKC",
				"SAS",
				"SAS",
				"GSW",
				"GSW",
				"GSW",
				"GSW",
				"GSW",
				"LAL",
				"PHX",
				"GSW"
			]

			ECF = [
				"MIA",
				"MIA",
				"MIA",
				"CLE",
				"CLE",
				"CLE",
				"CLE",
				"TOR",
				"MIA",
				"MIL",
				"BOS"
			]

			for i in range(len(ECF)):
				for team in [ECF[i], WCF[i]]:
					holder = self.full_df[(self.full_df['Year'] == float(self.years[i])) & (self.full_df['Team'] == team)]
					conf_champions.append(holder)

			conf_champs = pd.concat(conf_champions, ignore_index=True)

			conf_champs = conf_champs.drop(14)
			conf_champs = conf_champs.drop(15)

			fig, ax = plt.subplots(2, 2, figsize=(8, 8))
			x = champs['Year']
			y = champs['Gini Coef']

			coefficients = np.polyfit(x, y, 1)
			poly = np.poly1d(coefficients)
			y_fit = poly(x)
			correlation4 = x.corr(y)

			ax[0, 0].scatter(x, y, label='Data')
			ax[0, 0].plot(x, y_fit, color='red', label='Fitted Line')
			ax[0, 0].set_xlabel('Year')
			ax[0, 0].set_ylabel('Gini Coef')
			ax[0, 0].set_title(f'Line: y = {coefficients[0]:.3f} * x + {coefficients[1]:.2f} | Correlation: {correlation4:.2f}')
			ax[0, 0].text(0.5, 1.15, "Gini Coef vs. Year", ha='center', va='center', color='blue', transform=ax[0, 0].transAxes)
			ax[0, 0].legend()

			no_lakers = champs.drop(7)
			x = no_lakers['Year']
			y = no_lakers['Gini Coef']

			coefficients = np.polyfit(x, y, 1)
			poly = np.poly1d(coefficients)
			y_fit = poly(x)
			correlation5 = x.corr(y)

			ax[0, 1].scatter(x, y, label='Data')
			ax[0, 1].plot(x, y_fit, color='red', label='Fitted Line')
			ax[0, 1].set_xlabel('Year')
			ax[0, 1].set_ylabel('Gini Coef')
			ax[0, 1].set_title(f'Line: y = {coefficients[0]:.3f} * x + {coefficients[1]:.2f} | Correlation: {correlation5:.2f}')
			ax[0, 1].text(0.5, 1.15, "Gini Coef vs. Year (no bubble)", ha='center', va='center', color='blue', transform=ax[0, 1].transAxes)
			ax[0, 1].legend()

			x = conf_champs['Year']
			y = conf_champs['Gini Coef']

			coefficients = np.polyfit(x, y, 1)
			poly = np.poly1d(coefficients)
			y_fit = poly(x)
			correlation6 = x.corr(y)

			ax[1, 0].scatter(x, y, label='Data')
			ax[1, 0].plot(x, y_fit, color='red', label='Fitted Line')
			ax[1, 0].set_xlabel('Year')
			ax[1, 0].set_ylabel('Gini coef')
			ax[1, 0].set_title(f'Line: y = {coefficients[0]:.3f} * x + {coefficients[1]:.2f} | Correlation: {correlation6:.2f}')
			ax[1, 0].text(0.5, 1.15, "Gini Coef vs. Year for Conf. Champs (no bubble)", ha='center', va='center', color='blue', transform=ax[1, 0].transAxes)
			ax[1, 0].legend()

			x = self.mdf['Gini Coef']
			y = self.mdf['Win PCT']
			ax[1, 1].scatter(x, y, label='Rest')
			ax[1, 1].scatter(champs['Gini Coef'], champs['Win PCT'], label='Champs')
			#ax[1, 1].plot(x, slope * x + intercept, color='red', label='Fitted Line')
			ax[1, 1].set_xlabel('Gini Coef')
			ax[1, 1].set_ylabel('Win PCT')
			ax[1, 1].set_title('Champion Teams Highlighted')
			ax[1, 1].legend()

			plt.tight_layout()
			if save:
				plt.savefig('Champion_metric.png')
			plt.show()

			return conf_champs

		def exponential_func(self,x, a, b):
			"""
			self explanatory
			"""
			return a * np.exp(b * x)

		def fit_curve_ex(self,plot=True):
			"""
			fit exp. curve to sample 
			inputs: 
				plot (bool) - plot or not
				
			outputs:
				none
			"""
			sns.set_theme()

			p = self.get_all_lorenz()
			holder = p[p['Year'] == '2011']
			holder = holder[holder['Team'] == 'SAS']
			params, covariance = curve_fit(self.exponential_func,holder['Number'], holder['Salary_percent'])
			predicted_values = self.exponential_func(holder['Number'], *params)
			r_squared = r2_score(holder['Salary_percent'], predicted_values)
			x_fit = np.linspace(1, 15, 100)
			y_fit = self.exponential_func(x_fit, *params)
			if plot:
				# Plot the original data and the fitted curve
				plt.scatter(holder['Number'], holder['Salary_percent'], label='Original Data')
				plt.plot(x_fit, y_fit, label='Fitted Exponential Curve', color='red')
				plt.xlabel('Number')
				plt.ylabel('Salary_percent')
				plt.legend()
				plt.title(f'R-squared: {r_squared:.4f}')  # Display the R-squared value in the plot title
				plt.text(0.5, 1.15, "Exponential fit to Lorenz Curve Sample", ha='center', va='center', color='blue',transform=plt.gca().transAxes)
				plt.show()

		def lorenz_curve_visual(self,save=True):
			"""
			viusalize lorenz curve trends
			
			inputs: 
				none
				
			outputs:
				df_exp_avg (df) - annual lorenz curve fit to exponential
			"""
			lorenz = self.get_all_lorenz()
			lorenz = lorenz.drop_duplicates()

			a1 = []
			a2 = []
			r2 = []
			year1 = []
			for i in self.teams_list:
				for j in self.years:
					try:
						holder = lorenz[lorenz['Year'] == j]
						holder = holder[holder['Team'] == i]

						params, covariance = curve_fit(self.exponential_func,holder['Number'], holder['Salary_percent'])
						predicted_values = self.exponential_func(holder['Number'], *params)
						r_squared = r2_score(holder['Salary_percent'], predicted_values)
						
						a1.append(params[0])
						a2.append(params[1])
						r2.append(r_squared)
						year1.append(float(j))
					except:
						continue
			df_exp = pd.DataFrame({'a1':a1,'a2':a2,'r2':r2,'Year':year1})

			avg_a1 = [np.mean(df_exp[df_exp['Year'] == float(i)]['a1']) for i in self.years]
			avg_a2 = [np.mean(df_exp[df_exp['Year'] == float(i)]['a2']) for i in self.years]
			year1 = [float(i) for i in self.years]

			df_exp_avg = pd.DataFrame({'a1 avg': avg_a1, 'a2 avg': avg_a2, 'Year': year1})
			df_exp_avg = df_exp_avg.drop(5)
			x_values = np.linspace(1, 15, 100)

			cmap = cm.get_cmap('RdBu')

			plt.figure(figsize=(6, 6))
			for index, row in df_exp_avg.iterrows():
				a1 = row['a1 avg']
				a2 = row['a2 avg']
				y_values = a1 * np.exp(a2 * x_values)
				color = cmap((row['Year'] - df_exp_avg['Year'].min()) / (df_exp_avg['Year'].max() - df_exp_avg['Year'].min()))
				plt.plot(x_values, y_values, label=f'Year {int(row["Year"])}', color=color)

			plt.xlabel('Player Count Salaries Ascending')
			plt.ylabel('Avg. Lorenz Curve')
			plt.title('Avg. Lorenz Curve by Year')
			plt.legend()
			plt.grid(True)
			if save:
				plt.savefig('Lorenz_visual.png')
			plt.show()

			self.df_lorenz_functions = df_exp_avg
			return df_exp_avg

		def predict_wins(self,team_name,sal_dists):
			"""
			predict win based on lorenz curve
			
			inputs: 
				team_name (str) - team abbrev
				sal_dists (list) - salaries
				
			outputs:
				none
			"""
			p = self.gini(np.array(sal_dists))
			similarity = 15 
			cutoff = 0.06
			while similarity >10:
				cutoff = cutoff - 0.001
				comp = self.full_df[self.full_df['Gini Coef'] > p-cutoff]
				comp = comp[comp['Gini Coef'] < p+cutoff]
				similarity = comp.shape[0]

			mean=round (82*np.mean(comp['Win PCT']))
			printer = f'{team_name} 2023-2024 Projected Wins: {mean}'
			print(printer)
			plt.hist(comp['Win PCT'],bins=15)
			plt.title('10 Nearest Gini Coef. Teams Win PCT')
			plt.xlabel('Win PCT')
			plt.ylabel('Counts')

		def get_columns(self,table_name):
			"""
			extract column names from SQL table
			
			inputs: 
				table name
				
			outputs:
				none
			"""

			connection,cursor = self.SQL_connect()

			cursor.execute(f"PRAGMA table_info({table_name})")
			columns = cursor.fetchall()
			names=[]
			for column in columns:
				column_name = column[1]
				names.append(column_name)

			self.SQL_disconnect(connection,cursor)
			return names

		def PCA_mapping(self,save=True):
			"""
			PCA mapping to 2 variables of top performing players
			
			inputs: 
				none
				
			outputs:
				none
			"""
			top_players = [
			"Gordon Hayward",
			"DeMar DeRozan",
			"Zach LaVine",
			"Stephen Curry",
			"Chris Paul",
			"James Harden",
			"James Harden",
			"Kobe Bryant",
			"Roy Hibbert",
			"Carmelo Anthony",
			"Robin Lopez",
			"Tobias Harris",
			"Al Horford",
			"Joel Embiid",
			"Mike Conley",
			"Chandler Parsons",
			"Marc Gasol"
			]

			top_years=['2019-20','2021-22','2021-22','2019-20','2018-19','2018-19','2019-20','2015-16','2015-16','2015-16','2015-16','2019-20','2019-20','2019-20','2017-18','2017-18','2017-18']
			wins = [0.667,0.561,0.561,0.231,0.646,0.646,0.611,0.207,0.207,0.390,0.390,0.589,0.589,0.589,0.268,0.268,0.268]
			player_PCA = []
			name = self.get_columns('player_stats')
			connection,cursor = self.SQL_connect()

			for i in range(len(top_players)):
				season_id = top_years[i] 
				partial_player_name = top_players[i]
				query = "SELECT * FROM player_stats WHERE SEASON_ID = ? AND Player_Name LIKE ?"
				cursor.execute(query, (season_id, '%' + partial_player_name + '%'))
				rows = cursor.fetchall()
				for row in rows:
					player_PCA.append(row)

			self.SQL_disconnect(connection,cursor)

			desired = ['PLAYER_AGE',
			 'GP',
			 'MIN',
			 'FGM',
			 'FGA',
			 'FG3M',
			 'FG3A',
			 'FTM',
			 'FTA',
			 'OREB',
			 'DREB',
			 'AST',
			 'STL',
			 'BLK',
			 'TOV',
			 'PTS']

			data_dict = {col: [row[i] for row in player_PCA] for i, col in enumerate(name)}
			df_all = pd.DataFrame(data_dict)
			df_PCA = df_all[desired]

			#PCA normalization + scaling
			player_PCA = StandardScaler().fit_transform(df_PCA) # normalizing the features
			norm_PCA = pd.DataFrame(player_PCA,columns=desired)
			PCA_transform = PCA(n_components=2)
			PrincipalComponents_Player = PCA_transform.fit_transform(player_PCA)

			PCA_wins = np.column_stack((PrincipalComponents_Player, wins))
			df_PCA1 = pd.DataFrame(data=PCA_wins, columns=['PCA1','PCA2', 'Win %'])
			ranks = sorted(list(df_PCA1['Win %'].unique()))

			f, ax = plt.subplots(figsize=(4, 4))
			sns.despine(f, left=True, bottom=True)
			sns.scatterplot(x="PCA1", y="PCA2",
							hue="Win %",
							palette="ch:r=-.2,d=.3_r",
							hue_order=ranks,
							sizes=(4, 8), linewidth=0,
							data=df_PCA1, ax=ax)

			top_indices = df_PCA1.nlargest(2, 'PCA1').index
			for idx in top_indices:
				x_coord = df_PCA1.loc[idx, 'PCA1']
				y_coord = df_PCA1.loc[idx, 'PCA2']
				ax.text(x_coord, y_coord, 'J. Harden', fontsize=12, color='black')

			x_coord = df_PCA1.loc[15, 'PCA1']
			y_coord = df_PCA1.loc[15, 'PCA2']
			ax.text(x_coord, y_coord, 'C. Parsons', fontsize=12, color='black')

			x_coord = df_PCA1.loc[14, 'PCA1']
			y_coord = df_PCA1.loc[14, 'PCA2']
			ax.text(x_coord, y_coord, 'M. Conley', fontsize=12, color='black')

			x_coord = df_PCA1.loc[10, 'PCA1']
			y_coord = df_PCA1.loc[10, 'PCA2']
			ax.text(x_coord, y_coord, 'R. Lopez', fontsize=12, color='black')

			x_coord = df_PCA1.loc[1, 'PCA1']
			y_coord = df_PCA1.loc[1, 'PCA2']
			ax.text(x_coord, y_coord, 'D. DeRozan (CHI)', fontsize=12, color='black')
			plt.title('Principal Component Analysis of Highest Gini Coefficient Teams')
			if save:
				plt.savefig('2_comp_PCA.png')
			plt.show()

		def PCA_win_mapping(self,save=True):
			"""
			1 variable PCA to wins
			
			inputs: 
				none
				
			outputs:
				none
			"""
			top_players = [
			"Gordon Hayward",
			"DeMar DeRozan",
			"Zach LaVine",
			"Stephen Curry",
			"Chris Paul",
			"James Harden",
			"James Harden",
			"Kobe Bryant",
			"Roy Hibbert",
			"Carmelo Anthony",
			"Robin Lopez",
			"Tobias Harris",
			"Al Horford",
			"Joel Embiid",
			"Mike Conley",
			"Chandler Parsons",
			"Marc Gasol"
			]

			top_years=['2019-20','2021-22','2021-22','2019-20','2018-19','2018-19','2019-20','2015-16','2015-16','2015-16','2015-16','2019-20','2019-20','2019-20','2017-18','2017-18','2017-18']
			wins = [0.667,0.561,0.561,0.231,0.646,0.646,0.611,0.207,0.207,0.390,0.390,0.589,0.589,0.589,0.268,0.268,0.268]
			player_PCA = []
			name = self.get_columns('player_stats')
			connection,cursor = self.SQL_connect()

			for i in range(len(top_players)):
				season_id = top_years[i]  # Replace with the desired season ID
				partial_player_name = top_players[i]  # Replace with the desired player's name
				query = "SELECT * FROM player_stats WHERE SEASON_ID = ? AND Player_Name LIKE ?"
				cursor.execute(query, (season_id, '%' + partial_player_name + '%'))
				rows = cursor.fetchall()
				for row in rows:
					player_PCA.append(row)

			self.SQL_disconnect(connection,cursor)

			desired = ['PLAYER_AGE',
			 'GP',
			 'MIN',
			 'FGM',
			 'FGA',
			 'FG3M',
			 'FG3A',
			 'FTM',
			 'FTA',
			 'OREB',
			 'DREB',
			 'AST',
			 'STL',
			 'BLK',
			 'TOV',
			 'PTS']

			data_dict = {col: [row[i] for row in player_PCA] for i, col in enumerate(name)}
			df_all = pd.DataFrame(data_dict)
			df_PCA = df_all[desired]

			player_PCA = StandardScaler().fit_transform(df_PCA) # normalizing the features
			norm_PCA = pd.DataFrame(player_PCA,columns=desired)
			PCA_transform = PCA(n_components=1)
			PrincipalComponents_Player = PCA_transform.fit_transform(player_PCA)

			PCA_wins = np.column_stack((PrincipalComponents_Player, wins))
			df_PCA2 = pd.DataFrame(data=PCA_wins, columns=['PCA1', 'Wins'])
			

			f, ax = plt.subplots(figsize=(4, 4))
			sns.despine(f, left=True, bottom=True)
			sns.scatterplot(x="PCA1", y="Wins",
							#hue="Wins",
							palette="ch:r=-.2,d=.3_r",
							data=df_PCA2, ax=ax)

			x_coord = df_PCA2.loc[5, 'PCA1']
			y_coord = df_PCA2.loc[5, 'Wins']
			ax.text(x_coord, y_coord, 'J. Harden (2018)', fontsize=12, color='black')

			x_coord = df_PCA2.loc[6, 'PCA1']
			y_coord = df_PCA2.loc[6, 'Wins']
			ax.text(x_coord, y_coord, 'J. Harden (2019)', fontsize=12, color='black')

			x_coord = df_PCA2.loc[15, 'PCA1']
			y_coord = df_PCA2.loc[15, 'Wins']
			ax.text(x_coord, y_coord, 'C. Parsons', fontsize=12, color='black')

			x_coord = df_PCA2.loc[1, 'PCA1']
			y_coord = df_PCA2.loc[1, 'Wins']
			ax.text(x_coord, y_coord, 'D. DeRozan (CHI)', fontsize=12, color='black')

			plt.title('Principal Component Analysis of Highest Gini Coefficient Teams')
			if save:
				plt.savefig('PCA_high_Gini_wins.png')
			plt.show()