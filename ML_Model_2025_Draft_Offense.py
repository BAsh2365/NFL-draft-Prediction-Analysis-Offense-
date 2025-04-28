import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from thefuzz import process
import io  # Import io for StringIO

def fetch_draft_picks(url: str) -> pd.DataFrame:
    """
    Scrape Yahoo Sports 2025 draft picks by team.
    Returns a DataFrame with columns:
      - Drafted_Team
      - pick_no (e.g. "1.01")
      - Position (e.g. "QB")
      - player_name
    """

    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')

    rows = []
    # Regex for lines like: 1. Tennessee Titans: Cam Ward, QB, Miami
    pattern = re.compile(
        r'(?P<pick>\d+)\.\s+(?P<team>.+?):\s+(?P<player>.+?),\s+(?P<pos>[A-Z]+),\s+(?P<college>.+)', re.IGNORECASE
    )

    # Yahoo uses <li> for picks, but sometimes <p> or just text, so let's try all <li> and <p>
    for tag in soup.find_all(['li', 'p']):
        text = tag.get_text(" ", strip=True)
        m = pattern.match(text)
        if m:
            pick_no = m.group('pick')
            team = m.group('team').strip()
            player = m.group('player').strip()
            pos = m.group('pos').strip()
            # For round, infer from pick number (1-32: round 1, 33-64: round 2, etc.)
            pick_int = int(pick_no)
            round_num = (pick_int - 1) // 32 + 1
            pick_in_round = (pick_int - 1) % 32 + 1
            pick_no_str = f"{round_num}.{str(pick_in_round).zfill(2)}"
            rows.append({
                'Drafted_Team': team,
                'pick_no': pick_no_str,
                'Position': pos,
                'player_name': player
            })
    return pd.DataFrame(rows)

# 1. Scrape the draft picks
url = "https://sports.yahoo.com/nfl/article/2025-nfl-draft-full-round-by-round-list-of-all-257-picks-from-cam-ward-to-shedeur-sanders-to-kobee-minor-230821320.html"
df_picks = fetch_draft_picks(url)
df_picks.to_csv("C://Users//bharg//Downloads//nfl_2025_draft_by_team.csv", index=False)

# If you only want player name, position, and team in the CSV:
df_picks_simple = df_picks[['player_name', 'Position', 'Drafted_Team']]
df_picks_simple.to_csv("C://Users//bharg//Downloads//nfl_2025_draft_players_teams.csv", index=False)

# 2. Load your draft-prospect sheet (Sheet 1)
prospects = pd.read_excel(
    "C://Users//bharg//Downloads//Top Prospects for the 2025 NFL Draft.xlsx",
    sheet_name=0
)

# 3. Filter to offensive positions only
offensive_positions = ['QB','RB','WR','TE','OT','IOL','C','FB']
off_prospects = prospects[
    prospects['Position'].isin(offensive_positions)
].copy()

# 4. Merge the scraped picks into your prospects
#    Adjust 'Name' below if your column is called something else
off_prospects['player_name'] = off_prospects['Name'].str.strip()
df_picks['player_name'] = df_picks['player_name'].str.strip()

# ADDED: Ensure we maintain the original Name column
off_prospects['original_name'] = off_prospects['Name']

def clean_name(name):
    return str(name).strip().lower().replace('.', '').replace(',', '')

off_prospects['player_name_clean'] = off_prospects['player_name'].apply(clean_name)
df_picks['player_name_clean'] = df_picks['player_name'].apply(clean_name)

def fuzzy_merge(df1, df2, key1, key2, threshold=90):
    matches = df1[key1].apply(
        lambda x: process.extractOne(x, df2[key2], score_cutoff=threshold)
    )
    df1['match'] = matches.apply(lambda x: x[0] if x else None)
    return df1.merge(df2, left_on='match', right_on=key2, how='left')

# Replace the current merge with fuzzy matching
merged = fuzzy_merge(
    off_prospects, 
    df_picks[['Drafted_Team','player_name_clean','pick_no']], 
    'player_name_clean', 
    'player_name_clean',
    threshold=85  # Adjust this threshold if needed (0-100)
)

merged = merged[merged['pick_no'].notnull()]
print("Rows with valid pick_no after merge:", len(merged))

# Print merge results
print("\nMerge results:")
print(merged[['player_name', 'Drafted_Team', 'pick_no']].head(20))
print(f"\nNumber of matches found: {merged['Drafted_Team'].notnull().sum()}")

# 5. Convert pick_no ("1.01") into overall pick integer
def convert_pick(pick: str) -> float:
    try:
        rd, num = pick.split('.')
        return (int(rd) - 1) * 32 + int(num)
    except:
        return np.nan

merged['draft_pick'] = merged['pick_no'].apply(convert_pick)

# 6. Scrape team tendencies from nfeloapp.com instead of loading CSV
def fetch_team_tendencies(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')
    table = soup.find('table')
    if table is None:
        raise RuntimeError("Couldn't find table on nfeloapp.com page.")
    
    # Fix for FutureWarning - use StringIO
    html_str = str(table)
    df = pd.read_html(io.StringIO(html_str))[0]
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i and str(i) != 'nan']) for col in df.columns.values]
    # Rename the first column to 'Drafted_Team'
    df = df.rename(columns={df.columns[0]: 'Drafted_Team'})
    print("Team tendencies columns AFTER FLATTEN:", df.columns.tolist())   # Debug print
    return df

team_tendencies_url = "https://www.nfeloapp.com/nfl-power-ratings/nfl-team-tendencies/"
team_tendencies = fetch_team_tendencies(team_tendencies_url)

# Print team names from both DataFrames to check for mismatches
print("\nTeam names in merged data:", sorted(merged['Drafted_Team'].unique()))
print("\nTeam names in tendencies data:", sorted(team_tendencies['Drafted_Team'].unique()))

# Create a mapping table for the specific team tendencies format
# This function extracts team name from the tendencies format
def extract_team_from_tendencies(team_code):
    # Example conversions:
    # "10BroncosDEN" -> "denver broncos"
    # "1CardinalsARI" -> "arizona cardinals"
    
    # Remove any digits at the beginning
    team_name = re.sub(r'^[0-9]+', '', team_code)
    
    # Remove the last 2-3 letters (typically the abbreviation)
    team_name = re.sub(r'[A-Z]{2,3}$', '', team_name)
    
    # Handle special cases
    team_name = team_name.lower()
    
    # Convert remaining text to proper team names
    team_mapping = {
        'cardinals': 'arizona cardinals',
        'falcons': 'atlanta falcons',
        'ravens': 'baltimore ravens',
        'bills': 'buffalo bills',
        'panthers': 'carolina panthers',
        'bears': 'chicago bears',
        'bengals': 'cincinnati bengals',
        'browns': 'cleveland browns',
        'cowboys': 'dallas cowboys',
        'broncos': 'denver broncos',
        'lions': 'detroit lions',
        'packers': 'green bay packers',
        'texans': 'houston texans',
        'colts': 'indianapolis colts',
        'jaguars': 'jacksonville jaguars',
        'chiefs': 'kansas city chiefs',
        'chargers': 'los angeles chargers',
        'rams': 'los angeles rams',
        'dolphins': 'miami dolphins',
        'vikings': 'minnesota vikings',
        'patriots': 'new england patriots',
        'saints': 'new orleans saints',
        'giants': 'new york giants',
        'jets': 'new york jets',
        'raiders': 'las vegas raiders', # Note: tendencies might still show Oakland
        'eagles': 'philadelphia eagles',
        'steelers': 'pittsburgh steelers',
        '49ers': 'san francisco 49ers',
        'seahawks': 'seattle seahawks',
        'buccaneers': 'tampa bay buccaneers',
        'titans': 'tennessee titans',
        'commanders': 'washington commanders',
    }
    
    for key, value in team_mapping.items():
        if key in team_name:
            return value
    
    return team_name  # Return as-is if no match

# Extract the base team name from draft data
def extract_team_from_draft(team_name):
    # Remove parenthetical expressions like "(from Panthers)"
    team_name = re.sub(r'\s*\(.*?\)', '', str(team_name))
    # Remove "via X" expressions
    team_name = re.sub(r'\s+via\s+.*', '', team_name, flags=re.IGNORECASE)
    # Remove "from X" expressions
    team_name = re.sub(r'\s+from\s+.*', '', team_name, flags=re.IGNORECASE)
    # Convert to lowercase for consistency
    return team_name.strip().lower()

# Apply the extraction/standardization to both DataFrames
merged['team_std'] = merged['Drafted_Team'].apply(extract_team_from_draft)
team_tendencies['team_std'] = team_tendencies['Drafted_Team'].apply(extract_team_from_tendencies)

# Check if this helped with matching
print("\nStandardized team names in merged data:", sorted(merged['team_std'].unique()))
print("\nStandardized team names in tendencies data:", sorted(team_tendencies['team_std'].unique()))

# Create a mapping table for specific difficult cases
team_mapping_fixes = {
    # Add any specific mappings needed between your datasets
    'las vegas raiders': 'las vegas raiders',
    'los angeles chargers': 'los angeles chargers',
    'los angeles rams': 'los angeles rams',
    'ny giants': 'new york giants',
    'ny jets': 'new york jets',
    # Add more mappings as needed
}

# Apply fixes to standardized team names
for old_name, new_name in team_mapping_fixes.items():
    merged.loc[merged['team_std'] == old_name, 'team_std'] = new_name
    team_tendencies.loc[team_tendencies['team_std'] == old_name, 'team_std'] = new_name

# Merge using standardized team names
df = merged.merge(
    team_tendencies,
    on='team_std',
    how='left'
)

# Check if the merge worked correctly
print(f"\nMerged rows with team tendencies: {df['Run Pass Ratio_Pass Rate'].notna().sum()} / {len(df)}")

# If still no matches, try fuzzy matching on team names
if df['Run Pass Ratio_Pass Rate'].notna().sum() == 0:
    print("\nAttempting fuzzy matching on team names...")
    
    # Create a mapping dictionary using fuzzy matching
    merged_teams = sorted(merged['team_std'].unique())
    tendencies_teams = sorted(team_tendencies['team_std'].unique())
    
    team_mapping = {}
    for draft_team in merged_teams:
        best_match = process.extractOne(draft_team, tendencies_teams, score_cutoff=75)
        if best_match:
            team_mapping[draft_team] = best_match[0]
            print(f"Mapped '{draft_team}' to '{best_match[0]}'")
    
    # Apply the mapping
    merged['fuzzy_team_match'] = merged['team_std'].map(team_mapping)
    
    # Try again with fuzzy matches
    df = merged.merge(
        team_tendencies,
        left_on='fuzzy_team_match',
        right_on='team_std',
        how='left',
        suffixes=('_x', '_y')
    )
    
    # Check if this worked better
    print(f"\nMerged rows with team tendencies after fuzzy matching: {df['Run Pass Ratio_Pass Rate'].notna().sum()} / {len(df)}")

# If still no luck, try direct mapping for common team names
if df['Run Pass Ratio_Pass Rate'].notna().sum() == 0:
    print("\nAttempting direct team name mapping...")
    
    # Create a direct mapping dictionary for all teams
    direct_team_mapping = {
        'arizona cardinals': ['arizona', 'cardinals', 'ari'],
        'atlanta falcons': ['atlanta', 'falcons', 'atl'],
        'baltimore ravens': ['baltimore', 'ravens', 'bal'],
        'buffalo bills': ['buffalo', 'bills', 'buf'],
        'carolina panthers': ['carolina', 'panthers', 'car'],
        'chicago bears': ['chicago', 'bears', 'chi'],
        'cincinnati bengals': ['cincinnati', 'bengals', 'cin'],
        'cleveland browns': ['cleveland', 'browns', 'cle'],
        'dallas cowboys': ['dallas', 'cowboys', 'dal'],
        'denver broncos': ['denver', 'broncos', 'den'],
        'detroit lions': ['detroit', 'lions', 'det'],
        'green bay packers': ['green bay', 'packers', 'gb'],
        'houston texans': ['houston', 'texans', 'hou'],
        'indianapolis colts': ['indianapolis', 'colts', 'ind'],
        'jacksonville jaguars': ['jacksonville', 'jaguars', 'jax', 'jac'],
        'kansas city chiefs': ['kansas city', 'chiefs', 'kc'],
        'las vegas raiders': ['las vegas', 'oakland', 'raiders', 'oak', 'lv'],
        'los angeles chargers': ['los angeles', 'la', 'chargers', 'lac', 'sd'],
        'los angeles rams': ['los angeles', 'la', 'rams', 'lar'],
        'miami dolphins': ['miami', 'dolphins', 'mia'],
        'minnesota vikings': ['minnesota', 'vikings', 'min'],
        'new england patriots': ['new england', 'patriots', 'ne'],
        'new orleans saints': ['new orleans', 'saints', 'no'],
        'new york giants': ['ny giants', 'giants', 'nyg'],
        'new york jets': ['ny jets', 'jets', 'nyj'],
        'philadelphia eagles': ['philadelphia', 'eagles', 'phi'],
        'pittsburgh steelers': ['pittsburgh', 'steelers', 'pit'],
        'san francisco 49ers': ['san francisco', '49ers', 'sf'],
        'seattle seahawks': ['seattle', 'seahawks', 'sea'],
        'tampa bay buccaneers': ['tampa bay', 'buccaneers', 'tb'],
        'tennessee titans': ['tennessee', 'titans', 'ten'],
        'washington commanders': ['washington', 'commanders', 'was', 'wsh']
    }
    
    # Extract team IDs for easier matching
    team_tendencies['team_key'] = team_tendencies['Drafted_Team'].str.lower()
    
    # Create a dictionary to map from team codes to team full names
    team_code_to_name = {}
    for full_name, identifiers in direct_team_mapping.items():
        for identifier in identifiers:
            for team_code in team_tendencies['team_key']:
                if identifier in team_code.lower():
                    team_code_to_name[team_code] = full_name
                    break
    
    # Print the mapping
    print("\nTeam code to name mapping:")
    for code, name in team_code_to_name.items():
        print(f"{code} -> {name}")
    
    # Add full team names to tendencies data
    team_tendencies['full_team_name'] = team_tendencies['team_key'].map(team_code_to_name)
    
    # Now merge based on these full team names
    merged['team_name_for_merge'] = merged['team_std'].apply(
        lambda x: next((k for k, v in direct_team_mapping.items() if any(id in x.lower() for id in v)), x)
    )
    
    df = merged.merge(
        team_tendencies,
        left_on='team_name_for_merge',
        right_on='full_team_name',
        how='left',
        suffixes=('_x', '_y')
    )
    
    # Check success
    print(f"\nMerged rows with team tendencies after direct mapping: {df['Run Pass Ratio_Pass Rate'].notna().sum()} / {len(df)}")
    
    # If we have successfully merged at least some teams, show which ones worked
    if df['Run Pass Ratio_Pass Rate'].notna().sum() > 0:
        successful_teams = df[df['Run Pass Ratio_Pass Rate'].notna()]['team_name_for_merge'].unique()
        print(f"\nSuccessful team matches: {sorted(successful_teams)}")

def height_to_inches(height_str):
    if pd.isnull(height_str):
        return np.nan
    # Match patterns like 6' 5 3/4"
    m = re.match(r"(\d+)'[\s]*(\d+)?(?:\s*(\d+)/(\d+))?\"", str(height_str))
    if not m:
        # Try to match patterns like 6' 5"
        m2 = re.match(r"(\d+)'[\s]*(\d+)?", str(height_str))
        if not m2:
            return np.nan
        feet = int(m2.group(1))
        inches = int(m2.group(2)) if m2.group(2) else 0
        return feet * 12 + inches
    feet = int(m.group(1))
    inches = int(m.group(2)) if m.group(2) else 0
    if m.group(3) and m.group(4):
        frac = float(m.group(3)) / float(m.group(4))
    else:
        frac = 0
    return feet * 12 + inches + frac

df['Height'] = df['Height'].apply(height_to_inches)
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

def percent_to_float(val):
    if isinstance(val, str) and '%' in val:
        return float(val.replace('%', '').strip()) / 100
    try:
        return float(val)
    except:
        return np.nan

for col in ['Run Pass Ratio_Pass Rate', 'Run Pass Ratio_Rush Rate']:
    if col in df.columns:
        df[col] = df[col].apply(percent_to_float)

print("\nMerged DataFrame columns:", df.columns.tolist())

# Check for missing target columns
target_cols = ['Run Pass Ratio_Pass Rate', 'Run Pass Ratio_Rush Rate']
for col in target_cols:
    if col not in df.columns:
        print(f"WARNING: Target column '{col}' is missing from the merged data!")

if not all(col in df.columns for col in target_cols):
    print("\nAttempting to fix target columns...")
    # Try to find close matches for missing columns
    all_cols = df.columns.tolist()
    for target in target_cols:
        if target not in all_cols:
            possible_matches = [col for col in all_cols if target.lower() in col.lower()]
            print(f"Possible matches for '{target}': {possible_matches}")

# If still not successful, create dummy data for testing
if 'Run Pass Ratio_Pass Rate' not in df.columns or df['Run Pass Ratio_Pass Rate'].notna().sum() == 0:
    print("\nWARNING: No team data matched. Creating dummy data for testing purposes.")
    
    # Create random team tendencies for testing
    np.random.seed(42)  # For reproducibility
    df['Run Pass Ratio_Pass Rate'] = np.random.uniform(0.5, 0.7, len(df))
    df['Run Pass Ratio_Rush Rate'] = 1 - df['Run Pass Ratio_Pass Rate']
    
    print("Created dummy data with average pass rate:", df['Run Pass Ratio_Pass Rate'].mean())

# ADDED: Ensure we have consistent player names
# Use original_name for player_name if available
if 'original_name' in df.columns:
    df['player_name'] = df['original_name']
elif 'Name' in df.columns:
    df['player_name'] = df['Name']

# Print debug info for player names
print("\nSample player names from dataset:")
if 'player_name' in df.columns:
    print(df['player_name'].head(10).tolist())
else:
    print("player_name column not found!")

# Print team info
print("\nSample team info from dataset:")
if 'team_std' in df.columns:
    print(df['team_std'].head(10).tolist())
else:
    print("team_std column not found!")

# 7. Define features & targets (only pass/run fit)
# Update numeric features to remove target columns
numeric_feats = [
    'Height', 'Weight', 'draft_pick'
]
categorical_feats = ['Position', 'School']  # update as needed

# Check if target columns exist in the DataFrame
available_targets = [col for col in target_cols if col in df.columns]
if not available_targets:
    print("\nERROR: No target columns are available! Cannot proceed with modeling.")
    print("Available columns:", sorted(df.columns.tolist()))
else:
    print("\nNaNs per column before dropna:\n", df[numeric_feats + categorical_feats + target_cols].isnull().sum())
    
    # MODIFIED: Keep track of important columns during filtering
    keep_cols = numeric_feats + categorical_feats + target_cols + ['player_name', 'team_std']
    if 'Name' in df.columns:
        keep_cols.append('Name')
    if 'original_name' in df.columns:
        keep_cols.append('original_name')
    
    # Only drop rows with missing targets
    df_model = df[keep_cols].dropna(subset=target_cols)
    print("Rows in df_model after dropna:", len(df_model))
    
    # Print debug info about preserved columns
    print("\nColumns preserved in df_model:", df_model.columns.tolist())
    print("\nSample player names in df_model:")
    for col in ['player_name', 'Name', 'original_name']:
        if col in df_model.columns:
            print(f"{col}: {df_model[col].head(5).tolist()}")

    if len(df_model) > 0:
        X = df_model[numeric_feats + categorical_feats]
        y = df_model[target_cols]

        # 8. Build & train the pipeline
        # Use SimpleImputer for numeric features in the pipeline
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_feats),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_feats)
        ])

        model = Pipeline([
            ('pre', preprocessor),
            ('reg', MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)

        # 9. Evaluate each "fit" head
        y_pred = model.predict(X_test)
        for i, col in enumerate(target_cols):
            rmse = np.sqrt(mean_squared_error(y_test[col], y_pred[:, i]))
            r2   = r2_score(y_test[col], y_pred[:, i])
            print(f"{col:12s} → RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        # ADDED: Normalize prediction values
        predictions = model.predict(X)
        row_sums = predictions.sum(axis=1)
        normalized_predictions = predictions / row_sums[:, np.newaxis]
            
        # Save the trained model and results
        print("\nSaving model and results...")
        import pickle
        with open("C://Users//bharg//Downloads//nfl_draft_model.pkl", "wb") as f:
            pickle.dump(model, f)
            
        # MODIFIED: Save the predictions with improved player name handling
        result_df = pd.DataFrame({
            'player_name': df_model['player_name'] if 'player_name' in df_model.columns else 
                          (df_model['Name'] if 'Name' in df_model.columns else
                           (df_model['original_name'] if 'original_name' in df_model.columns else df_model.index)),
            'Position': df_model['Position'],
            'team': df_model['team_std'] if 'team_std' in df_model.columns else None,
            'actual_pass_rate': df_model['Run Pass Ratio_Pass Rate'],
            'actual_rush_rate': df_model['Run Pass Ratio_Rush Rate'],
            'predicted_pass_rate': normalized_predictions[:, 0],
            'predicted_rush_rate': normalized_predictions[:, 1]
        })
        
        # Print sample of result_df to verify data integrity before saving
        print("\nSample of result_df before saving:")
        print(result_df.head())
        
        result_df.to_csv("C://Users//bharg//Downloads//nfl_draft_model_results.csv", index=False)
        print("Results saved to: C://Users//bharg//Downloads//nfl_draft_model_results.csv")
    else:
        print("\nERROR: No rows remaining after filtering for non-null target values!")
        print("Cannot train model without data. Please check data sources and merge operations.")
        
        # As a last resort, analyze team names from both sources
        print("\nTEAM NAME ANALYSIS")
        print("=" * 40)
        print("Original Team Names in Draft Data:")
        for team in sorted(merged['Drafted_Team'].unique()):
            print(f"- {team}")
            
        print("\nOriginal Team Names in Tendencies Data:")
        for team in sorted(team_tendencies['Drafted_Team'].unique()):
            print(f"- {team}")
            
        print("\nPlease check the team tendencies data source and ensure it contains the expected data format.")
