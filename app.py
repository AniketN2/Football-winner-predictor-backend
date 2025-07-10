from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import threading
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Sample team data for search functionality
TEAMS_DATA = [
  #England – Premier League 2024‑25 (20 teams)
  {"id": 1, "name": "Arsenal", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Arsenal-Logo.png"},
  {"id": 2, "name": "Aston Villa", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Aston-Villa-Logo.png"},
  {"id": 3, "name": "Brighton & Hove Albion", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Brighton-Logo.png"},
  {"id": 4, "name": "Chelsea", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Chelsea-Logo.png"},
  {"id": 5, "name": "Crystal Palace", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Crystal-Palace-Logo.png"},
  {"id": 6, "name": "Everton", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Everton-Logo.png"},
  {"id": 7, "name": "Fulham", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Fulham-Logo.png"},
  {"id": 8, "name": "Liverpool", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Liverpool-Logo.png"},
  {"id": 9, "name": "Luton Town", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Luton-Logo.png"},
  {"id":10, "name": "Manchester City", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Manchester-City-Logo.png"},
  {"id":11, "name": "Manchester United", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Manchester-United-Logo.png"},
  {"id":12, "name": "Newcastle United", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Newcastle-Logo.png"},
  {"id":13, "name": "Nottingham Forest", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Nottingham-Forest-Logo.png"},
  {"id":14, "name": "Tottenham Hotspur", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Tottenham-Logo.png"},
  {"id":15, "name": "West Ham United", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/West-Ham-Logo.png"},
  {"id":16, "name": "Brentford", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Brentford-Logo.png"},
  {"id":17, "name": "Wolverhampton Wanderers", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Wolves-Logo.png"},
  {"id":18, "name": "Bournemouth", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Bournemouth-Logo.png"},
  {"id":19, "name": "Sheffield United", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Sheffield-United-Logo.png"},
  {"id":20, "name": "Everton", "country": "England", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Everton-Logo.png"},

  #Spain – La Liga 2025‑26 (20 teams)
  {"id":21, "name": "Alavés", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Alaves-Logo.png"},
  {"id":22, "name": "Athletic Bilbao", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Athletic-Bilbao-Logo.png"},
  {"id":23, "name": "Atlético Madrid", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Atletico-Madrid-Logo.png"},
  {"id":24, "name": "Barcelona", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Barcelona-Logo.png"},
  {"id":25, "name": "Celta Vigo", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Celta-Vigo-Logo.png"},
  {"id":26, "name": "Elche", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Elche-Logo.png"},
  {"id":27, "name": "Espanyol", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Espanyol-Logo.png"},
  {"id":28, "name": "Getafe", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Getafe-Logo.png"},
  {"id":29, "name": "Girona", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Girona-Logo.png"},
  {"id":30, "name": "Levante", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Levante-Logo.png"},
  {"id":31, "name": "Mallorca", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Mallorca-Logo.png"},
  {"id":32, "name": "Osasuna", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Osasuna-Logo.png"},
  {"id":33, "name": "Rayo Vallecano", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Rayo-Vallecano-Logo.png"},
  {"id":34, "name": "Real Betis", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Real-Betis-Logo.png"},
  {"id":35, "name": "Real Madrid", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Real-Madrid-Logo.png"},
  {"id":36, "name": "Real Oviedo", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Real-Oviedo-Logo.png"},
  {"id":37, "name": "Real Sociedad", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Real-Sociedad-Logo.png"},
  {"id":38, "name": "Sevilla", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Sevilla-Logo.png"},
  {"id":39, "name": "Valencia", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Valencia-Logo.png"},
  {"id":40, "name": "Villarreal", "country": "Spain", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Villarreal-Logo.png"},

  #Germany – Bundesliga (18 teams)
  {"id":41, "name": "Bayern Munich", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Bayern-Munich-Logo.png"},
  {"id":42, "name": "Borussia Dortmund", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Borussia-Dortmund-Logo.png"},
  {"id":43, "name": "Bayer Leverkusen", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Bayer-Leverkusen-Logo.png"},
  {"id":44, "name": "RB Leipzig", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/RB-Leipzig-Logo.png"},
  {"id":45, "name": "Eintracht Frankfurt", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Eintracht-Frankfurt-Logo.png"},
  {"id":46, "name": "VfL Bochum", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Bochum-Logo.png"},
  {"id":47, "name": "VfB Stuttgart", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Stuttgart-Logo.png"},
  {"id":48, "name": "1. FC Heidenheim", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Heidenheim-Logo.png"},
  {"id":49, "name": "Borussia Mönchengladbach", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Gladbach-Logo.png"},
  {"id":50, "name": "Eintracht Braunschweig", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Braunschweig-Logo.png"},
  {"id":51, "name": "SC Freiburg", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Freiburg-Logo.png"},
  {"id":52, "name": "FC Augsburg", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Augsburg-Logo.png"},
  {"id":53, "name": "TSG Hoffenheim", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Hoffenheim-Logo.png"},
  {"id":54, "name": "Werder Bremen", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Bremen-Logo.png"},
  {"id":55, "name": "1. FC Köln", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Cologne-Logo.png"},
  {"id":56, "name": "Hamburger SV", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Hamburg-Logo.png"},
  {"id":57, "name": "Holstein Kiel", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Kiel-Logo.png"},
  {"id":58, "name": "FC Heidenheim", "country": "Germany", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Heidenheim-Logo.png"},

  #France – Ligue 1 (20 teams)
  {"id":61, "name": "Paris Saint-Germain", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/PSG-Logo.png"},
  {"id":62, "name": "Lyon", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Lyon-Logo.png"},
  {"id":63, "name": "Marseille", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Marseille-Logo.png"},
  {"id":64, "name": "Monaco", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Monaco-Logo.png"},
  {"id":65, "name": "Lille", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Lille-Logo.png"},
  {"id":66, "name": "Rennes", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Rennes-Logo.png"},
  {"id":67, "name": "Nice", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Nice-Logo.png"},
  {"id":68, "name": "Lens", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Lens-Logo.png"},
  {"id":69, "name": "Bordeaux", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Bordeaux-Logo.png"},
  {"id":70, "name": "Saint-Étienne", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Saint-Etienne-Logo.png"},
  {"id":71, "name": "Nantes", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Nantes-Logo.png"},
  {"id":72, "name": "Reims", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Reims-Logo.png"},
  {"id":73, "name": "Montpellier", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Montpellier-Logo.png"},
  {"id":74, "name": "Strasbourg", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Strasbourg-Logo.png"},
  {"id":75, "name": "Toulouse", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Toulouse-Logo.png"},
  {"id":76, "name": "Lorient", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Lorient-Logo.png"},
  {"id":77, "name": "Angers", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Angers-Logo.png"},
  {"id":78, "name": "Brest", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Brest-Logo.png"},
  {"id":79, "name": "Clermont Foot", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Clermont-Logo.png"},
  {"id":80, "name": "AJ Auxerre", "country": "France", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Auxerre-Logo.png"},

  #Netherlands – Eredivisie (18 teams)
  {"id":91, "name": "Ajax", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Ajax-Logo.png"},
  {"id":92, "name": "PSV Eindhoven", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/PSV-Logo.png"},
  {"id":93, "name": "Feyenoord", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Feyenoord-Logo.png"},
  {"id":94, "name": "AZ Alkmaar", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/AZ-Logo.png"},
  {"id":95, "name": "FC Utrecht", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Utrecht-Logo.png"},
  {"id":96, "name": "SC Heerenveen", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Heerenveen-Logo.png"},
  {"id":97, "name": "Vitesse", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Vitesse-Logo.png"},
  {"id":98, "name": "Sparta Rotterdam", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Sparta-Rotterdam-Logo.png"},
  {"id":99, "name": "SC Cambuur", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Cambuur-Logo.png"},
  {"id":100,"name": "FC Groningen", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Groningen-Logo.png"},
  {"id":101,"name": "PEC Zwolle", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/PEC-Zwolle-Logo.png"},
  {"id":102,"name": "Go Ahead Eagles", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Go-Ahead-Eagles-Logo.png"},
  {"id":103,"name": "RKC Waalwijk", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/RKC-Logo.png"},
  {"id":104,"name": "Fortuna Sittard", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Fortuna-Logo.png"},
  {"id":105,"name": "Excelsior", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Excelsior-Logo.png"},
  {"id":106,"name": "NEC Nijmegen", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/NEC-Logo.png"},
  {"id":107,"name": "FC Twente", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Twente-Logo.png"},
  {"id":108,"name": "Willem II", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Willem-II-Logo.png"},
  {"id":109,"name": "Rijnsburgse Boys", "country": "Netherlands", "logo": "https://logos-world.net/wp-content/uploads/2020/06/Rijnsburg-Logo.png"}
]


class AdvancedFootballPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_training = False
        self.model_ready = False
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            with open('balanced_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['features']
                self.model_ready = True
            print("✓ Model loaded successfully from file")
        except FileNotFoundError:
            print("⚠ Model file not found. Training new model in background...")
            self.train_model_async()
        except Exception as e:
            print(f"⚠ Error loading model: {e}. Training new model in background...")
            self.train_model_async()
    
    def train_model_async(self):
        """Train model in background thread"""
        def train_thread():
            self.is_training = True
            self.train_model()
            self.is_training = False
            self.model_ready = True
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def train_model(self):
        """Train the advanced prediction model with balanced data"""
        print("🔄 Starting model training...")
        np.random.seed(42)
        n_samples = 5000
        
        # Generate realistic team features
        # Team 1 (Home) features
        team1_goals_for = np.random.normal(1.5, 0.6, n_samples)
        team1_goals_against = np.random.normal(1.3, 0.5, n_samples)
        team1_win_rate = np.random.beta(2, 3, n_samples)
        team1_form = np.random.uniform(0, 15, n_samples)
        team1_rating = np.random.normal(7.0, 1.0, n_samples)
        team1_injuries = np.random.poisson(2, n_samples)
        
        # Team 2 (Away) features  
        team2_goals_for = np.random.normal(1.4, 0.6, n_samples)
        team2_goals_against = np.random.normal(1.4, 0.5, n_samples)
        team2_win_rate = np.random.beta(2, 3, n_samples)
        team2_form = np.random.uniform(0, 15, n_samples)
        team2_rating = np.random.normal(6.8, 1.0, n_samples)
        team2_injuries = np.random.poisson(2, n_samples)
        
        # Head to head history
        h2h = np.random.uniform(-2, 2, n_samples)
        
        # Ensure positive values
        team1_goals_for = np.abs(team1_goals_for)
        team1_goals_against = np.abs(team1_goals_against)
        team2_goals_for = np.abs(team2_goals_for)
        team2_goals_against = np.abs(team2_goals_against)
        team1_rating = np.clip(team1_rating, 5.0, 9.0)
        team2_rating = np.clip(team2_rating, 5.0, 9.0)
        
        # Create features array
        X = np.column_stack([
            team1_goals_for, team1_goals_against, team1_win_rate, team1_form, team1_rating, team1_injuries,
            team2_goals_for, team2_goals_against, team2_win_rate, team2_form, team2_rating, team2_injuries,
            team1_goals_for - team2_goals_for,  # Goal difference
            team1_win_rate - team2_win_rate,    # Win rate difference
            team1_rating - team2_rating,        # Rating difference
            team1_form - team2_form,            # Form difference
            h2h,                                # Head to head
            np.ones(n_samples) * 0.15           # Home advantage
        ])
        
        # Create balanced labels
        team1_strength = (
            team1_goals_for * 0.3 + 
            (3 - team1_goals_against) * 0.2 + 
            team1_win_rate * 0.2 + 
            team1_form/15 * 0.15 + 
            team1_rating/10 * 0.15
        )
        
        team2_strength = (
            team2_goals_for * 0.3 + 
            (3 - team2_goals_against) * 0.2 + 
            team2_win_rate * 0.2 + 
            team2_form/15 * 0.15 + 
            team2_rating/10 * 0.15
        )
        
        # Add home advantage and randomness
        home_advantage = 0.15
        random_factor = np.random.normal(0, 0.1, n_samples)
        
        strength_diff = team1_strength - team2_strength + home_advantage + h2h/10 + random_factor
        
        # Create balanced outcomes using percentiles
        y = np.zeros(n_samples, dtype=int)
        draw_threshold_low = np.percentile(strength_diff, 35)
        draw_threshold_high = np.percentile(strength_diff, 65)
        
        y[strength_diff < draw_threshold_low] = 0    # Away win
        y[strength_diff > draw_threshold_high] = 2   # Home win  
        y[(strength_diff >= draw_threshold_low) & (strength_diff <= draw_threshold_high)] = 1  # Draw
        
        # Train model with balanced classes
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_scaled, y)
        
        # Verify distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Label distribution: Away Win: {counts[0]}, Draw: {counts[1]}, Home Win: {counts[2]}")
        
        self.feature_columns = [
            'team1_goals_for', 'team1_goals_against', 'team1_win_rate', 'team1_form', 'team1_rating', 'team1_injuries',
            'team2_goals_for', 'team2_goals_against', 'team2_win_rate', 'team2_form', 'team2_rating', 'team2_injuries',
            'goal_diff', 'win_rate_diff', 'rating_diff', 'form_diff', 'h2h', 'home_advantage'
        ]
        
        # Save model
        try:
            with open('balanced_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'features': self.feature_columns
                }, f)
            print("✓ Model trained and saved successfully")
        except Exception as e:
            print(f"⚠ Error saving model: {e}")
    
    def extract_features(self, team1_stats, team2_stats):
        """Extract features for prediction - now uses both teams properly"""
        
        # Team 1 features
        t1_goals_for = float(team1_stats.get('goals_for_avg', 1.5))
        t1_goals_against = float(team1_stats.get('goals_against_avg', 1.3))
        t1_win_rate = float(team1_stats.get('win_rate', 0.5))
        t1_form = float(team1_stats.get('form_points', 7.5))
        t1_rating = float(team1_stats.get('player_ratings', 7.0))
        t1_injuries = float(team1_stats.get('injuries', 2))
        
        # Team 2 features
        t2_goals_for = float(team2_stats.get('goals_for_avg', 1.4))
        t2_goals_against = float(team2_stats.get('goals_against_avg', 1.4))
        t2_win_rate = float(team2_stats.get('win_rate', 0.5))
        t2_form = float(team2_stats.get('form_points', 7.5))
        t2_rating = float(team2_stats.get('player_ratings', 6.8))
        t2_injuries = float(team2_stats.get('injuries', 2))
        
        # Head to head and home advantage
        h2h = float(team1_stats.get('head_to_head', 0))
        
        features = [
            t1_goals_for, t1_goals_against, t1_win_rate, t1_form, t1_rating, t1_injuries,
            t2_goals_for, t2_goals_against, t2_win_rate, t2_form, t2_rating, t2_injuries,
            t1_goals_for - t2_goals_for,  # Goal difference
            t1_win_rate - t2_win_rate,    # Win rate difference  
            t1_rating - t2_rating,        # Rating difference
            t1_form - t2_form,            # Form difference
            h2h,                          # Head to head
            0.15                          # Home advantage
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_match(self, team1_stats, team2_stats):
        """Make prediction with proper team comparison"""
        if not self.model_ready:
            if self.is_training:
                raise Exception("Model is currently training. Please wait a moment and try again.")
            else:
                raise Exception("Model not initialized")
            
        features = self.extract_features(team1_stats, team2_stats)
        features_scaled = self.scaler.transform(features)
        
        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence
        confidence = max(probabilities) * 100
        
        return {
            'prediction': int(prediction),
            'probabilities': {
                'loss': float(probabilities[0]),      # Away win
                'draw': float(probabilities[1]),      # Draw
                'win': float(probabilities[2])        # Home win
            },
            'confidence': float(confidence)
        }

# Initialize the predictor
predictor = AdvancedFootballPredictor()

def get_team_stats(team_name):
    """Generate realistic team statistics"""
    # Find team in database
    team_data = next((team for team in TEAMS_DATA if team['name'].lower() == team_name.lower()), None)
    
    # Generate realistic stats with some variation
    base_stats = {
        'goals_for_avg': random.uniform(1.2, 2.5),
        'goals_against_avg': random.uniform(0.8, 1.8),
        'win_rate': random.uniform(0.3, 0.8),
        'form_points': random.randint(5, 15),
        'player_ratings': random.uniform(6.5, 8.5),
        'injuries': random.randint(0, 5),
        'head_to_head': random.uniform(-1, 1)
    }
    
    return {
        'name': team_name,
        'logo': team_data['logo'] if team_data else 'https://via.placeholder.com/50',
        'stats': base_stats,
        'recent_form': [random.choice(['W', 'D', 'L']) for _ in range(5)]
    }

@app.route('/api/teams/search', methods=['GET'])
def search_teams():
    """Search for teams by name"""
    query = request.args.get('q', '').lower()
    
    if len(query) < 2:
        return jsonify({'teams': []})
    
    # Filter teams based on search query
    matching_teams = [
        team for team in TEAMS_DATA 
        if query in team['name'].lower()
    ]
    
    return jsonify({'teams': matching_teams[:10]})  # Limit to 10 results

@app.route('/api/predict', methods=['POST'])
def predict_match():
    try:
        data = request.get_json()
        team1_name = data.get('team1')
        team2_name = data.get('team2')
        
        if not team1_name or not team2_name:
            return jsonify({'error': 'Both team names are required'}), 400
        
        # Get team statistics
        team1_data = get_team_stats(team1_name)
        team2_data = get_team_stats(team2_name)
        
        # Make prediction
        prediction_result = predictor.predict_match(team1_data['stats'], team2_data['stats'])
        
        # Format response to match frontend expectations
        response = {
            'teams': {
                'home': team1_data,
                'away': team2_data
            },
            'prediction': {
                'probabilities': {
                    'home_win': round(prediction_result['probabilities']['win'] * 100, 1),
                    'draw': round(prediction_result['probabilities']['draw'] * 100, 1),
                    'away_win': round(prediction_result['probabilities']['loss'] * 100, 1)
                },
                'confidence': round(prediction_result['confidence'], 1)
            },
            'analysis': {
                'key_factors': [
                    f"{team1_name} has {team1_data['stats']['goals_for_avg']:.1f} goals per game average",
                    f"{team2_name} concedes {team2_data['stats']['goals_against_avg']:.1f} goals per game",
                    f"Home advantage factor included in analysis",
                    f"Recent form: {team1_name} vs {team2_name} comparison"
                ],
                'recommendation': f"Based on statistical analysis, the model suggests {team1_name if prediction_result['prediction'] == 2 else team2_name if prediction_result['prediction'] == 0 else 'a draw'} as the most likely outcome.",
                'risk_assessment': 'Low' if prediction_result['confidence'] > 70 else 'Medium' if prediction_result['confidence'] > 50 else 'High'
            },
            'timestamp': '2024-01-01T12:00:00Z'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_ready': predictor.model_ready,
        'is_training': predictor.is_training
    })

if __name__ == '__main__':
    print("🚀 Starting Football Predictor API...")
    print("🔍 Team search available at: http://localhost:5000/api/teams/search")
    print("🎯 Prediction endpoint at: http://localhost:5000/api/predict")
    app.run(debug=True, host='0.0.0.0', port=5000)