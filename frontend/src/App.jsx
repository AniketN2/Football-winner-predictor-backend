import React, { useState, useEffect } from 'react';
import './App.css';
import { Search, TrendingUp, TrendingDown, Minus, Trophy, Target, AlertTriangle, CheckCircle, Clock, Star, Activity, Users, BarChart3, Zap } from 'lucide-react';

const FootballPredictor = () => {
  const [team1, setTeam1] = useState('');
  const [team2, setTeam2] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchResults1, setSearchResults1] = useState([]);
  const [searchResults2, setSearchResults2] = useState([]);
  const [showSearch1, setShowSearch1] = useState(false);
  const [showSearch2, setShowSearch2] = useState(false);
  const [selectedTeams, setSelectedTeams] = useState({ team1: null, team2: null });

  const searchTeams = async (query, setResults, setShow) => {
    if (query.length < 2) {
      setResults([]);
      setShow(false);
      return;
    }

    try {
      const response = await fetch(`http://localhost:5000/api/teams/search?q=${encodeURIComponent(query)}`);
      const data = await response.json();
      setResults(data.teams || []);
      setShow(true);
    } catch (err) {
      console.error('Search error:', err);
      setResults([]);
    }
  };

  const handleTeamSelect = (team, teamNumber) => {
    if (teamNumber === 1) {
      setTeam1(team.name);
      setSelectedTeams(prev => ({ ...prev, team1: team }));
      setShowSearch1(false);
    } else {
      setTeam2(team.name);
      setSelectedTeams(prev => ({ ...prev, team2: team }));
      setShowSearch2(false);
    }
  };

  const predictMatch = async () => {
    if (!team1.trim() || !team2.trim()) {
      setError('Please enter both team names');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ team1: team1.trim(), team2: team2.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getPredictionIcon = (result) => {
    switch (result) {
      case 'Win': return <TrendingUp className="w-6 h-6 text-green-500" />;
      case 'Draw': return <Minus className="w-6 h-6 text-yellow-500" />;
      case 'Loss': return <TrendingDown className="w-6 h-6 text-red-500" />;
      default: return <Target className="w-6 h-6 text-gray-500" />;
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'Low': return 'text-green-500 bg-green-50';
      case 'Medium': return 'text-yellow-600 bg-yellow-50';
      case 'High': return 'text-red-500 bg-red-50';
      default: return 'text-gray-500 bg-gray-50';
    }
  };

  const getFormIcon = (result) => {
    switch (result) {
      case 'W': return <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-xs font-bold">W</div>;
      case 'D': return <div className="w-6 h-6 bg-yellow-500 rounded-full flex items-center justify-center text-white text-xs font-bold">D</div>;
      case 'L': return <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center text-white text-xs font-bold">L</div>;
      default: return <div className="w-6 h-6 bg-gray-400 rounded-full flex items-center justify-center text-white text-xs font-bold">-</div>;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-green-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-pulse delay-2000"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12 pt-8">
          <div className="flex items-center justify-center mb-6">
            <div className="p-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full shadow-lg">
              <Trophy className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-6xl font-bold bg-gradient-to-r from-white via-purple-200 to-white bg-clip-text text-transparent mb-4">
            AI Football Predictor
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Advanced machine learning algorithms analyze team performance, player statistics, and historical data to predict match outcomes with unprecedented accuracy.
          </p>
        </div>

        {/* Main Prediction Interface */}
        <div className="bg-white/10 backdrop-blur-lg rounded-3xl shadow-2xl border border-white/20 p-8 mb-8">
          <div className="grid md:grid-cols-2 gap-8 mb-8">
            {/* Team 1 Input */}
            <div className="relative">
              <label className="block text-white font-semibold mb-3 text-lg">Home Team</label>
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  value={team1}
                  onChange={(e) => {
                    setTeam1(e.target.value);
                    searchTeams(e.target.value, setSearchResults1, setShowSearch1);
                  }}
                  placeholder="Search for home team..."
                  className="w-full pl-12 pr-4 py-4 bg-white/20 border border-white/30 rounded-2xl text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300"
                />
                {selectedTeams.team1 && (
                  <img src={selectedTeams.team1.logo} alt="" className="absolute right-4 top-1/2 transform -translate-y-1/2 w-8 h-8 rounded-full" />
                )}
              </div>
              {showSearch1 && searchResults1.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-xl border border-gray-200 max-h-60 overflow-y-auto z-50">
                  {searchResults1.map((team) => (
                    <div
                      key={team.id}
                      onClick={() => handleTeamSelect(team, 1)}
                      className="flex items-center p-3 hover:bg-gray-50 cursor-pointer border-b last:border-b-0"
                    >
                      <img src={team.logo} alt="" className="w-8 h-8 rounded-full mr-3" />
                      <div>
                        <div className="font-semibold text-gray-800">{team.name}</div>
                        <div className="text-sm text-gray-500">{team.country}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Team 2 Input */}
            <div className="relative">
              <label className="block text-white font-semibold mb-3 text-lg">Away Team</label>
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  value={team2}
                  onChange={(e) => {
                    setTeam2(e.target.value);
                    searchTeams(e.target.value, setSearchResults2, setShowSearch2);
                  }}
                  placeholder="Search for away team..."
                  className="w-full pl-12 pr-4 py-4 bg-white/20 border border-white/30 rounded-2xl text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300"
                />
                {selectedTeams.team2 && (
                  <img src={selectedTeams.team2.logo} alt="" className="absolute right-4 top-1/2 transform -translate-y-1/2 w-8 h-8 rounded-full" />
                )}
              </div>
              {showSearch2 && searchResults2.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-xl border border-gray-200 max-h-60 overflow-y-auto z-50">
                  {searchResults2.map((team) => (
                    <div
                      key={team.id}
                      onClick={() => handleTeamSelect(team, 2)}
                      className="flex items-center p-3 hover:bg-gray-50 cursor-pointer border-b last:border-b-0"
                    >
                      <img src={team.logo} alt="" className="w-8 h-8 rounded-full mr-3" />
                      <div>
                        <div className="font-semibold text-gray-800">{team.name}</div>
                        <div className="text-sm text-gray-500">{team.country}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Predict Button */}
          <div className="text-center">
            <button
              onClick={predictMatch}
              disabled={loading}
              className="px-12 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-bold text-xl rounded-2xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              {loading ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                  Analyzing Match...
                </div>
              ) : (
                <div className="flex items-center">
                  <Zap className="w-6 h-6 mr-3" />
                  Predict Match Outcome
                </div>
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-2xl p-6 mb-8 backdrop-blur-sm">
            <div className="flex items-center">
              <AlertTriangle className="w-6 h-6 text-red-400 mr-3" />
              <span className="text-red-200 font-semibold">{error}</span>
            </div>
          </div>
        )}

        {/* Prediction Results */}
        {prediction && (
          <div className="space-y-8">
            {/* Main Prediction Card */}
            <div className="bg-white/10 backdrop-blur-lg rounded-3xl shadow-2xl border border-white/20 p-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-white mb-4">Match Prediction</h2>
                <div className="flex items-center justify-center space-x-8">
                  <div className="text-center">
                    <img src={prediction.teams.home.logo} alt="" className="w-16 h-16 mx-auto mb-2 rounded-full" />
                    <h3 className="text-xl font-bold text-white">{prediction.teams.home.name}</h3>
                  </div>
                  <div className="text-center">
                    <div className="text-6xl font-bold text-white mb-2">VS</div>
                  </div>
                  <div className="text-center">
                    <img src={prediction.teams.away.logo} alt="" className="w-16 h-16 mx-auto mb-2 rounded-full" />
                    <h3 className="text-xl font-bold text-white">{prediction.teams.away.name}</h3>
                  </div>
                </div>
              </div>

              {/* Prediction Results */}
              <div className="grid md:grid-cols-3 gap-6 mb-8">
                <div className="bg-green-500/20 border border-green-500/50 rounded-2xl p-6 text-center">
                  <TrendingUp className="w-12 h-12 text-green-400 mx-auto mb-3" />
                  <h4 className="text-lg font-bold text-white mb-2">Home Win</h4>
                  <div className="text-3xl font-bold text-green-400">{prediction.prediction.probabilities.home_win}%</div>
                </div>
                <div className="bg-yellow-500/20 border border-yellow-500/50 rounded-2xl p-6 text-center">
                  <Minus className="w-12 h-12 text-yellow-400 mx-auto mb-3" />
                  <h4 className="text-lg font-bold text-white mb-2">Draw</h4>
                  <div className="text-3xl font-bold text-yellow-400">{prediction.prediction.probabilities.draw}%</div>
                </div>
                <div className="bg-red-500/20 border border-red-500/50 rounded-2xl p-6 text-center">
                  <TrendingDown className="w-12 h-12 text-red-400 mx-auto mb-3" />
                  <h4 className="text-lg font-bold text-white mb-2">Away Win</h4>
                  <div className="text-3xl font-bold text-red-400">{prediction.prediction.probabilities.away_win}%</div>
                </div>
              </div>

              {/* Confidence and Risk */}
              <div className="grid md:grid-cols-2 gap-6 mb-8">
                <div className="bg-white/10 rounded-2xl p-6">
                  <div className="flex items-center mb-4">
                    <CheckCircle className="w-6 h-6 text-blue-400 mr-3" />
                    <h4 className="text-lg font-bold text-white">Confidence Level</h4>
                  </div>
                  <div className="text-3xl font-bold text-blue-400 mb-2">{prediction.prediction.confidence}%</div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-1000"
                      style={{ width: `${prediction.prediction.confidence}%` }}
                    ></div>
                  </div>
                </div>
                <div className="bg-white/10 rounded-2xl p-6">
                  <div className="flex items-center mb-4">
                    <AlertTriangle className="w-6 h-6 text-orange-400 mr-3" />
                    <h4 className="text-lg font-bold text-white">Risk Assessment</h4>
                  </div>
                  <div className={`inline-block px-4 py-2 rounded-full font-bold ${getRiskColor(prediction.analysis.risk_assessment)}`}>
                    {prediction.analysis.risk_assessment} Risk
                  </div>
                </div>
              </div>
            </div>

            {/* Team Statistics */}
            <div className="grid md:grid-cols-2 gap-8">
              {/* Home Team Stats */}
              <div className="bg-white/10 backdrop-blur-lg rounded-3xl shadow-2xl border border-white/20 p-6">
                <div className="flex items-center mb-6">
                  <img src={prediction.teams.home.logo} alt="" className="w-12 h-12 rounded-full mr-4" />
                  <h3 className="text-2xl font-bold text-white">{prediction.teams.home.name}</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Goals For (Avg)</span>
                    <span className="text-white font-bold">{prediction.teams.home.stats.goals_for_avg.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Goals Against (Avg)</span>
                    <span className="text-white font-bold">{prediction.teams.home.stats.goals_against_avg.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Win Rate</span>
                    <span className="text-white font-bold">{(prediction.teams.home.stats.win_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Form Points</span>
                    <span className="text-white font-bold">{prediction.teams.home.stats.form_points}/15</span>
                  </div>
                </div>

                <div className="mt-6">
                  <h4 className="text-white font-bold mb-3">Recent Form</h4>
                  <div className="flex space-x-2">
                    {prediction.teams.home.recent_form.map((result, index) => (
                      <div key={index}>{getFormIcon(result)}</div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Away Team Stats */}
              <div className="bg-white/10 backdrop-blur-lg rounded-3xl shadow-2xl border border-white/20 p-6">
                <div className="flex items-center mb-6">
                  <img src={prediction.teams.away.logo} alt="" className="w-12 h-12 rounded-full mr-4" />
                  <h3 className="text-2xl font-bold text-white">{prediction.teams.away.name}</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Goals For (Avg)</span>
                    <span className="text-white font-bold">{prediction.teams.away.stats.goals_for_avg.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Goals Against (Avg)</span>
                    <span className="text-white font-bold">{prediction.teams.away.stats.goals_against_avg.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Win Rate</span>
                    <span className="text-white font-bold">{(prediction.teams.away.stats.win_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Form Points</span>
                    <span className="text-white font-bold">{prediction.teams.away.stats.form_points}/15</span>
                  </div>
                </div>

                <div className="mt-6">
                  <h4 className="text-white font-bold mb-3">Recent Form</h4>
                  <div className="flex space-x-2">
                    {prediction.teams.away.recent_form.map((result, index) => (
                      <div key={index}>{getFormIcon(result)}</div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Section */}
            <div className="bg-white/10 backdrop-blur-lg rounded-3xl shadow-2xl border border-white/20 p-8">
              <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                <BarChart3 className="w-6 h-6 mr-3" />
                Match Analysis
              </h3>
              
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4">Key Factors</h4>
                  <div className="space-y-3">
                    {prediction.analysis.key_factors.map((factor, index) => (
                      <div key={index} className="flex items-start">
                        <Star className="w-5 h-5 text-yellow-400 mr-3 mt-0.5 flex-shrink-0" />
                        <span className="text-gray-300">{factor}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4">AI Recommendation</h4>
                  <div className="bg-purple-500/20 border border-purple-500/50 rounded-2xl p-6">
                    <div className="flex items-start">
                      <Activity className="w-6 h-6 text-purple-400 mr-3 mt-1 flex-shrink-0" />
                      <span className="text-purple-200 font-semibold">{prediction.analysis.recommendation}</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-6 pt-6 border-t border-white/20">
                <div className="flex items-center text-gray-400 text-sm">
                  <Clock className="w-4 h-4 mr-2" />
                  Analysis generated on {new Date(prediction.timestamp).toLocaleString()}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FootballPredictor;