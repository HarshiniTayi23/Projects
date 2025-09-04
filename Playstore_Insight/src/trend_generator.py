import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .database_setup import DatabaseManager
import os

class AdvancedTrendGenerator:
    """
    Advanced trend analysis engine with statistical insights and forecasting
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or DatabaseManager()
        
    def generate_comprehensive_trends(self, analysis_date, window_days=30):
        """Generate comprehensive trend analysis with correct date range T to T-30"""
        print(f"Generating comprehensive trends for {analysis_date} (T to T-{window_days})")
        
        # CORRECTED: Calculate date range T to T-30 (past 30 days)
        end_date = analysis_date  # T
        start_date = end_date - timedelta(days=window_days-1)  # T-30
        
        print(f"   Date range: {start_date} to {end_date}")
        
        # Generate daily trends for each date in range
        trend_data = self._generate_daily_trends_for_range(start_date, end_date)
        
        # Create pivot table (topics as rows, dates as columns)
        trend_matrix = self._create_trend_matrix(trend_data)
        
        # Calculate advanced metrics
        trend_insights = self._calculate_trend_insights(trend_matrix)
        
        # Generate statistical analysis
        statistical_summary = self._generate_statistical_summary(trend_matrix)
        
        return {
            'trend_matrix': trend_matrix,
            'insights': trend_insights,
            'statistics': statistical_summary,
            'date_range': {'start': start_date, 'end': end_date}
        }
    
    def _generate_daily_trends_for_range(self, start_date, end_date):
        """Generate daily trend data for specified date range"""
        conn = self.db_manager.get_connection()
        
        # Query to get topic frequencies for each date in range
        query = '''
            SELECT 
                t.topic_name,
                r.review_date,
                COUNT(*) as frequency,
                AVG(r.rating) as avg_sentiment,
                AVG(rt.confidence_score) as avg_confidence
            FROM reviews r
            JOIN review_topics rt ON r.id = rt.review_id
            JOIN topics t ON rt.topic_id = t.id
            WHERE r.review_date BETWEEN ? AND ?
            GROUP BY t.topic_name, r.review_date
            ORDER BY r.review_date, t.topic_name
        '''
        
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()
        
        return df
    
    def _create_trend_matrix(self, trend_data):
        """Create pivot table with topics as rows and dates as columns"""
        if trend_data.empty:
            return pd.DataFrame()
        
        # Pivot to get desired format
        pivot_df = trend_data.pivot(
            index='topic_name', 
            columns='review_date', 
            values='frequency'
        )
        
        # Fill missing values with 0
        pivot_df = pivot_df.fillna(0).astype(int)
        
        # Sort columns (dates) chronologically
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
        
        # Sort rows by total frequency (most active topics first)
        pivot_df['total'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('total', ascending=False)
        pivot_df = pivot_df.drop('total', axis=1)
        
        return pivot_df
    
    def _calculate_trend_insights(self, trend_matrix):
        """Calculate advanced trend insights and patterns"""
        if trend_matrix.empty:
            return {}
        
        insights = {}
        
        # 1. Trending topics (increasing frequency)
        trending_topics = self._identify_trending_topics(trend_matrix)
        insights['trending_topics'] = trending_topics
        
        # 2. Declining topics (decreasing frequency)
        declining_topics = self._identify_declining_topics(trend_matrix)
        insights['declining_topics'] = declining_topics
        
        # 3. Emerging topics (new in recent period)
        emerging_topics = self._identify_emerging_topics(trend_matrix)
        insights['emerging_topics'] = emerging_topics
        
        # 4. Stable topics (consistent frequency)
        stable_topics = self._identify_stable_topics(trend_matrix)
        insights['stable_topics'] = stable_topics
        
        # 5. Volatile topics (high frequency variation)
        volatile_topics = self._identify_volatile_topics(trend_matrix)
        insights['volatile_topics'] = volatile_topics
        
        # 6. Peak detection
        peak_analysis = self._analyze_peaks(trend_matrix)
        insights['peak_analysis'] = peak_analysis
        
        return insights
    
    def _identify_trending_topics(self, trend_matrix):
        """Identify topics with upward trend"""
        trending = []
        
        for topic in trend_matrix.index:
            topic_data = trend_matrix.loc[topic].values
            
            if len(topic_data) >= 7:  # Need at least a week of data
                # Compare recent period vs earlier period
                recent_avg = np.mean(topic_data[-7:])  # Last 7 days
                earlier_avg = np.mean(topic_data[-14:-7])  # Previous 7 days
                
                if earlier_avg > 0 and recent_avg > earlier_avg * 1.3:  # 30% increase
                    trend_strength = (recent_avg - earlier_avg) / earlier_avg
                    trending.append({
                        'topic': topic,
                        'trend_strength': trend_strength,
                        'recent_avg': recent_avg,
                        'earlier_avg': earlier_avg
                    })
        
        # Sort by trend strength
        trending.sort(key=lambda x: x['trend_strength'], reverse=True)
        return trending[:5]  # Top 5 trending
    
    def _identify_declining_topics(self, trend_matrix):
        """Identify topics with downward trend"""
        declining = []
        
        for topic in trend_matrix.index:
            topic_data = trend_matrix.loc[topic].values
            
            if len(topic_data) >= 7:
                recent_avg = np.mean(topic_data[-7:])
                earlier_avg = np.mean(topic_data[-14:-7])
                
                if recent_avg > 0 and recent_avg < earlier_avg * 0.7:  # 30% decrease
                    trend_strength = (earlier_avg - recent_avg) / earlier_avg
                    declining.append({
                        'topic': topic,
                        'decline_strength': trend_strength,
                        'recent_avg': recent_avg,
                        'earlier_avg': earlier_avg
                    })
        
        declining.sort(key=lambda x: x['decline_strength'], reverse=True)
        return declining[:5]
    
    def _identify_emerging_topics(self, trend_matrix):
        """Identify new topics that appeared recently"""
        emerging = []
        
        for topic in trend_matrix.index:
            topic_data = trend_matrix.loc[topic].values
            
            # Check if topic has zero activity in first half but activity in second half
            first_half = topic_data[:len(topic_data)//2]
            second_half = topic_data[len(topic_data)//2:]
            
            first_half_sum = np.sum(first_half)
            second_half_sum = np.sum(second_half)
            
            if first_half_sum == 0 and second_half_sum > 0:
                emerging.append({
                    'topic': topic,
                    'recent_activity': second_half_sum,
                    'emergence_point': len(first_half)
                })
        
        emerging.sort(key=lambda x: x['recent_activity'], reverse=True)
        return emerging
    
    def _identify_stable_topics(self, trend_matrix):
        """Identify topics with stable, consistent frequency"""
        stable = []
        
        for topic in trend_matrix.index:
            topic_data = trend_matrix.loc[topic].values
            
            if len(topic_data) >= 7:
                # Calculate coefficient of variation
                mean_freq = np.mean(topic_data)
                std_freq = np.std(topic_data)
                
                if mean_freq > 0:
                    cv = std_freq / mean_freq
                    
                    # Low coefficient of variation indicates stability
                    if cv < 0.5 and mean_freq >= 2:  # Stable and active
                        stable.append({
                            'topic': topic,
                            'stability_score': 1 - cv,
                            'avg_frequency': mean_freq,
                            'coefficient_variation': cv
                        })
        
        stable.sort(key=lambda x: x['stability_score'], reverse=True)
        return stable[:5]
    
    def _identify_volatile_topics(self, trend_matrix):
        """Identify topics with high frequency variation"""
        volatile = []
        
        for topic in trend_matrix.index:
            topic_data = trend_matrix.loc[topic].values
            
            if len(topic_data) >= 7:
                mean_freq = np.mean(topic_data)
                std_freq = np.std(topic_data)
                
                if mean_freq > 0:
                    cv = std_freq / mean_freq
                    
                    # High coefficient of variation indicates volatility
                    if cv > 1.0 and mean_freq >= 1:
                        volatile.append({
                            'topic': topic,
                            'volatility_score': cv,
                            'avg_frequency': mean_freq,
                            'max_frequency': np.max(topic_data),
                            'min_frequency': np.min(topic_data)
                        })
        
        volatile.sort(key=lambda x: x['volatility_score'], reverse=True)
        return volatile[:5]
    
    def _analyze_peaks(self, trend_matrix):
        """Analyze peak frequency days for each topic"""
        peak_analysis = {}
        
        for topic in trend_matrix.index:
            topic_data = trend_matrix.loc[topic].values
            dates = trend_matrix.columns
            
            if len(topic_data) > 0:
                max_freq = np.max(topic_data)
                max_date_idx = np.argmax(topic_data)
                max_date = dates[max_date_idx]
                
                peak_analysis[topic] = {
                    'peak_frequency': int(max_freq),
                    'peak_date': str(max_date),
                    'peak_percentage': (max_freq / np.sum(topic_data) * 100) if np.sum(topic_data) > 0 else 0
                }
        
        return peak_analysis
    
    def _generate_statistical_summary(self, trend_matrix):
        """Generate comprehensive statistical summary"""
        if trend_matrix.empty:
            return {}
        
        summary = {}
        
        # Overall statistics
        summary['total_topics'] = len(trend_matrix)
        summary['total_mentions'] = int(trend_matrix.sum().sum())
        summary['date_range_days'] = len(trend_matrix.columns)
        summary['avg_daily_mentions'] = float(trend_matrix.sum().mean())
        
        # Topic statistics
        topic_totals = trend_matrix.sum(axis=1)
        summary['most_active_topic'] = {
            'name': topic_totals.idxmax(),
            'total_mentions': int(topic_totals.max())
        }
        summary['least_active_topic'] = {
            'name': topic_totals.idxmin(),
            'total_mentions': int(topic_totals.min())
        }
        
        # Daily statistics  
        daily_totals = trend_matrix.sum(axis=0)
        summary['busiest_day'] = {
            'date': str(daily_totals.idxmax()),
            'total_mentions': int(daily_totals.max())
        }
        summary['quietest_day'] = {
            'date': str(daily_totals.idxmin()),
            'total_mentions': int(daily_totals.min())
        }
        
        # Distribution statistics
        all_values = trend_matrix.values.flatten()
        summary['frequency_distribution'] = {
            'mean': float(np.mean(all_values)),
            'median': float(np.median(all_values)),
            'std': float(np.std(all_values)),
            'max': int(np.max(all_values)),
            'min': int(np.min(all_values))
        }
        
        return summary
    
    def export_comprehensive_report(self, analysis_date, window_days=30, output_dir="output"):
        """Export comprehensive trend report with all insights"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate comprehensive trends
        trend_analysis = self.generate_comprehensive_trends(analysis_date, window_days)
        
        if trend_analysis['trend_matrix'].empty:
            print("   No trend data available for export")
            return None, None
        
        # Create timestamps for files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{output_dir}/comprehensive_trend_report_{analysis_date}_{timestamp}.csv"
        html_filename = f"{output_dir}/comprehensive_trend_report_{analysis_date}_{timestamp}.html"
        
        # Export CSV
        trend_analysis['trend_matrix'].to_csv(csv_filename)
        
        # Export enhanced HTML report
        html_content = self._generate_enhanced_html_report(trend_analysis, analysis_date, window_days)
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"    Comprehensive reports exported:")
        print(f"      CSV: {csv_filename}")
        print(f"      HTML: {html_filename}")
        
        return csv_filename, html_filename
    
    def _generate_enhanced_html_report(self, trend_analysis, analysis_date, window_days):
        """Generate enhanced HTML report with insights and visualizations"""
        trend_matrix = trend_analysis['trend_matrix']
        insights = trend_analysis['insights']
        statistics = trend_analysis['statistics']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Trend Analysis Report - {analysis_date}</title>
            <style>
                body {{ 
                    font-family: 'Times New Roman', serif; 
                    margin: 20px; 
                    background-color: white;
                    color: black;
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border: 2px solid #000080;
                    border-radius: 5px;
                }}
                h1 {{ 
                    color: black; 
                    text-align: center;
                    border-bottom: 3px solid #000080;
                    padding-bottom: 15px;
                    font-size: 28px;
                    font-weight: bold;
                    margin-bottom: 30px;
                }}
                h2 {{ 
                    color: #000080; 
                    border-left: 4px solid #000080;
                    padding-left: 15px;
                    margin-top: 40px;
                    font-size: 20px;
                    font-weight: bold;
                }}
                h3 {{ 
                    color: black;
                    font-size: 16px;
                    font-weight: bold;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border: 2px solid #000080;
                    text-align: center;
                }}
                .stat-number {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #000080;
                }}
                .stat-label {{
                    color: black;
                    margin-top: 10px;
                    font-weight: bold;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 30px 0;
                    border: 2px solid black;
                }}
                th, td {{ 
                    border: 1px solid black; 
                    padding: 12px; 
                    text-align: center;
                    font-family: 'Times New Roman', serif;
                }}
                th {{ 
                    background-color: #000080; 
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                }}
                .trend-table td {{
                    background-color: white;
                    color: black;
                }}
                .high {{ background-color: #000080 !important; color: white; font-weight: bold; }}
                .medium {{ background-color: #4169E1 !important; color: white; }}
                .low {{ background-color: #E6F3FF !important; color: black; }}
                .insights-section {{
                    background: #F8F8FF;
                    padding: 25px;
                    border: 2px solid #000080;
                    margin: 30px 0;
                }}
                .insight-item {{
                    background: white;
                    padding: 15px;
                    margin: 15px 0;
                    border: 1px solid #000080;
                    border-left: 4px solid #000080;
                }}
                .trending {{ border-left-color: #000080 !important; border-left-width: 6px !important; }}
                .declining {{ border-left-color: #4169E1 !important; border-left-width: 6px !important; }}
                .emerging {{ border-left-color: #191970 !important; border-left-width: 6px !important; }}
                .stable {{ border-left-color: #708090 !important; border-left-width: 6px !important; }}
                .volatile {{ border-left-color: #483D8B !important; border-left-width: 6px !important; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Comprehensive Trend Analysis Report</h1>
                <p style="text-align: center; font-size: 1.2em; color: #7f8c8d;">
                    Analysis Period: {trend_analysis['date_range']['start']} to {trend_analysis['date_range']['end']} 
                    ({window_days} days window)
                </p>
                <p style="text-align: center; color: #95a5a6;">
                    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </p>
                
                <h2>Key Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{statistics.get('total_topics', 0)}</div>
                        <div class="stat-label">Total Topics</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{statistics.get('total_mentions', 0)}</div>
                        <div class="stat-label">Total Mentions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{statistics.get('avg_daily_mentions', 0):.1f}</div>
                        <div class="stat-label">Avg Daily Mentions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{statistics.get('date_range_days', 0)}</div>
                        <div class="stat-label">Days Analyzed</div>
                    </div>
                </div>
                
                <h2>Trend Insights</h2>
                <div class="insights-section">
                    {self._generate_insights_html(insights)}
                </div>
                
                <h2>Complete Trend Matrix</h2>
                <p style="color: #7f8c8d; margin-bottom: 20px;">
                    <strong>Format:</strong> Rows = Topics | Columns = Dates | Cells = Frequency counts
                </p>
                {self._style_trend_table(trend_matrix.to_html(classes='trend-table', escape=False))}
                
                <h2>Most Active Topics</h2>
                {self._generate_top_topics_table(trend_matrix)}
                
                <h2>Daily Activity Summary</h2>
                {self._generate_daily_summary_table(trend_matrix)}
                
                <footer style="margin-top: 50px; text-align: center; color: #95a5a6; border-top: 1px solid #bdc3c7; padding-top: 20px;">
                    <p>Automated Trend Analysis System | Powered by Agentic AI</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_insights_html(self, insights):
        """Generate HTML for trend insights"""
        html = ""
        
        if insights.get('trending_topics'):
            html += "<h3>Trending Topics (Increasing)</h3>"
            for item in insights['trending_topics'][:3]:
                html += f"""
                <div class="insight-item trending">
                    <strong>{item['topic']}</strong> - 
                    Trend Strength: {item['trend_strength']:.1%} increase 
                    (Recent: {item['recent_avg']:.1f}, Earlier: {item['earlier_avg']:.1f})
                </div>
                """
        
        if insights.get('declining_topics'):
            html += "<h3>Declining Topics</h3>"
            for item in insights['declining_topics'][:3]:
                html += f"""
                <div class="insight-item declining">
                    <strong>{item['topic']}</strong> - 
                    Decline: {item['decline_strength']:.1%} decrease
                    (Recent: {item['recent_avg']:.1f}, Earlier: {item['earlier_avg']:.1f})
                </div>
                """
        
        if insights.get('emerging_topics'):
            html += "<h3>Emerging Topics</h3>"
            for item in insights['emerging_topics'][:3]:
                html += f"""
                <div class="insight-item emerging">
                    <strong>{item['topic']}</strong> - 
                    New topic with {item['recent_activity']} mentions in recent period
                </div>
                """
        
        if insights.get('stable_topics'):
            html += "<h3>Stable Topics</h3>"
            for item in insights['stable_topics'][:3]:
                html += f"""
                <div class="insight-item stable">
                    <strong>{item['topic']}</strong> - 
                    Consistent activity (Avg: {item['avg_frequency']:.1f}, Stability: {item['stability_score']:.1%})
                </div>
                """
        
        return html if html else "<p>No significant trends detected in this period.</p>"
    
    def _style_trend_table(self, html_table):
        """Add color coding to trend table based on frequency values"""
        return html_table.replace('<table', '<table class="trend-table"')
    
    def _generate_top_topics_table(self, trend_matrix):
        """Generate table of most active topics"""
        if trend_matrix.empty:
            return "<p>No data available</p>"
        
        topic_totals = trend_matrix.sum(axis=1).sort_values(ascending=False).head(10)
        
        html = "<table><tr><th>Rank</th><th>Topic</th><th>Total Mentions</th><th>Avg per Day</th></tr>"
        
        for rank, (topic, total) in enumerate(topic_totals.items(), 1):
            avg_per_day = total / len(trend_matrix.columns)
            html += f"<tr><td>{rank}</td><td>{topic}</td><td>{total}</td><td>{avg_per_day:.1f}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_daily_summary_table(self, trend_matrix):
        """Generate daily activity summary"""
        if trend_matrix.empty:
            return "<p>No data available</p>"
        
        daily_totals = trend_matrix.sum(axis=0)
        
        html = "<table><tr><th>Date</th><th>Total Mentions</th><th>Active Topics</th></tr>"
        
        for date, total in daily_totals.items():
            active_topics = (trend_matrix[date] > 0).sum()
            html += f"<tr><td>{date}</td><td>{total}</td><td>{active_topics}</td></tr>"
        
        html += "</table>"
        return html

if __name__ == "__main__":
    generator = AdvancedTrendGenerator()
    print("Advanced Trend Generation System initialized!")