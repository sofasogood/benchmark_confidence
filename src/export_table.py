import json
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

def get_score_color(score):
    """Return color based on score."""
    if score == 3:
        return "#4CAF50"  # Green
    elif score == 2:
        return "#8BC34A"  # Light Green
    elif score == 1:
        return "#FFC107"  # Yellow
    else:
        return "#F44336"  # Red

def create_radar_plot(results):
    """Create a radar plot of the metric scores."""
    # Define metric info
    metric_info = {
        'construct_validity': 'Construct Validity',
        'coverage': 'Coverage',
        'external_validity': 'External Validity',
        'difficulty_discrimination': 'Difficulty & Discrimination',
        'robustness': 'Robustness',
        'power_ci': 'Power'
    }
    
    # Prepare data
    categories = []
    scores = []
    colors = []
    
    for metric, name in metric_info.items():
        if metric in results:
            score = results[metric].get('score', 0)
            categories.append(name)
            scores.append(score)
            colors.append(get_score_color(score))
    # Close the polygon by repeating the first value
    if categories and scores:
        categories.append(categories[0])
        scores.append(scores[0])
        colors.append(colors[0])
    
    # Create radar plot
    fig = go.Figure()
    
    # Add the main radar trace
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Scores',
        line=dict(color='#2196F3', width=3),
        fillcolor='rgba(33, 150, 243, 0.2)',
        marker=dict(size=10, color='#2196F3', line=dict(width=2, color='white')),
        mode='lines+markers'
    ))
    
    # Add score dots with color and score labels
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        mode='markers+text',
        marker=dict(size=16, color=colors, line=dict(width=2, color='white')),
        text=[str(s) for s in scores],
        textposition='top center',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text='Benchmark Confidence Radar', x=0.5, font=dict(size=24, family='Arial Black')),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 3],
                ticktext=['Poor', 'Fair', 'Good', 'Excellent'],
                tickvals=[0, 1, 2, 3],
                tickangle=0,
                gridcolor='#E0E0E0',
                linecolor='#E0E0E0',
                tickfont=dict(size=14, family='Arial Black')
            ),
            angularaxis=dict(
                gridcolor='#E0E0E0',
                linecolor='#E0E0E0',
                tickfont=dict(size=14, family='Arial Black')
            ),
            bgcolor='white'
        ),
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=120, r=120, t=120, b=120),
        width=1000,
        height=750
    )
    
    # Save as HTML
    output_dir = "data/metrics_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "confidence_radar.html")
    pio.write_html(fig, output_path)
    print(f"Radar plot saved to {output_path}")

def create_html_table():
    # Load the results
    results_path = "data/metrics_results/all_metrics_results.json"
    with open(results_path) as f:
        results = json.load(f)
    
    # Define metric info
    metric_info = {
        'construct_validity': {
            'name': 'Construct Validity',
            'description': 'Consistency of LLM subject tagging',
            'levels': {
                3: 'Excellent (&#954; &ge; 0.8)',
                2: 'Good (0.6 &le; &#954; &lt; 0.8)',
                1: 'Fair (0.4 &le; &#954; &lt; 0.6)',
                0: 'Poor (&#954; &lt; 0.4)'
            }
        },
        'coverage': {
            'name': 'Coverage',
            'description': 'Distribution of questions across subjects',
            'levels': {
                3: 'Excellent (H/Hmax &ge; 0.90)',
                2: 'Good (0.75 &le; H/Hmax &lt; 0.90)',
                1: 'Fair (0.50 &le; H/Hmax &lt; 0.75)',
                0: 'Poor (H/Hmax &lt; 0.50)'
            }
        },
        'external_validity': {
            'name': 'External Validity',
            'description': 'Performance gap between STEM and non-STEM',
            'levels': {
                3: 'Excellent (gap &le; 2pp)',
                2: 'Good (2pp &lt; gap &le; 5pp)',
                1: 'Fair (5pp &lt; gap &le; 10pp)',
                0: 'Poor (gap &gt; 10pp)'
            }
        },
        'difficulty_discrimination': {
            'name': 'Difficulty & Discrimination',
            'description': 'Distribution of question difficulty',
            'levels': {
                3: 'Excellent (&lt; 5% ceiling/floor)',
                2: 'Good (5-10% ceiling/floor)',
                1: 'Fair (10-20% ceiling/floor)',
                0: 'Poor (&gt; 20% ceiling/floor)'
            }
        },
        'robustness': {
            'name': 'Robustness',
            'description': 'Performance stability under perturbations',
            'levels': {
                3: 'Excellent (drop &le; 2pp)',
                2: 'Good (2pp &lt; drop &le; 5pp)',
                1: 'Fair (5pp &lt; drop &le; 10pp)',
                0: 'Poor (drop &gt; 10pp)'
            }
        },
        'power_ci': {
            'name': 'Power',
            'description': 'Statistical confidence in results',
            'levels': {
                3: 'Excellent (CI width &le; 2pp)',
                2: 'Good (2pp &lt; CI width &le; 5pp)',
                1: 'Fair (5pp &lt; CI width &le; 10pp)',
                0: 'Poor (CI width &gt; 10pp)'
            }
        }
    }

    # Create HTML
    html = """
    <html>
    <head>
    <meta charset='UTF-8'>
    <style>
        .confidence-table {
            border-collapse: collapse;
            width: 100%;
            max-width: 1200px;
            margin: 20px auto;
            font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .confidence-table th {
            background-color: #f8f9fa;
            padding: 12px 15px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
            font-weight: 600;
        }
        .confidence-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #dee2e6;
            line-height: 1.4;
        }
        .confidence-table tr:hover {
            background-color: #f5f5f5;
        }
        .score-cell {
            font-weight: bold;
            text-align: center;
            border-radius: 4px;
            padding: 4px 8px;
            color: white;
        }
        .metric-name {
            font-weight: 600;
        }
        .metric-description {
            color: #666;
            font-size: 0.9em;
        }
        .metadata {
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }
    </style>
    </head>
    <body>
    <table class=\"confidence-table\">\n        <tr>\n            <th>Metric</th>\n            <th>Score</th>\n            <th>Confidence Level</th>\n            <th>Details</th>\n        </tr>\n    """

    # Add rows
    for metric, info in metric_info.items():
        if metric in results:
            result = results[metric]
            score = result.get('score', 0)
            score_color = get_score_color(score)
            
            # Get metadata
            metadata = ""
            if metric == 'coverage':
                metadata = f"H/Hmax = {result.get('normalized_score', 'N/A'):.3f}"
            elif metric == 'external_validity':
                metadata = f"Gap = {result.get('accuracy_gap', 'N/A'):.1f}pp"
            elif metric == 'difficulty_discrimination':
                metadata = f"{result.get('ceiling_percentage', 'N/A'):.1f}% ceiling, {result.get('floor_percentage', 'N/A'):.1f}% floor"
            elif metric == 'robustness':
                metadata = f"Drop = {result.get('accuracy_drop', 'N/A'):.1f}pp"
            elif metric == 'power_ci':
                metadata = f"CI width = {result.get('ci_width', 'N/A'):.1f}pp, Accuracy = {result.get('accuracy', 'N/A'):.1f}%"

            html += f"""
            <tr>
                <td>
                    <div class="metric-name">{info['name']}</div>
                    <div class="metric-description">{info['description']}</div>
                </td>
                <td><div class="score-cell" style="background-color: {score_color}">{score}</div></td>
                <td>{info['levels'][score]}</td>
                <td><span class="metadata">{metadata}</span></td>
            </tr>
            """

    # Calculate overall confidence
    metric_keys = [k for k in metric_info.keys() if k in results]
    scores = [results[k]['score'] for k in metric_keys if 'score' in results[k]]
    if scores:
        avg_score = sum(scores) / len(scores)
        if avg_score >= 2.5:
            overall_label = 'Excellent'
            overall_color = get_score_color(3)
        elif avg_score >= 2.0:
            overall_label = 'Good'
            overall_color = get_score_color(2)
        elif avg_score >= 1.0:
            overall_label = 'Fair'
            overall_color = get_score_color(1)
        else:
            overall_label = 'Poor'
            overall_color = get_score_color(0)
        html += f"""
        <tr style='font-size:1.1em;font-weight:bold;'>
            <td>Overall Confidence</td>
            <td><div class='score-cell' style='background-color: {overall_color}'>{avg_score:.2f}</div></td>
            <td colspan='2' style='font-weight:bold;color:{overall_color};'>{overall_label}</td>
        </tr>
        """

    html += """
    </table>
    </body>
    </html>
    """

    # Save HTML
    output_dir = "data/metrics_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "confidence_table.html")
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"HTML table saved to {output_path}")

def main():
    # Load results
    results_path = "data/metrics_results/all_metrics_results.json"
    with open(results_path) as f:
        results = json.load(f)
    
    # Create both visualizations
    create_html_table()
    create_radar_plot(results)

if __name__ == "__main__":
    main() 