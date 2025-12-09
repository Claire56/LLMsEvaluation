"""
Create interactive dashboard visualizations for evaluation results.

Shows comparisons across:
- Accuracy (LLM judge scores)
- Latency (response time)
- Cost (API costs)
- ROUGE/BLEU scores
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict
import os


def generate_summary_stats(results: List[Dict]) -> Dict:
    """
    Generate summary statistics from evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Dictionary with aggregated statistics
    """
    df = pd.DataFrame([
        {
            'template': r['prompt_template'],
            'accuracy': r['metrics']['judge']['overall'],
            'relevance': r['metrics']['judge']['relevance'],
            'rouge_l': r['metrics']['rouge']['rougeL'],
            'bleu': r['metrics']['bleu'],
            'latency': r['latency'],
            'cost': r['cost'],
            'hallucination_risk': r['metrics']['hallucination_risk']
        }
        for r in results
    ])
    
    summary = {}
    for template in df['template'].unique():
        template_df = df[df['template'] == template]
        summary[template] = {
            'avg_accuracy': template_df['accuracy'].mean(),
            'avg_relevance': template_df['relevance'].mean(),
            'avg_rouge_l': template_df['rouge_l'].mean(),
            'avg_bleu': template_df['bleu'].mean(),
            'avg_latency': template_df['latency'].mean(),
            'total_cost': template_df['cost'].sum(),
            'avg_hallucination_risk': template_df['hallucination_risk'].mean(),
            'num_evaluations': len(template_df)
        }
    
    return summary


def create_dashboard(results: List[Dict], output_file: str = "evaluation_dashboard.html"):
    """
    Create an interactive HTML dashboard from evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        output_file: Path to save the HTML dashboard
    """
    df = pd.DataFrame([
        {
            'template': r['prompt_template'],
            'accuracy': r['metrics']['judge']['overall'],
            'relevance': r['metrics']['judge']['relevance'],
            'completeness': r['metrics']['judge']['completeness'],
            'clarity': r['metrics']['judge']['clarity'],
            'rouge_1': r['metrics']['rouge']['rouge1'],
            'rouge_2': r['metrics']['rouge']['rouge2'],
            'rouge_l': r['metrics']['rouge']['rougeL'],
            'bleu': r['metrics']['bleu'],
            'latency': r['latency'],
            'cost': r['cost'],
            'hallucination_risk': r['metrics']['hallucination_risk']
        }
        for r in results
    ])
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Accuracy by Prompt Template',
            'Latency by Prompt Template',
            'Cost by Prompt Template',
            'ROUGE-L Score by Template',
            'BLEU Score by Template',
            'Hallucination Risk by Template'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Calculate averages by template
    template_stats = df.groupby('template').agg({
        'accuracy': 'mean',
        'latency': 'mean',
        'cost': 'sum',
        'rouge_l': 'mean',
        'bleu': 'mean',
        'hallucination_risk': 'mean'
    }).reset_index()
    
    templates = template_stats['template'].tolist()
    
    # 1. Accuracy
    fig.add_trace(
        go.Bar(
            x=templates,
            y=template_stats['accuracy'],
            name='Accuracy',
            marker_color='#2ecc71',
            text=[f"{v:.3f}" for v in template_stats['accuracy']],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # 2. Latency
    fig.add_trace(
        go.Bar(
            x=templates,
            y=template_stats['latency'],
            name='Latency (s)',
            marker_color='#e74c3c',
            text=[f"{v:.2f}s" for v in template_stats['latency']],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # 3. Cost
    fig.add_trace(
        go.Bar(
            x=templates,
            y=template_stats['cost'],
            name='Total Cost ($)',
            marker_color='#f39c12',
            text=[f"${v:.4f}" for v in template_stats['cost']],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 4. ROUGE-L
    fig.add_trace(
        go.Bar(
            x=templates,
            y=template_stats['rouge_l'],
            name='ROUGE-L',
            marker_color='#3498db',
            text=[f"{v:.3f}" for v in template_stats['rouge_l']],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    # 5. BLEU
    fig.add_trace(
        go.Bar(
            x=templates,
            y=template_stats['bleu'],
            name='BLEU',
            marker_color='#9b59b6',
            text=[f"{v:.3f}" for v in template_stats['bleu']],
            textposition='outside'
        ),
        row=3, col=1
    )
    
    # 6. Hallucination Risk
    fig.add_trace(
        go.Bar(
            x=templates,
            y=template_stats['hallucination_risk'],
            name='Hallucination Risk',
            marker_color='#e67e22',
            text=[f"{v:.3f}" for v in template_stats['hallucination_risk']],
            textposition='outside'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="LLM Evaluation Dashboard: Prompt Template Comparison",
        height=1200,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Prompt Template", row=i, col=j)
    
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Seconds", row=1, col=2)
    fig.update_yaxes(title_text="Dollars", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=2)
    fig.update_yaxes(title_text="Score", row=3, col=1)
    fig.update_yaxes(title_text="Risk Score", row=3, col=2)
    
    # Create comparison scatter plot
    fig2 = go.Figure()
    
    for template in templates:
        template_df = df[df['template'] == template]
        fig2.add_trace(go.Scatter(
            x=template_df['latency'],
            y=template_df['accuracy'],
            mode='markers',
            name=template,
            text=[f"Cost: ${c:.4f}" for c in template_df['cost']],
            hovertemplate='<b>%{text}</b><br>Latency: %{x:.2f}s<br>Accuracy: %{y:.3f}<extra></extra>'
        ))
    
    fig2.update_layout(
        title="Accuracy vs Latency Trade-off (bubble size = cost)",
        xaxis_title="Latency (seconds)",
        yaxis_title="Accuracy Score",
        hovermode='closest'
    )
    
    # Create comprehensive comparison table
    summary_stats = generate_summary_stats(results)
    summary_df = pd.DataFrame(summary_stats).T
    summary_df = summary_df.round(4)
    
    # Combine all visualizations in HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Evaluation Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .summary {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 LLM Evaluation Dashboard</h1>
            
            <div class="summary">
                <h2>📊 Summary Statistics</h2>
                <p>Total Evaluations: {len(results)}</p>
                <p>Number of Prompt Templates: {len(summary_stats)}</p>
                <p>Q&A Pairs Evaluated: {len(results) // len(summary_stats)}</p>
            </div>
            
            <h2>📈 Detailed Metrics Comparison</h2>
            <div id="main-chart"></div>
            
            <h2>⚖️ Accuracy vs Latency Trade-off</h2>
            <div id="scatter-chart"></div>
            
            <h2>📋 Summary Table</h2>
            <div id="summary-table"></div>
        </div>
        
        <script>
            var mainChart = {fig.to_json()};
            Plotly.newPlot('main-chart', mainChart.data, mainChart.layout, {{responsive: true}});
            
            var scatterChart = {fig2.to_json()};
            Plotly.newPlot('scatter-chart', scatterChart.data, scatterChart.layout, {{responsive: true}});
            
            var summaryData = {summary_df.to_dict('records')};
            var tableHtml = '<table><thead><tr>';
            {chr(10).join([f"tableHtml += '<th>{col}</th>';" for col in summary_df.columns])}
            tableHtml += '</tr></thead><tbody>';
            summaryData.forEach(function(row) {{
                tableHtml += '<tr>';
                {chr(10).join([f"tableHtml += '<td>' + (row['{col}'] !== undefined ? row['{col}'] : '') + '</td>';" for col in summary_df.columns])}
                tableHtml += '</tr>';
            }});
            tableHtml += '</tbody></table>';
            document.getElementById('summary-table').innerHTML = tableHtml;
        </script>
    </body>
    </html>
    """
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard saved to {output_file}")
    print(f"\nSummary Statistics:")
    print(summary_df.to_string())
    
    return summary_stats

