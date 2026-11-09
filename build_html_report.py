import markdown
from pathlib import Path
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def main():
    project_root = Path(__file__).resolve().parent
    report_path = project_root / "reports" / "final_research_report.md"
    figures_dir = project_root / "results" / "figures"
    out_html = project_root / "NSE_Stock_Prediction_Research_Report.html"
    
    with open(report_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert markdown to HTML with tables support
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

    # Inject figures into the HTML body
    
    # 1. Fig 1 under Predictive Performance
    fig1_b64 = get_base64_image(figures_dir / "fig1_model_accuracy_auc.png")
    fig1_html = f'<div class="figure"><img src="data:image/png;base64,{fig1_b64}" alt="Model Accuracy & AUC"><p class="caption">Figure 1: Predictive Performance (Accuracy and ROC-AUC) across 1-day and 5-day horizons.</p></div>'
    html_body = html_body.replace('## 4. Predictive Performance (Classification)', '## 4. Predictive Performance (Classification)\n' + fig1_html)

    # 1b. Fig 8 under Regression Metrics
    fig8_b64 = get_base64_image(figures_dir / "fig8_regression_metrics.png")
    fig8_html = f'<div class="figure"><img src="data:image/png;base64,{fig8_b64}" alt="Regression Metrics"><p class="caption">Figure 8: Regression Metrics (MAE, MSE, RMSE, MAPE, R²) across 1-day and 5-day horizons.</p></div>'
    html_body = html_body.replace('## 4b. Regression Metrics', '## 4b. Regression Metrics\n' + fig8_html)

    # 2. Fig 2 & 7 under Trading Performance
    fig2_b64 = get_base64_image(figures_dir / "fig2_net_cagr_comparison.png")
    fig7_b64 = get_base64_image(figures_dir / "fig7_equity_curves.png")
    fig2_7_html = f'<div class="figure"><img src="data:image/png;base64,{fig2_b64}" alt="Net CAGR Comparison"><p class="caption">Figure 2: Gross vs Net CAGR after 20 bps Costs.</p></div>'
    fig2_7_html += f'<div class="figure"><img src="data:image/png;base64,{fig7_b64}" alt="Equity Curves"><p class="caption">Figure 3: Net Equity Curves (1-day Horizon) across 6 walk-forward test periods.</p></div>'
    html_body = html_body.replace('## 7. Trading Performance & Transaction Costs', '## 7. Trading Performance & Transaction Costs\n' + fig2_7_html)

    # 3. Fig 3 under Robustness & Sensitivity
    fig3_b64 = get_base64_image(figures_dir / "fig3_cost_sensitivity.png")
    fig3_html = f'<div class="figure"><img src="data:image/png;base64,{fig3_b64}" alt="Cost Sensitivity"><p class="caption">Figure 4: Transaction Cost Sensitivity Sweep (0-50 bps).</p></div>'
    html_body = html_body.replace('## 8. Robustness & Sensitivity', '## 8. Robustness & Sensitivity\n' + fig3_html)

    # 4. Fig 4, 5, 6 under Feature Importance
    fig4_b64 = get_base64_image(figures_dir / "fig4_shap_importance_1d.png")
    fig5_b64 = get_base64_image(figures_dir / "fig5_shap_importance_5d.png")
    fig6_b64 = get_base64_image(figures_dir / "fig6_feature_stability.png")
    
    fig_shap_html = f'<div class="figure-row"><div class="figure"><img src="data:image/png;base64,{fig4_b64}" alt="SHAP 1d"><p class="caption">Figure 5: Top 20 SHAP Features (1-day).</p></div>'
    fig_shap_html += f'<div class="figure"><img src="data:image/png;base64,{fig5_b64}" alt="SHAP 5d"><p class="caption">Figure 6: Top 20 SHAP Features (5-day).</p></div></div>'
    fig_shap_html += f'<div class="figure"><img src="data:image/png;base64,{fig6_b64}" alt="Feature Stability"><p class="caption">Figure 7: Feature Ranking Stability (Spearman r) Across Retraining Splits.</p></div>'
    
    html_body = html_body.replace('## 9. Feature Importance (SHAP)', '## 9. Feature Importance (SHAP)\n' + fig_shap_html)

    # Define a clean, modern CSS
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSE ML Project - Final Report</title>
    <style>
        :root {{
            --bg-color: #f9f9fb;
            --text-color: #2b2b2b;
            --link-color: #0366d6;
            --border-color: #e1e4e8;
            --code-bg: #f6f8fa;
            --table-header-bg: #f6f8fa;
            --accent-color: #24292e;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        .container {{
            background: #ffffff;
            padding: 50px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}
        h1, h2, h3, h4 {{
            color: var(--accent-color);
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        h1 {{ font-size: 2em; padding-bottom: 0.3em; border-bottom: 1px solid var(--border-color); text-align: center; }}
        h2 {{ font-size: 1.5em; padding-bottom: 0.3em; border-bottom: 1px solid var(--border-color); margin-top: 40px; }}
        h3 {{ font-size: 1.25em; }}
        p {{ margin-top: 0; margin-bottom: 16px; }}
        a {{ color: var(--link-color); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
        table {{
            border-spacing: 0;
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 24px;
            font-size: 0.9em;
        }}
        table th, table td {{
            padding: 10px 14px;
            border: 1px solid var(--border-color);
            text-align: left;
        }}
        table th {{
            background-color: var(--table-header-bg);
            font-weight: 600;
        }}
        table tr:nth-child(even) {{ background-color: #fafbfc; }}
        
        blockquote {{
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
            margin: 0 0 16px 0;
            background: #fdfdfd;
            padding: 10px 16px;
            border-radius: 4px;
        }}
        
        hr {{
            height: 0.25em;
            padding: 0;
            margin: 32px 0;
            background-color: #e1e4e8;
            border: 0;
        }}
        
        .figure {{
            margin: 30px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .caption {{
            font-size: 0.85em;
            color: #555;
            margin-top: 8px;
            font-style: italic;
        }}
        .figure-row {{
            display: flex;
            gap: 20px;
            justify-content: space-between;
        }}
        .figure-row .figure {{
            width: 48%;
            margin: 10px 0;
        }}
        
        /* Utility for yes/no checks */
        td:contains("Yes"), td:contains("Confirmed") {{ font-weight: bold; color: #22863a; }}
        td:contains("No") {{ font-weight: bold; color: #cb2431; }}
        td:contains("Partially") {{ font-weight: bold; color: #b08800; }}
        
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print(f"Successfully created comprehensive HTML report at: {out_html}")

if __name__ == "__main__":
    main()
