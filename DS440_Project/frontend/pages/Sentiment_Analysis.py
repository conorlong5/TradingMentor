import streamlit as st
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Sentiment Analysis", layout="wide", initial_sidebar_state="collapsed")

load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

# ----------------------------
# CUSTOM STYLING
# ----------------------------
st.markdown("""
<style>
/* Enhanced styling for metrics and cards */
[data-testid="stMetricValue"] {
    font-size: 32px;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    font-size: 16px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.article-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    transition: all 0.3s ease;
}

.article-card:hover {
    border-color: rgba(39, 238, 245, 0.5);
    box-shadow: 0 4px 20px rgba(39, 238, 245, 0.2);
}

.sentiment-positive { border-left: 4px solid #00C853; }
.sentiment-negative { border-left: 4px solid #E53935; }
.sentiment-neutral { border-left: 4px solid #FFA726; }

.chart-wrapper {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
    border: 2px solid rgba(39, 238, 245, 0.4);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25), 0 0 15px rgba(39, 238, 245, 0.1);
}

.chart-title {
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    color: #27EEF5;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(39, 238, 245, 0.2);
}

/* Target Streamlit plotly charts that follow chart-wrapper divs */
.chart-wrapper + div,
.chart-wrapper ~ div {
    background: transparent !important;
}

/* Style for columns containing chart wrappers */
div[data-testid="column"] .chart-wrapper {
    position: relative;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER & NAVIGATION
# ----------------------------

col_back, _ = st.columns([1, 11])
with col_back:
    if st.button("← Home", use_container_width=True):
        st.switch_page("app.py")

st.markdown("""
<h1 style='text-align:left; color:#27EEF5; margin-bottom:10px;'>
    📰 Stock Sentiment Analysis
</h1>
<p style='text-align:left; color:#b0b0b0; margin-top:-10px; margin-bottom:20px;'>
    Let the LLM design a trading strategy for you, then backtest it.
</p>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------
# INPUT SECTION
# ----------------------------
col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    symbol = st.text_input("Enter a Stock Symbol:", value="AAPL", help="Enter the ticker symbol (e.g., AAPL, TSLA, MSFT)")

with col2:
    days = st.selectbox("Time Period", options=[7, 14, 30], index=0, help="Number of days to analyze")

with col3:
    num_articles = st.selectbox("Articles to Analyze", options=[10, 20, 50], index=0, help="Number of articles to fetch")

analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

st.markdown("""
<style>
/* Hide the entire sidebar */
[data-testid="stSidebar"] {
    display: none;
}

/* Hide the sidebar toggle (arrow) */
[data-testid="stSidebarNav"] {
    display: none;
}

button[kind="header"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# ANALYSIS SECTION
# ----------------------------
if analyze_btn or (symbol and 'sentiment_data' in st.session_state and st.session_state.get('last_symbol') == symbol):
    with st.spinner(f"🔍 Analyzing sentiment for {symbol.upper()}..."):
        analyzer = SentimentIntensityAnalyzer()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        query = f"{symbol} stock"

        try:
            articles_data = newsapi.get_everything(
        q=query,
        from_param=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d"),
        language="en",
        sort_by="relevancy",
                page_size=num_articles
            )

            if not articles_data.get("articles"):
                st.warning(f"⚠️ No articles found for {symbol.upper()}. Try a different symbol or increase the time period.")
                st.stop()

            # Process articles
            articles_with_sentiment = []
            for article in articles_data["articles"]:
                if not article.get('title') or not article.get('description'):
                    continue
                    
                text = f"{article.get('title', '')} {article.get('description', '')}"
                scores = analyzer.polarity_scores(text)
                
                articles_with_sentiment.append({
                    'title': article.get('title', 'N/A'),
                    'description': article.get('description', 'N/A'),
                    'url': article.get('url', '#'),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', ''),
                    'compound': scores['compound'],
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu']
                })

            if not articles_with_sentiment:
                st.error("⚠️ No valid articles found. Please try again.")
                st.stop()

            # Calculate statistics
            sentiments = [a['compound'] for a in articles_with_sentiment]
            avg_sentiment = sum(sentiments) / len(sentiments)
            positive_count = sum(1 for s in sentiments if s > 0.05)
            negative_count = sum(1 for s in sentiments if s < -0.05)
            neutral_count = len(sentiments) - positive_count - negative_count

            # Store in session state
            st.session_state.sentiment_data = articles_with_sentiment
            st.session_state.avg_sentiment = avg_sentiment
            st.session_state.last_symbol = symbol

        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.stop()

    # ----------------------------
    # OVERVIEW METRICS
    # ----------------------------
    st.markdown(f"### 📊 Analysis Overview for **{symbol.upper()}**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_label = "Positive 😀" if avg_sentiment > 0.05 else \
            "Negative 😞" if avg_sentiment < -0.05 else "Neutral 😐"
        sentiment_color = "#00C853" if avg_sentiment > 0.05 else \
                         "#E53935" if avg_sentiment < -0.05 else "#FFA726"
        
        st.metric(
            "Overall Sentiment", 
            sentiment_label,
            delta=f"{avg_sentiment:.3f}",
            delta_color="normal"
        )
    
    with col2:
        st.metric("Total Articles", len(articles_with_sentiment))
    
    with col3:
        st.metric("Positive Articles", f"{positive_count} ({positive_count/len(sentiments)*100:.1f}%)")
    
    with col4:
        st.metric("Negative Articles", f"{negative_count} ({negative_count/len(sentiments)*100:.1f}%)")

    st.divider()

    # ----------------------------
    # VISUALIZATIONS
    # ----------------------------
    st.markdown("""
    <style>
    .chart-section-wrapper {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 2px solid rgba(39, 238, 245, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25), 0 0 15px rgba(39, 238, 245, 0.1);
        position: relative;
    }
    </style>
    """, unsafe_allow_html=True)

    
    
    col1, col2 = st.columns(2)

    with col1:
        # Create container with title
        container_id = "sentiment-dist-wrapper"
        st.markdown(f"""
        <div id="{container_id}" class="chart-section-wrapper">
            <div class="chart-title">📈 Sentiment Distribution</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Pie chart
        sentiment_counts = [positive_count, negative_count, neutral_count]
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['#00C853', '#E53935', '#FFA726']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=sentiment_counts,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=14
        )])
        fig_pie.update_layout(
            showlegend=True,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")
        
        # JavaScript to wrap chart in container using MutationObserver
        st.markdown(f"""
        <script>
            (function() {{
                const container = document.getElementById('{container_id}');
                if (!container) return;
                
                function moveChartIntoContainer() {{
                    const column = container.closest('[data-testid="column"]');
                    if (!column) return;
                    
                    const chartDiv = column.querySelector('[data-testid="stPlotlyChart"]');
                    if (chartDiv) {{
                        let chartParent = chartDiv.parentElement;
                        // Go up to find the top-level Streamlit element
                        while (chartParent && chartParent.parentElement && 
                               chartParent.parentElement !== column && 
                               !chartParent.parentElement.hasAttribute('data-testid')) {{
                            chartParent = chartParent.parentElement;
                        }}
                        
                        if (chartParent && !container.contains(chartParent)) {{
                            container.appendChild(chartParent);
                            return true;
                        }}
                    }}
                    return false;
                }}
                
                // Try immediately and on intervals
                if (moveChartIntoContainer()) return;
                
                // Watch for chart to be added
                const observer = new MutationObserver(function(mutations) {{
                    if (moveChartIntoContainer()) {{
                        observer.disconnect();
                    }}
                }});
                
                const column = container.closest('[data-testid="column"]');
                if (column) {{
                    observer.observe(column, {{ childList: true, subtree: true }});
                    setTimeout(() => observer.disconnect(), 3000);
                }}
                
                setTimeout(moveChartIntoContainer, 100);
                setTimeout(moveChartIntoContainer, 500);
                setTimeout(moveChartIntoContainer, 1000);
            }})();
        </script>
        """, unsafe_allow_html=True)

    with col2:
        container_id2 = "sentiment-gauge-wrapper"
        st.markdown(f"""
        <div id="{container_id2}" class="chart-section-wrapper">
            <div class="chart-title">🎯 Sentiment Gauge</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Score"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': sentiment_color},
                'steps': [
                    {'range': [-1, -0.05], 'color': 'rgba(229, 57, 53, 0.2)'},
                    {'range': [-0.05, 0.05], 'color': 'rgba(255, 167, 38, 0.2)'},
                    {'range': [0.05, 1], 'color': 'rgba(0, 200, 83, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': avg_sentiment
                }
            }
        ))
        fig_gauge.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")
        
        st.markdown(f"""
        <script>
            (function() {{
                const container = document.getElementById('{container_id2}');
                if (!container) return;
                
                function moveChartIntoContainer() {{
                    const column = container.closest('[data-testid="column"]');
                    if (!column) return;
                    
                    const chartDiv = column.querySelector('[data-testid="stPlotlyChart"]');
                    if (chartDiv) {{
                        let chartParent = chartDiv.parentElement;
                        while (chartParent && chartParent.parentElement && 
                               chartParent.parentElement !== column && 
                               !chartParent.parentElement.hasAttribute('data-testid')) {{
                            chartParent = chartParent.parentElement;
                        }}
                        
                        if (chartParent && !container.contains(chartParent)) {{
                            container.appendChild(chartParent);
                            return true;
                        }}
                    }}
                    return false;
                }}
                
                if (moveChartIntoContainer()) return;
                
                const observer = new MutationObserver(function(mutations) {{
                    if (moveChartIntoContainer()) {{
                        observer.disconnect();
                    }}
                }});
                
                const column = container.closest('[data-testid="column"]');
                if (column) {{
                    observer.observe(column, {{ childList: true, subtree: true }});
                    setTimeout(() => observer.disconnect(), 3000);
                }}
                
                setTimeout(moveChartIntoContainer, 100);
                setTimeout(moveChartIntoContainer, 500);
                setTimeout(moveChartIntoContainer, 1000);
            }})();
        </script>
        """, unsafe_allow_html=True)

    # Sentiment timeline (if dates are available)
    df_timeline = pd.DataFrame(articles_with_sentiment)
    
    container_id3 = "sentiment-timeline-wrapper"
    st.markdown(f"""
    <div id="{container_id3}" class="chart-section-wrapper">
        <div class="chart-title">📅 Sentiment Timeline</div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'published_at' in df_timeline.columns and df_timeline['published_at'].notna().any():
        df_timeline['published_at'] = pd.to_datetime(df_timeline['published_at'], errors='coerce')
        df_timeline = df_timeline.dropna(subset=['published_at'])
        df_timeline = df_timeline.sort_values('published_at')
        df_timeline['date'] = df_timeline['published_at'].dt.date
        
        daily_sentiment = df_timeline.groupby('date')['compound'].mean().reset_index()
        
        fig_timeline = px.line(
            daily_sentiment, 
            x='date', 
            y='compound',
            markers=True,
            title="Average Sentiment Over Time"
        )
        fig_timeline.update_traces(
            line_color='#27EEF5',
            line_width=3,
            marker_size=8
        )
        fig_timeline.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="rgba(255,255,255,0.5)",
            annotation_text="Neutral Line"
        )
        fig_timeline.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            hovermode='x unified',
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_timeline, use_container_width=True, key="timeline_chart")
        
        # JavaScript to wrap chart in container
        st.markdown(f"""
        <script>
            (function() {{
                const container = document.getElementById('{container_id3}');
                if (!container) return;
                
                function moveChartIntoContainer() {{
                    // Find plotly chart after the container
                    let currentElement = container.nextElementSibling;
                    while (currentElement) {{
                        const chartDiv = currentElement.querySelector('[data-testid="stPlotlyChart"]');
                        if (chartDiv) {{
                            let chartParent = chartDiv.parentElement;
                            while (chartParent && chartParent.parentElement && 
                                   chartParent.parentElement !== currentElement && 
                                   !chartParent.parentElement.hasAttribute('data-testid')) {{
                                chartParent = chartParent.parentElement;
                            }}
                            
                            if (chartParent && !container.contains(chartParent)) {{
                                container.appendChild(chartParent);
                                return true;
                            }}
                        }}
                        currentElement = currentElement.nextElementSibling;
                        if (!currentElement || currentElement.id) break;
                    }}
                    return false;
                }}
                
                if (moveChartIntoContainer()) return;
                
                const observer = new MutationObserver(function(mutations) {{
                    if (moveChartIntoContainer()) {{
                        observer.disconnect();
                    }}
                }});
                
                observer.observe(document.body, {{ childList: true, subtree: true }});
                setTimeout(() => observer.disconnect(), 3000);
                
                setTimeout(moveChartIntoContainer, 100);
                setTimeout(moveChartIntoContainer, 500);
                setTimeout(moveChartIntoContainer, 1000);
            }})();
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="text-align: center; color: #b0b0b0; padding: 20px;">Timeline data not available for all articles.</p>', unsafe_allow_html=True)

    st.divider()

    # ----------------------------
    # DETAILED BREAKDOWN
    # ----------------------------
    st.markdown("#### 📰 Individual Article Analysis")
    
    # Sort articles by sentiment (most positive first)
    sorted_articles = sorted(articles_with_sentiment, key=lambda x: x['compound'], reverse=True)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 All Articles", "😀 Most Positive", "😞 Most Negative"])
    
    with tab1:
        for article in sorted_articles:
            sentiment_class = "sentiment-positive" if article['compound'] > 0.05 else \
                            "sentiment-negative" if article['compound'] < -0.05 else "sentiment-neutral"
            
            st.markdown(f"""
            <div class="article-card {sentiment_class}">
                <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;'>
                    <h4 style='margin: 0; color: white;'>{article['title']}</h4>
                    <span style='font-weight: bold; color: {'#00C853' if article['compound'] > 0.05 else '#E53935' if article['compound'] < -0.05 else '#FFA726'};'>
                        {article['compound']:.3f}
                    </span>
                </div>
                <p style='color: #b0b0b0; margin: 8px 0;'>{article['description'][:200]}...</p>
                <div style='display: flex; justify-content: space-between; font-size: 12px; color: #888; margin-top: 10px;'>
                    <span>📰 {article['source']}</span>
                    <span>📅 {article['published_at'][:10] if article['published_at'] else 'N/A'}</span>
                    <a href="{article['url']}" target="_blank" style='color: #27EEF5; text-decoration: none;'>Read More →</a>
                </div>
                <div style='margin-top: 10px; font-size: 11px; color: #666;'>
                    Positive: {article['positive']:.2%} | Negative: {article['negative']:.2%} | Neutral: {article['neutral']:.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        positive_articles = [a for a in sorted_articles if a['compound'] > 0.05][:10]
        if positive_articles:
            for article in positive_articles:
                st.markdown(f"""
                <div class="article-card sentiment-positive">
                    <h4 style='margin: 0; color: white;'>{article['title']}</h4>
                    <p style='color: #b0b0b0; margin: 8px 0;'>{article['description'][:200]}...</p>
                    <div style='font-size: 12px; color: #888;'>
                        <strong>Score:</strong> {article['compound']:.3f} | 
                        <strong>Source:</strong> {article['source']} | 
                        <a href="{article['url']}" target="_blank" style='color: #27EEF5;'>Read More →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strongly positive articles found.")
    
    with tab3:
        negative_articles = sorted([a for a in sorted_articles if a['compound'] < -0.05], key=lambda x: x['compound'])[:10]
        if negative_articles:
            for article in negative_articles:
                st.markdown(f"""
                <div class="article-card sentiment-negative">
                    <h4 style='margin: 0; color: white;'>{article['title']}</h4>
                    <p style='color: #b0b0b0; margin: 8px 0;'>{article['description'][:200]}...</p>
                    <div style='font-size: 12px; color: #888;'>
                        <strong>Score:</strong> {article['compound']:.3f} | 
                        <strong>Source:</strong> {article['source']} | 
                        <a href="{article['url']}" target="_blank" style='color: #27EEF5;'>Read More →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strongly negative articles found.")

    st.divider()

    # ----------------------------
    # SUMMARY INSIGHTS
    # ----------------------------
    st.markdown("#### 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **📊 Analysis Summary:**
        - Analyzed **{len(articles_with_sentiment)}** articles from the past **{days} days**
        - Overall sentiment: **{sentiment_label}** ({avg_sentiment:.3f})
        - Positive coverage: **{positive_count}** articles ({positive_count/len(sentiments)*100:.1f}%)
        - Negative coverage: **{negative_count}** articles ({negative_count/len(sentiments)*100:.1f}%)
        """)
    
    with col2:
        if avg_sentiment > 0.1:
            insight = f"**{symbol.upper()}** is receiving mostly positive coverage. Investors appear optimistic."
            st.success(insight)
        elif avg_sentiment < -0.1:
            insight = f"**{symbol.upper()}** is facing negative sentiment. Market concerns may be present."
            st.error(insight)
        else:
            insight = f"**{symbol.upper()}** has mixed sentiment. Monitor for clearer trends."
            st.warning(insight)

else:
    st.info("👆 Enter a stock symbol above and click 'Analyze Sentiment' to get started!")
    
    # Show example/demo section
    st.markdown("""
    ### 🎯 What You'll Get:
    - **Sentiment Score**: Overall positive/negative/neutral rating
    - **Visual Charts**: Distribution graphs and sentiment gauge
    - **Article Breakdown**: Individual analysis of each news article
    - **Timeline View**: Sentiment trends over time
    - **Key Insights**: Summary of market sentiment
    
    ### 💡 Tips:
    - Try popular stocks like AAPL, TSLA, MSFT, GOOGL
    - Increase the time period for more comprehensive analysis
    - Review individual articles to understand sentiment drivers
    """)
