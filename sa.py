import pandas as pd
import requests
import json
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, Legend
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
from collections import Counter
import time
import re

def load_comments(file_path):
    """Load comments from CSV file."""
    df = pd.read_csv(file_path)
    # Keep only non-empty comments
    df = df[df['content'].notna()]
    return df

def get_sentiment(text, ollama_url="http://localhost:11434/api/generate"):
    """Get sentiment from Ollama API."""
    prompt = f"do only three word sentiment analysis to this sentence without any explanation. pure three words\n\n{text}"
    
    payload = {
        "model": "phi4-mini:latest",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        result = response.json()
        # Extract just the three words (assuming they're the first three words in the response)
        sentiment_words = result['response'].strip().split()[:3]
        return " ".join(sentiment_words)
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return "API error"

def clean_sentiment(sentiment):
    """Clean up sentiment text to standardize responses."""
    # Convert to lowercase
    sentiment = sentiment.lower()
    
    # Remove punctuation and normalize whitespace
    sentiment = re.sub(r'[^\w\s]', '', sentiment)
    sentiment = re.sub(r'\s+', ' ', sentiment).strip()
    
    return sentiment

def analyze_comments(df, sample_size=None):
    """Analyze sentiment of comments."""
    if sample_size and sample_size < len(df):
        df_sample = df.sample(sample_size)
    else:
        df_sample = df
    
    results = []
    
    for i, row in df_sample.iterrows():
        comment = row['content']
        print(f"Analyzing comment {i+1}/{len(df_sample)}: {comment[:50]}...")
        
        sentiment = get_sentiment(comment)
        clean_sent = clean_sentiment(sentiment)
        
        results.append({
            'comment': comment,
            'sentiment': clean_sent,
            'likes': row.get('likes', 0)
        })
        
        # Be nice to the API with a small delay
        time.sleep(0.5)
    
    return results

def create_sentiment_visualizations(sentiment_data, output_html="sentiment_analysis.html"):
    """Create Bokeh visualizations for sentiment analysis."""
    # Convert to DataFrame
    df_results = pd.DataFrame(sentiment_data)
    
    # Count occurrences of each sentiment
    sentiment_counts = Counter(df_results['sentiment'])
    top_sentiments = dict(sentiment_counts.most_common(10))
    
    # Create DataFrames for plotting
    sentiment_df = pd.DataFrame({
        'sentiment': list(top_sentiments.keys()),
        'count': list(top_sentiments.values())
    }).sort_values('count', ascending=False)
    
    # Create Bokeh output file
    output_file(output_html)
    
    # Create bar chart for sentiment distribution
    source = ColumnDataSource(sentiment_df)
    
    p1 = figure(
        x_range=sentiment_df['sentiment'].tolist(),
        height=400,
        width=800,
        title="Top 10 Sentiment Distributions",
        toolbar_location=None,
        tools="hover",
        tooltips=[("Sentiment", "@sentiment"), ("Count", "@count")]
    )
    
    p1.vbar(
        x='sentiment',
        top='count',
        width=0.9,
        source=source,
        fill_color=factor_cmap('sentiment', palette=Category10[10], factors=sentiment_df['sentiment'].tolist()),
        line_color='white'
    )
    
    p1.xgrid.grid_line_color = None
    p1.y_range.start = 0
    p1.xaxis.major_label_orientation = 1.2
    
    # Create HTML file with all plots
    save(p1)
    
    print(f"Visualization saved to {output_html}")
    
    # Return summary for console output
    return {
        'total_comments': len(sentiment_data),
        'unique_sentiments': len(sentiment_counts),
        'top_sentiments': dict(sentiment_counts.most_common(5))
    }

def main():
    print("YouTube Comments Sentiment Analysis")
    print("----------------------------------")
    
    # Load the CSV file
    file_path = "Outscraper20250313003622xs18.csv"
    print(f"Loading comments from {file_path}...")
    df = load_comments(file_path)
    print(f"Loaded {len(df)} comments.")
    
    # Ask user for sample size
    sample_size = int(input("How many comments to analyze? (Enter a number or 0 for all): "))
    if sample_size == 0:
        sample_size = None
    
    # Analyze comments
    print("\nAnalyzing comments...")
    results = analyze_comments(df, sample_size)
    print("Analysis complete!")
    
    # Create visualizations
    print("\nCreating visualizations...")
    summary = create_sentiment_visualizations(results)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total comments analyzed: {summary['total_comments']}")
    print(f"Unique sentiment patterns: {summary['unique_sentiments']}")
    print("Top 5 sentiments:")
    for sentiment, count in summary['top_sentiments'].items():
        print(f"  - '{sentiment}': {count} comments")

if __name__ == "__main__":
    main()