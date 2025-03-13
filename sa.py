import pandas as pd
import requests
import json
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, Legend, BasicTickFormatter, Label
from bokeh.palettes import Category20, Spectral11, Turbo256
from bokeh.transform import factor_cmap
from collections import Counter, defaultdict
import time
import re
import random

# Predefined sentiment categories with descriptions
SENTIMENT_CATEGORIES = {
    # Positive sentiments
    "very positive": ["excellent", "outstanding", "amazing", "perfect", "wonderful", "brilliant", "fantastic", "exceptional"],
    "positive": ["good", "nice", "great", "happy", "pleased", "satisfied", "enjoyable", "favorable"],
    "slightly positive": ["decent", "fine", "okay", "acceptable", "fair", "adequate", "satisfactory"],
    
    # Neutral sentiments
    "neutral": ["neutral", "balanced", "objective", "impartial", "moderate", "middle ground", "neither"],
    "factual": ["informative", "educational", "descriptive", "explanatory", "instructive"],
    "mixed": ["mixed", "ambivalent", "conflicted", "complicated", "nuanced", "both"],
    
    # Negative sentiments
    "slightly negative": ["mediocre", "subpar", "underwhelming", "disappointing", "unimpressive"],
    "negative": ["bad", "poor", "dislike", "unsatisfied", "unhappy", "unpleasant", "unfavorable"],
    "very negative": ["terrible", "awful", "horrible", "dreadful", "atrocious", "abysmal", "disgusting"],
    
    # Emotional categories
    "angry": ["angry", "furious", "outraged", "irritated", "mad", "frustrated", "annoyed"],
    "sad": ["sad", "depressed", "unhappy", "melancholy", "sorrowful", "heartbroken", "gloomy"],
    "anxious": ["anxious", "worried", "nervous", "concerned", "uneasy", "apprehensive", "fearful"],
    "surprised": ["surprised", "shocked", "astonished", "amazed", "startled", "stunned", "unexpected"],
    "confused": ["confused", "puzzled", "perplexed", "bewildered", "uncertain", "unclear", "ambiguous"],
    "amused": ["funny", "amusing", "humorous", "entertaining", "comical", "hilarious", "laughable"],
    "excited": ["excited", "enthusiastic", "eager", "thrilled", "energetic", "passionate", "exhilarated"],
    "bored": ["bored", "dull", "tedious", "monotonous", "uninteresting", "tiresome", "repetitive"],
    "inspired": ["inspired", "motivated", "uplifted", "encouraged", "stimulated", "moved", "influenced"],
    "thoughtful": ["thoughtful", "insightful", "reflective", "contemplative", "profound", "thought-provoking"]
}

def load_comments(file_path):
    """Load comments from CSV file."""
    df = pd.read_csv(file_path)
    # Keep only non-empty comments
    df = df[df['content'].notna()]
    # Convert likes column to numeric if it's not already
    if 'likes' in df.columns:
        df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
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
        # Extract just the three words
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

def map_to_predefined_category(sentiment_words):
    """Map the three-word sentiment to one of our predefined categories."""
    # Split the sentiment into individual words
    words = sentiment_words.lower().split()
    
    # Score for each category based on word matches
    category_scores = defaultdict(int)
    
    # Check each word against our category keywords
    for word in words:
        for category, keywords in SENTIMENT_CATEGORIES.items():
            if word in keywords:
                category_scores[category] += 1
            # Also check for partial matches (e.g. "positive" matching "positivity")
            else:
                for keyword in keywords:
                    if (word in keyword or keyword in word) and len(word) > 3:
                        category_scores[category] += 0.5
    
    # If we found matches, return the highest scoring category
    if category_scores:
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        return best_category
    
    # If no match is found, try to classify based on individual words
    for word in words:
        # Positive words
        if word in ["good", "great", "nice", "excellent", "positive", "love", "best"]:
            return "positive"
        # Negative words
        elif word in ["bad", "poor", "negative", "worst", "terrible", "awful", "hate"]:
            return "negative"
        # Neutral words
        elif word in ["neutral", "okay", "fair", "average", "moderate", "balanced"]:
            return "neutral"
    
    # Default to "neutral" if no classification was possible
    return "neutral"

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
        category = map_to_predefined_category(clean_sent)
        
        results.append({
            'comment': comment[:100] + ('...' if len(comment) > 100 else ''),
            'full_comment': comment,
            'raw_sentiment': clean_sent,
            'sentiment_category': category,
            'likes': int(row.get('likes', 0)),
            'publish_date': row.get('published', '')
        })
        
        # Be nice to the API with a small delay
        time.sleep(0.5)
    
    return results

def create_sentiment_visualizations(sentiment_data, output_html="sentiment_analysis.html"):
    """Create Bokeh visualizations for sentiment analysis with predefined categories."""
    # Convert to DataFrame
    df_results = pd.DataFrame(sentiment_data)
    
    # Count occurrences of each sentiment category
    category_counts = Counter(df_results['sentiment_category'])
    
    # Create DataFrames for plotting, ensuring all predefined categories are included
    all_categories = list(SENTIMENT_CATEGORIES.keys())
    categories_found = list(category_counts.keys())
    
    # Merge found categories with predefined ones
    all_found_categories = list(set(all_categories + categories_found))
    
    # Create DataFrame with all categories
    sentiment_df = pd.DataFrame({
        'sentiment': all_found_categories,
        'count': [category_counts.get(cat, 0) for cat in all_found_categories]
    }).sort_values('count', ascending=False)
    
    # Filter to only show categories with at least one occurrence
    sentiment_df = sentiment_df[sentiment_df['count'] > 0]
    
    # Group sentiments into categories for color coding
    sentiment_groups = {
        'positive': ['very positive', 'positive', 'slightly positive'],
        'neutral': ['neutral', 'factual', 'mixed'],
        'negative': ['slightly negative', 'negative', 'very negative'],
        'emotional': ['angry', 'sad', 'anxious', 'surprised', 'confused', 'amused', 'excited', 'bored', 'inspired', 'thoughtful']
    }
    
    # Create colors dictionary with color groups
    colors = {}
    for sentiment in sentiment_df['sentiment']:
        for group, members in sentiment_groups.items():
            if sentiment in members:
                if group == 'positive':
                    colors[sentiment] = Turbo256[int(220 - (members.index(sentiment) * 40))]
                elif group == 'neutral':
                    colors[sentiment] = Turbo256[int(150 - (members.index(sentiment) * 30))]
                elif group == 'negative':
                    colors[sentiment] = Turbo256[int(50 + (members.index(sentiment) * 30))]
                else:  # emotional categories
                    idx = list(SENTIMENT_CATEGORIES.keys()).index(sentiment) % len(Spectral11)
                    colors[sentiment] = Spectral11[idx]
                break
        if sentiment not in colors:
            # Assign a random color for any sentiment not in our groups
            colors[sentiment] = Turbo256[random.randint(0, 255)]
    
    # Create Bokeh output file
    output_file(output_html)
    
    # Create bar chart for sentiment distribution
    source = ColumnDataSource(sentiment_df)
    
    # Set figure options with better tooltip
    p = figure(
        x_range=sentiment_df['sentiment'].tolist(),
        height=500,
        width=900,
        title="Sentiment Distribution in Comments",
        toolbar_location="right",
        tools="hover,pan,wheel_zoom,box_zoom,reset,save",
        tooltips=[("Sentiment", "@sentiment"), ("Count", "@count"), ("Percentage", "@percentage%")]
    )
    
    # Calculate percentage
    sentiment_df['percentage'] = round((sentiment_df['count'] / sentiment_df['count'].sum()) * 100, 1)
    source = ColumnDataSource(sentiment_df)
    
    # Add color column to the source data
    sentiment_df['color'] = [colors[x] for x in sentiment_df['sentiment']]
    source = ColumnDataSource(sentiment_df)
    
    # Create the bars with colors from the data source
    p.vbar(
        x='sentiment',
        top='count',
        width=0.8,
        source=source,
        fill_color='color',
        line_color='white',
        legend_field='sentiment'
    )
    
    # Improve the appearance
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.major_label_orientation = 1.2
    p.legend.orientation = "vertical"
    p.legend.location = "top_right"
    p.legend.label_text_font_size = "8pt"
    
    # If there are more than 10 categories, hide the legend
    if len(sentiment_df) > 10:
        p.legend.visible = False
    
    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [
        ("Sentiment", "@sentiment"),
        ("Count", "@count"),
        ("Percentage", "@percentage%")
    ]
    p.add_tools(hover)
    
    # Add value labels on top of bars
    labels = Label(x=0, y=0, text="", text_font_size="8pt", text_color="black", text_align="center")
    p.add_layout(labels)
    
    # Save the figure
    save(p)
    
    print(f"Visualization saved to {output_html}")
    
    # Return summary for console output
    return {
        'total_comments': len(sentiment_data),
        'unique_sentiments': len(sentiment_df),
        'top_sentiments': dict(sentiment_df.iloc[:5][['sentiment', 'count']].set_index('sentiment')['count'])
    }

def main():
    print("YouTube Comments Sentiment Analysis")
    print("----------------------------------")
    
    # Load the CSV file
    file_path = "Outscraper-20250313003622xs18.csv"
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
    print(f"Unique sentiment categories: {summary['unique_sentiments']}")
    print("Top 5 sentiment categories:")
    for sentiment, count in summary['top_sentiments'].items():
        print(f"  - '{sentiment}': {count} comments")

if __name__ == "__main__":
    main()