# YouTube Comment Sentiment Analysis

This project analyzes the sentiment of YouTube comments using a local Large Language Model (LLM) served by [Ollama](https://ollama.com/). It retrieves comments from a CSV file, sends them to a locally running Ollama instance for sentiment analysis, and visualizes the results.

## Project Structure

- `sa.py`: The main Python script containing the sentiment analysis logic.
- `requirements.txt`: Lists the Python dependencies for the project.
- `Outscraper-20250313003622xs18.csv`: An example CSV file containing YouTube comments (likely scraped using a tool like Outscraper).
- `Outscraper-20250313003622xs18.csv.html`: An HTML version of the CSV.
- `Outscraper-20250313003622xs18.xlsx`: An Excel version of the CSV.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore (in this case, the `venv` virtual environment directory).
- `sentiment_analysis.html`: The output HTML file containing the Bokeh visualization of the sentiment analysis results.

## Setup

1.  **Install Ollama:** Download and install Ollama from [https://ollama.com/](https://ollama.com/).

2.  **Pull a Model:** After installing Ollama, you'll need to pull a model. This project was developed and tested with `phi4-mini`. You can pull it by running the following command in your terminal:

    ```bash
    ollama pull phi4-mini:latest
    ```

3.  **Install Python Dependencies:** It is highly recommended to create a Python virtual environment before installing dependencies. You can create one using `python3 -m venv venv` and activate it with `source venv/bin/activate` (on Linux/macOS) or `venv\Scripts\activate` (on Windows). Then, install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

4. **Obtain YouTube Comments:** This project expects a CSV file containing YouTube comments. The example file `Outscraper-20250313003622xs18.csv` suggests that a tool like [Outscraper](https://outscraper.com/) was used to scrape comments. You can use a similar tool or any method to obtain a CSV file with a 'content' column containing the comment text and optionally a 'likes' column.

## Usage

1.  **Run the Script:** Execute the `sa.py` script:

    ```bash
    python sa.py
    ```

2.  **Input:** The script will prompt you to enter the number of comments to analyze. Enter a number or `0` to analyze all comments in the specified CSV file (`Outscraper-20250313003622xs18.csv`).

3.  **Output:** The script will:
    *   Print progress messages to the console.
    *   Generate an HTML file named `sentiment_analysis.html` containing interactive visualizations of the sentiment analysis results (using Bokeh).
    *   Print a summary of the analysis to the console, including the total comments analyzed, unique sentiment patterns, and the top 5 sentiments.

## Dependencies

*   **Ollama:** For running the local LLM.
*   **Python Packages:**
    *   `bokeh`: For creating interactive visualizations.
    *   `pandas`: For data manipulation and analysis.
    *   `Requests`: For making HTTP requests to the Ollama API.

## Visualization

The script generates an interactive Bokeh visualization (`sentiment_analysis.html`) showing the distribution of the top 10 sentiments found in the analyzed comments. The visualization includes a hover tool to display the sentiment and count for each bar.
