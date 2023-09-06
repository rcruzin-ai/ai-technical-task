
# test.py

# Import the SentimentAnalyzer class from the sentiment_analyzer module
from sentiment_analyzer import SentimentAnalyzer

if __name__ == "__main__":

    # Create a SentimentAnalyzer object
    analyzer = SentimentAnalyzer()

    # Get user input
    user_input = input("Input: ")

    # Analyze sentiment and display result
    result = analyzer.analyze_text(user_input, "cardiffnlp/twitter-roberta-base-sentiment-latest")

    if result:
        # Sort the results by confidence score in descending order
        result.sort(key=lambda x: x["score"], reverse=True)
        
        # Select the sentiment with the highest confidence score
        sentiment = result[0]["label"]
        confidence = result[0]["score"]
        
        output = {"model_output": sentiment, "confidence_score": confidence}
        print(f"Output: {output}")
