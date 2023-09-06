# Sentiment Analysis Project

The task at hand is to have a model that can identify the general sentiment of a given query. This can be useful for understanding how our customers feel about a product or situation. The model should be able to take in user input and output the sentiment and confidence score.

In this presentation, I will walk you through my methodology, give a brief overview of the code, show a demo of the working model, present the test cases output with metrics such as accuracy, and suggest possible future improvements


## Approach to Problem

1. **Define the problem**: Our test cases originate from the Sentiment140 dataset, which is a multiclass dataset. We will use a hybrid lexicon approach to analyze the data {positive,neutral,negative}

2. **Model selection**: We will benchmark open source models based on their size, parameters, the dataset they were trained on, and the date they were trained.


| Model | Base Model | Description | Latest Update | Downloads last month | Language | Fine-tuned Dataset | Model Size | Parameters | Labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cardiffnlp/twitter-roberta-base-sentiment-latest | RoBERTa-base | RoBERTa-base model trained on Twitter 2021 ~124M (RoBERTa-base) fine-tuned on TweetEval benchmark (~124M tweets) | 2022 | 1,618,710 | English | tweet_eval ~66k tweets (train=45.6k,test=12.3k,val=2k) | 500MB | 125M | Negative, Neutral, Positive |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | XLM-RoBERTa-base | Multilingual XLM-RoBERTa-base model trained on ~198M tweets and fine-tuned for sentiment analysis | 2021 | 1,179,935 | Multilingual | 8 languages (Ar, En, Fr, De, Hi, It, Sp, Pt) tweets | 1.3GB | 270M | Negative, Neutral, Positive |
| cardiffnlp/twitter-roberta-base-sentiment | RoBERTa-base | RoBERTa-base model trained on Twitter ~58M (RoBERTa-base) fine-tuned on TweetEval benchmark (~58M tweets) | 2021 | 771,567 | English | tweet_eval ~66k tweets	| 500MB	| 125M	| LABEL_0, LABEL_1, LABEL_2 |
| finiteautomata/bertweet-base-sentiment-analysis	| BERTweet-base	| VinAIResearch/BERTweet model trained on Twitter 2012-2019 845M English Tweets and 5M COVID-19 Tweets	| 2023	| 230,948	| English	| SemEval 2017 corpus (around ~40k tweets)	| 530MB	| 135M	| NEG, NEU, POS |

* To download each of the models, you can use the following commands:
    * `git clone https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest`
    * `git clone https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment`
    * `git clone https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis`
    * `git clone https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment`


3. **Development**: 



    * **Script**: Load any model via inference 

        * `Input:` text 
        * `Output:` model output {sentiment, confidence_score}

            * [GitHub Repository](<URL>)


        * `RUN ON LOCAL SCRIRT` model_name = `cardiffnlp/twitter-xlm-roberta-base-sentiment`
        
        <img src="image-1.png" alt="LOCAL SCRIRT" width="600" height="600"/>

        * `RUN ON CLOUD APPLICATION` model_name = `cardiffnlp/twitter-xlm-roberta-base-sentiment-latest`
            
        
        <img src="image-7.png" alt="CLOUD INFERENCE" width="1200" height="800"/>
     
        
    * **Quick code walk through:** 
        

        | `sentiment_analyzer.py` | `test.py` |
        | --- | --- |
        | <img src="image-2.png" alt="SENTIMENT ANALYZER CLASS" width="600" height="600"/> | <img src="image-6.png" alt="TEST SCRIPT" width="600" height="600"/> |

        
    
    * **Notebook**: Download the model, run the model locally, save model_output in csv for all test cases (not sampled).

        * `Input:` sentiment_test_cases.csv
        * `Output:` model output in csv files for per model

            * [GitHub Repository](<URL>)

        
        * Also measure the computational efficiency of each model (Running on CPU)

        * `Computational Efficiency`

            | Model | Time (seconds) |
            | --- | --- |
            | twitter-roberta-base-sentiment-latest | 27.27 |
            | bertweet-base-sentiment-analysis | 27.52 |
            | twitter-xlm-roberta-base-sentiment | 30.28 |
            | twitter-roberta-base-sentiment | 29.30 |


    * **Script**: Load selected model.

        * Quick code walk through:

            * Input: sentiment_test_cases.csv
            * Output: output_.csv
            
            * [GitHub Repository](<URL>)

4. **Results and analysis**: Test dataset leaderboard (accuracy, weighted average f1_score, etc.)

    * **Web application**: Walk through of how to choose a model and by evaluating the result of model leaderboard against the test_cases. 
        
        * Input: `output_{model_name}.csv`
        * Output: summary of model performance in dataframe and visuals

        * `Evaluation Metrics` 
            * <img src="image-5.png" alt="METRICS" width="600" height="300"/>
            
        * `Weighted F1 Score Per Class` 
            * <img src="image-4.png" alt="F1 SCORE" width="1000" height="600"/>


            * [GitHub Repository](<https://github.com/rcruzin-ai/ai-task.git>)
            * [Streamlit Web App](<https://ai-task-rcruzin-ai.streamlit.app/>)


    * **Summary of Results**

        * Winner `twitter-roberta-base-sentiment-latest` 😊
            - The best model for the `sentiment_test_cases.csv` dataset. 
            - This model has the highest overall performance in terms of accuracy, precision, recall, and F1 score. 
            - It also has a good balance between computational efficiency and performance, with a relatively fast processing time and high F1 scores for all sentiment classes.

        * Additional insights:
            - The results suggest that `RoBERTa-base model` is well suited for sentiment analysis tasks, particularly when fine-tuned on a large dataset of English tweets.
            - The `twitter-xlm-roberta-base-sentiment` model is a multilingual model that supports 8 languages. However, since the `test_cases.csv` dataset only contains English tweets, this model may not have performed as well as other models that are specifically trained on English data.
            - The `bertweet-base-sentiment-analysis` model has a high F1 score for the negative class, indicating that it may be particularly good at identifying negative sentiment in tweets. 
            - The `twitter-roberta-base-sentiment-latest` and `bertweet-base-sentiment-analysis` models are both relatively popular, with a large number of downloads last month. This may indicate that they are widely used and well-regarded by the community.


5. **Possible future improvements**:

    * Insights from the result of model benchmarking
    * Fine-tuning the model (using base models such as roBERTa or BERTweet is an option)





6. (Optional) Additional slides/demo to showcase personal projects:
    * Chat AI application [own pdf documents, open ai, embeddings, vector database, langchain]
    * A fine-tuned model (if I have extra time)