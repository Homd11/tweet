class TweetSentimentError(Exception):
    pass


class DataLoadingError(TweetSentimentError):
    pass


class PreprocessingError(TweetSentimentError):
    pass


class FeatureExtractionError(TweetSentimentError):
    pass


class ModelTrainingError(TweetSentimentError):
    pass


class ModelPredictionError(TweetSentimentError):
    pass


class DashboardError(TweetSentimentError):
    pass


class ConfigurationError(TweetSentimentError):
    pass