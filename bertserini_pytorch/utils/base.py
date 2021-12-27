from typing import List, Union, Optional, Mapping, Any
import abc

__all__ = ['Question', 'Context', 'Reader', 'Answer', 'TextType']


TextType = Union['Question', 'Context', 'Answer']


class Question:
    """
    A wrapper around a question text. Can contain other metadata.
    
    Args:
        text (str): The question text.
        id (Optional[str]): The question id. Defaults to None.
        language (str): The language of the posed question. Defaults to "en".
    """
    def __init__(self, text: str, id: Optional[str] = None, language: str = "en"):
        self.text = text
        self.id = id
        self.language = language


class Context:
    """
    A wrapper around a context text.
    The text is unspecified with respect to it length; in principle, it could be a full-length document, a paragraph-sized passage, or
    even a short phrase.

    Args:
        text (str): The context that contains potential answer.
        language (str): The language of the posed question. Defaults to "en".
        metadata (Mapping[str, Any]): Additional metadata and other annotations.
        score (Optional[float]): The score of the context. For example, the score might be the BM25 score from an initial retrieval stage. Defaults to None.
    """

    def __init__(self,
                 text: str,
                 language: str = "en",
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0):

        self.text = text
        self.language = language

        if metadata is None:
            metadata = dict()

        self.metadata = metadata
        self.score = score


class Answer:
    """    
    A wrapper around a question text. Can contain other metadata.

    Args:
    
        text (str): The answer text.
        language (str): The language of the posed question. Defaults to "en".
        metadata (Mapping[str, Any]): Additional metadata and other annotations.
        ans_score (Optional[float]): The score of the answer.
        ctx_score (Optional[float]): The context score of the answer.
        total_score (Optional[float]): The aggregated score of answer score and ctx_score.
    """
    def __init__(self,
                 text: str,
                 language: str = "en",
                 metadata: Mapping[str, Any] = None,
                 ans_score: Optional[float] = 0,
                 ctx_score: Optional[float] = 0,
                 total_score: Optional[float] = 0):

        self.text = text
        self.language = language
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.ans_score = ans_score
        self.ctx_score = ctx_score
        self.total_score = total_score

    def aggregate_score(self, weight: float) -> float:
        """
        Computes the aggregate score between ans_score and ctx_score as a linear interpolation given a weight.

        Args:
            weight (float): The weight to assign to ans_score and ctx_score.
        
        Returns:
            float: The aggregated score.
        """
        
        self.total_score = weight*self.ans_score + (1-weight)*self.ctx_score
