
from typing import List
from pyserini.search import SimpleSearcher, JSimpleSearcherResult
from transformers.data.processors.squad import SquadExample
from bertserini_on_telegram.utils.base import Context, Question


class Searcher:
    """A wrapper around pyserini SimpleSearcher.
    
    Args:
        index_name (str): The name of the prebuilt index.
        k1 (float): BM25 k1 parameter. Controls how quickly an increase in term frequency results in term-fequency saturation. Defaults to 0.9.
        b (float): BM25 b parameter. Controls how much effect field-length normalization should have. Defaults to 0.4.
        language (str): The language of the posed question. Defaults to "en".
    
    Attributes:
        searcher (SimpleSearcher): The searcher object.
        self.language: The language of all the texts seen by this class. Defaults to "en".

    """
    
    def __init__(self, index_name: str, k1=0.9, b=0.4, language="en"):
        super().__init__()
        
        self.searcher = SimpleSearcher.from_prebuilt_index(index_name)
        self.searcher.set_bm25(k1, b)
        self.searcher.object.setLanguage(language)
        
        self.language = language
    
    def hits_to_contexts(self, hits: List[JSimpleSearcherResult], field='raw', blacklist=[]) -> List[Context]:
        """Converts hits from Pyserini into a list of texts.
            
            Args:

                hits (List[JSimpleSearcherResult]): The list of results.
                field (str): Field to use for the Context. Defaults to "raw".
                blacklist (List[str]): strings that should not contained in the results.    
                        
            Returns:
                List[Context]: List of Context.
        """

        contexts = []

        for hit in hits:
            if field == 'raw':
                text = hit.raw
            else:
                text = hit.content
            
            for s in blacklist:
                if s in text:
                    continue
            
            metadata = {'raw': text, 'docid': hit.docid}
            contexts.append(Context(text, self.language, metadata, hit.score))
        return contexts
        
    
    def retrieve(self, question: Question, num_results: int = 20) -> List[Context]:
        """Retrieves contexts from prebuilt index given a question.

        Args:
            question (Question): The posed question.
            num_results (int, optional): Number of hits to consider. Defaults to 20.

        Returns:
            List[Context]: List of Context objects.
        """
        try:
            hits = self.searcher.search(question.text, k=num_results)
        except ValueError as e:
            print("Search failure: {}, {}".format(question.text, e))
            return []
        
        return self.hits_to_contexts(hits)
        

def craft_squad_examples(question: Question, contexts: List[Context]) -> List[SquadExample]:
    """Convert a Question with multiple Contexts into a list of SquadExample.

    Args:
        question (Question): The question to be answered.
        contexts (List[Context]): The list of contexts to look for an answer in.

    Returns:
        List[SquadExample]: List of SquadExample to be fed to a NLP model.
    """
    
    examples = []
    for idx, ctx in enumerate(contexts):
        examples.append(
            SquadExample(
                qas_id=idx,
                question_text=question.text,
                context_text=ctx.text,
                answer_text=None,
                start_position_character=None,
                title="",
                is_impossible=False,
                answers=[],
            )
        )
    return examples
