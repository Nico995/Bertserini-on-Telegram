
from typing import List
from pyserini.search import SimpleSearcher, JSimpleSearcherResult
from transformers.data.processors.squad import SquadExample
from bertserini_pytorch.utils.base import Context, Question


def build_searcher(index_name, k1=0.9, b=0.4, language="en"):
    searcher = SimpleSearcher.from_prebuilt_index(index_name)
    searcher.set_bm25(k1, b)
    searcher.object.setLanguage(language)
    return searcher


def retriever(question: Question, searcher: SimpleSearcher, para_num=20):
    language = question.language
    try:
        hits = searcher.search(question.text, k=para_num)
    except ValueError as e:
        print("Search failure: {}, {}".format(question.text, e))
        return []
    return hits_to_contexts(hits, language)


def hits_to_contexts(hits: List[JSimpleSearcherResult], language="en", field='raw', blacklist=[]) -> List[Context]:
    """
        Converts hits from Pyserini into a list of texts.
        Parameters
        ----------
        hits : List[JSimpleSearcherResult]
            The hits.
        field : str
            Field to use.
        language : str
            Language of corpus
        blacklist : List[str]
            strings that should not contained
        Returns
        -------
        List[Text]
            List of texts.
     """
    contexts = []
    for i in range(0, len(hits)):
        t = hits[i].raw if field == 'raw' else hits[i].contents
        for s in blacklist:
            if s in t:
                continue
        metadata = {'raw': hits[i].raw, 'docid': hits[i].docid}
        contexts.append(Context(t, language, metadata, hits[i].score))
    return contexts


def get_best_answer(candidates, weight=0.5):
    for ans in candidates:
        ans.aggregate_score(weight)
    return sorted(candidates, key=lambda x: x.total_score, reverse=True)[0]


def craft_squad_examples(question: Question, contexts: List[Context]) -> List[SquadExample]:
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
