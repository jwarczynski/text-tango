import logging
import evaluate
import numpy as np

from collections import defaultdict, namedtuple
from datasets import load_dataset
from text_preprocessing import normalize


logger = logging.getLogger(__name__)

RDFTriple = namedtuple("RDFTriple", ["subj", "pred", "obj"])


class DataEntry:
    """
    An entry in the dataset
    """

    def __init__(self, data, refs, data_type, align=None, num_ref_sentences=None, category=None, dialhist=None):
        self.data = data
        self.refs = refs
        self.data_type = data_type
        self.align = align
        self.num_ref_sentences = num_ref_sentences
        self.category = category
        self.dialhist = dialhist

    def __repr__(self):
        return str(self.__dict__)


class WebNLG:
    """
    The WebNLG dataset: https://gem-benchmark.com/data_cards/web_nlg
    Contains RDF triples from DBPedia and their crowdsourced verbalizations.
    """

    name = "webnlg"

    def __init__(self, *args, **kwargs):
        self.data = []

    def load(self, splits, path=None):
        # load the dataset from HF datasets
        dataset = load_dataset("gem", "web_nlg_en")

        for split in splits:
            data = dataset[split if split != "dev" else "validation"]

            for example in data:
                triples = example["input"]
                triples = [t.split("|") for t in triples]
                triples = [(normalize(x, remove_parentheses=False) for x in t) for t in triples]
                triples = [RDFTriple(*t) for t in triples]

                if split == "test":
                    refs = example["references"]
                else:
                    refs = [example["target"]]

                entry = DataEntry(data=triples, refs=refs, data_type="triples", category=example["category"])
                self.data.append(entry)


# METRICS ==============================

class SingleReferenceMetric:
    def __init__(self) -> None:
        self.name = "SingleReferenceMetric"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        pass

    def compute(self, preds, refs, ref_lens, is_out_domain):
        results = self.eval(preds, refs, ref_lens, is_out_domain)

        i = 0
        merged_results = []
        for len_r in ref_lens:
            merged_results.append(results[i:i + len_r].mean())
            i += len_r

        results = np.array(merged_results)
        print(
            f"{self.name}: {results.mean()} +- {results.std()}; OOD: {results[is_out_domain].mean()}; InD: {results[~is_out_domain].mean()}")


class MultiReferenceMetric:
    def __init__(self) -> None:
        self.name = "MultiReferenceMetric"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        pass

    def compute(self, preds, refs, ref_lens, is_out_domain):
        results = self.eval(preds, refs, ref_lens, is_out_domain)
        print(f"{self.name}: {results} ")


class BLEURT(SingleReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bleurt", module_type="metric")
        self.name = "BLEURT"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs)
        return np.array(results["scores"])


class BERTScore(SingleReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bertscore")
        self.name = "BERTScore"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs, lang="en")
        return np.array(results["f1"])


class BLEU(MultiReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bleu")
        self.name = "BLEU"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs)
        return np.array(results["bleu"])


class METEOR(MultiReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("meteor")
        self.name = "METEOR"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs)
        return np.array(results["meteor"])


# EVAL ==============================

def get_basic_metrics():
    return [METEOR(), BLEU()]


def evaluate_program(program, metrics):
    data = WebNLG()
    data.load(['test'])

    refs_single = []
    preds_single = []
    refs_multi = []
    preds_multi = []
    ref_lens = []
    is_out_domain = []
    for dataEntry in data.data:
        is_out_domain.append(dataEntry.category in ["Film", "MusicalWork", "Scientist"])

        relations = tuple(sorted([i.pred for i in dataEntry.data]))
        input = [tuple([triplet.subj, triplet.pred, triplet.obj]) for triplet in dataEntry.data]
        output = program.process_input(relations, input)

        refs_multi.append(dataEntry.refs)
        preds_multi.append(output)
        for reference_text in dataEntry.refs:
            refs_single.append(reference_text)
            preds_single.append(output)
        ref_lens.append(len(dataEntry.refs))
    is_out_domain = np.array(is_out_domain)

    for metric in metrics:
        if isinstance(metric, MultiReferenceMetric):
            metric.compute(preds_multi, refs_multi, ref_lens, is_out_domain)
        else:
            metric.compute(preds_single, refs_single, ref_lens, is_out_domain)
