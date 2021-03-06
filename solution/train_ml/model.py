from typing import Dict, Optional, Tuple, Any, List
import attr
import logging
import copy

from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.training.metrics import AttachmentScores, CategoricalAccuracy
from train_ml.categorical_accuracy_my import CategoricalAccuracyMy

from train_ml.lemmatize_helper import LemmatizeHelper

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {'``', "''", ':', ',', '.', 'PU', 'PUNCT', 'SYM'}


@attr.s
class TaskConfig(object):
    task_type = attr.ib(default='multitask', validator=attr.validators.in_(['single', 'multitask']))
    params = attr.ib(factory=dict)


class WeightDrop(torch.nn.Module):
    def __init__(self, module, dropout, layer_names=['weight_hh_l0']):
        super().__init__()

        self.module, self.dropout, self.layer_names = module, dropout, layer_names
        for layer in self.layer_names:
            w = getattr(self.module, layer)
            self.register_parameter('{}_raw'.format(layer), torch.nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.dropout, training=False)

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, '{}_raw'.format(layer))
            self.module._parameters[layer] = F.dropout(raw_w, p=self.dropout, training=self.training)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, '{}_raw'.format(layer))
            self.module._parameters[layer] = F.dropout(raw_w, p=self.dropout, training=False)
        if hasattr(self.module, 'reset'):
            self.module.reset()


class LstmWeightDrop(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        variational_dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        rnns = []
        rnns.append(torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=1,
            bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional
        ))

        input_size = hidden_size * (2 if bidirectional else 1)
        for _ in range(num_layers - 1):
            rnns.append(torch.nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=1,
                bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional
            ))

        rnns = [WeightDrop(rnn, dropout=variational_dropout) for rnn in rnns]
        self._rnns = torch.nn.ModuleList(rnns)

    def forward(self, inputs, hidden_state):
        outputs = inputs
        hidden_states = []
        for rnn in self._rnns:
            outputs, hidden_state = rnn(outputs)
            hidden_states.append(hidden_state)

        h_n = torch.cat([h for h, c in hidden_states], dim=0)
        c_n = torch.cat([c for h, c in hidden_states], dim=0)

        return outputs, (h_n, c_n)


@Seq2SeqEncoder.register("lstm_weight_drop")
class LstmWeightDropSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        variational_dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False,
    ):
        module = LstmWeightDrop(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            variational_dropout=variational_dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module, stateful=stateful)


@Model.register("e2e_parser")
class DependencyParser(Model):
    """
    This dependency parser follows the model of
    ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ .

    Word representations are generated using a bidirectional LSTM,
    followed by separate biaffine classifiers for pairs of words,
    predicting whether a directed arc exists between the two words
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimal Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : ``int``, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for head arc prediction.
    tag_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 tag_representation_dim: int,
                 arc_representation_dim: int,
                 lemmatize_helper: LemmatizeHelper,
                 task_config: TaskConfig,
                 morpho_vector_dim: int = 0,
                 gram_val_representation_dim: int = -1,
                 lemma_representation_dim: int = -1,
                 tag_feedforward: FeedForward = None,
                 arc_feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 use_mst_decoding_for_validation: bool = True,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DependencyParser, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.lemmatize_helper = lemmatize_helper
        self.task_config = task_config

        encoder_dim = encoder.get_output_dim()

        self.head_arc_feedforward = arc_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    arc_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(arc_representation_dim,
                                                     arc_representation_dim,
                                                     use_input_biases=True)

        num_labels = self.vocab.get_vocab_size("head_tags")

        self.head_tag_feedforward = tag_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    tag_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(tag_representation_dim,
                                                      tag_representation_dim,
                                                      num_labels)

        self._pos_tag_embedding = pos_tag_embedding or None
        assert self.task_config.params.get("use_pos_tag", False) == (self._pos_tag_embedding is not None)

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder.get_output_dim()]))

        # рекуррентная сеть, порождающая цепочку вариантов разбора
        self.multilabeler_lstm = torch.nn.LSTM(encoder_dim, encoder_dim, num_layers=1, batch_first=True, bidirectional=False)

        if gram_val_representation_dim <= 0:
            self._gram_val_output = torch.nn.Linear(encoder_dim, self.vocab.get_vocab_size("grammar_value_tags"))
        else:
            self._gram_val_output = torch.nn.Sequential(
                Dropout(dropout),
                torch.nn.Linear(encoder_dim, gram_val_representation_dim),
                Dropout(dropout),
                torch.nn.Linear(gram_val_representation_dim, self.vocab.get_vocab_size("grammar_value_tags"))
            )

        if lemma_representation_dim <= 0:
            self._lemma_output = torch.nn.Linear(encoder_dim, len(lemmatize_helper))
        else:
            self._lemma_output = torch.nn.Sequential(
                Dropout(dropout),
                torch.nn.Linear(encoder_dim, lemma_representation_dim),
                Dropout(dropout),
                torch.nn.Linear(lemma_representation_dim, len(lemmatize_helper))
            )

        representation_dim = text_field_embedder.get_output_dim() + morpho_vector_dim
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        check_dimensions_match(tag_representation_dim, self.head_tag_feedforward.get_output_dim(),
                               "tag representation dim", "tag feedforward output dim")
        check_dimensions_match(arc_representation_dim, self.head_arc_feedforward.get_output_dim(),
                               "arc representation dim", "arc feedforward output dim")

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        tags = self.vocab.get_token_to_index_vocabulary("pos")
        punctuation_tag_indices = {tag: index for tag, index in tags.items() if tag in POS_TO_IGNORE}
        self._pos_to_ignore = set(punctuation_tag_indices.values())
        logger.info(f"Found POS tags corresponding to the following punctuation : {punctuation_tag_indices}. "
                    "Ignoring words with these POS tags for evaluation.")

        self._attachment_scores = AttachmentScores()
        self._gram_val_prediction_accuracy = CategoricalAccuracy()
        self._lemma_prediction_accuracy = CategoricalAccuracy()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                morpho_embedding: torch.FloatTensor = None,
                pos_tags: torch.LongTensor = None,
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None,
                grammar_values: torch.LongTensor = None,
                lemma_indices: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, sequence_length)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pos_tags : ``torch.LongTensor``, required
            The output of a ``SequenceLabelField`` containing POS tags.
            POS tags are required regardless of whether they are used in the model,
            because they are used to filter the evaluation metric to only consider
            heads of words which are not punctuation.
        metadata : List[Dict[str, Any]], optional (default=None)
            A dictionary of metadata for each batch element which has keys:
                words : ``List[str]``, required.
                    The tokens in the original sentence.
                pos : ``List[str]``, required.
                    The dependencies POS tags for each word.
        head_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape ``(batch_size, sequence_length)``.
        head_indices : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length)``.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        arc_loss : ``torch.FloatTensor``
            The loss contribution from the unlabeled arcs.
        loss : ``torch.FloatTensor``, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : ``torch.FloatTensor``
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : ``torch.FloatTensor``
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.
        """
        embedded_text_input = self.text_field_embedder(words)
        
        # для отладки
#        for name, param in self.named_parameters():
#            if torch.any(torch.isnan(param)):
#                assert False, "NaN in {} layer".format(name) 
#            if torch.any(torch.isinf(param)):
#                assert False, "INF in {} layer".format(name) 

        if morpho_embedding is not None:
            embedded_text_input = torch.cat([embedded_text_input, morpho_embedding], -1)

        if grammar_values is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(grammar_values)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(words)

        output_dict = self._parse(embedded_text_input, mask, head_tags, head_indices, grammar_values, lemma_indices)

        if self.task_config.task_type == "multitask":
            losses = ["arc_nll", "tag_nll", "grammar_nll", "lemma_nll"]
        elif self.task_config.task_type == "single":
            if self.task_config.params["model"] == "morphology":
                losses = ["grammar_nll"]
            elif self.task_config.params["model"] == "lemmatization":
                losses = ["lemma_nll"]
            elif self.task_config.params["model"] == "syntax":
                losses = ["arc_nll", "tag_nll"]
            else:
                assert False, "Unknown model type {}".format(self.task_config.params["model"])
        else:
            assert False, "Unknown task type {}".format(self.task_config.task_type)

        output_dict["loss"] = sum(output_dict[loss_name] for loss_name in losses)

        if head_indices is not None and head_tags is not None:
            evaluation_mask = self._get_mask_for_eval(mask, pos_tags)
            # We calculate attatchment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(output_dict["heads"][:, 1:],
                                    output_dict["head_tags"][:, 1:],
                                    head_indices,
                                    head_tags,
                                    evaluation_mask)

        output_dict["words"] = [meta["words"] for meta in metadata]
        if metadata and "pos" in metadata[0]:
            output_dict["pos"] = [meta["pos"] for meta in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        head_tags = output_dict.pop("head_tags").cpu().detach().numpy()
        heads = output_dict.pop("heads").cpu().detach().numpy()
        predicted_gram_vals = output_dict.pop("gram_vals").cpu().detach().numpy()
        predicted_lemmas = output_dict.pop("lemmas").cpu().detach().numpy()
        mask = output_dict.pop("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)

        assert len(head_tags) == len(heads) == len(lengths) == len(predicted_gram_vals) == len(predicted_lemmas)

        head_tag_labels, head_indices, decoded_gram_vals, decoded_lemmas = [], [], [], []
        for instance_index in range(len(head_tags)):
            instance_heads, instance_tags = heads[instance_index], head_tags[instance_index]
            words, length = output_dict["words"][instance_index], lengths[instance_index]
            gram_vals, lemmas = predicted_gram_vals[instance_index], predicted_lemmas[instance_index]

            words = words[: length.item() - 1]
            gram_vals = gram_vals[: length.item() - 1, :]
            lemmas = lemmas[: length.item() - 1, :]

            instance_heads = list(instance_heads[1:length])
            instance_tags = instance_tags[1:length]
            labels = [self.vocab.get_token_from_index(label, "head_tags") for label in instance_tags]
            head_tag_labels.append(labels)
            head_indices.append(instance_heads)

            inst_gram_vals = []
            for tok_gram_vals in gram_vals:
                dtgv = [self.vocab.get_token_from_index(gram_val, "grammar_value_tags") for gram_val in tok_gram_vals]
                inst_gram_vals.append(dtgv)
            decoded_gram_vals.append(inst_gram_vals)
#             print("\n\n------------------------------------------------- ITLOG-BEGIN ------------------------------------------\n")
#             print( "ITLOG: decoded_gram_vals = {}".format(decoded_gram_vals) )
#             print("\n------------------------------------------------- ITLOG-END ------------------------------------------\n")            

            inst_lemmas = []
            for word, word_lrules in zip(words, lemmas):
                dtl = [self.lemmatize_helper.lemmatize(word, lrule) for lrule in word_lrules]
                inst_lemmas.append(dtl)
            decoded_lemmas.append(inst_lemmas)

        if self.task_config.task_type == "multitask":
            output_dict["predicted_dependencies"] = head_tag_labels
            output_dict["predicted_heads"] = head_indices
            output_dict["predicted_gram_vals"] = decoded_gram_vals
            output_dict["predicted_lemmas"] = decoded_lemmas
        elif self.task_config.task_type == "single":
            if self.task_config.params["model"] == "morphology":
                output_dict["predicted_gram_vals"] = decoded_gram_vals
            elif self.task_config.params["model"] == "lemmatization":
                output_dict["predicted_lemmas"] = decoded_lemmas
            elif self.task_config.params["model"] == "syntax":
                output_dict["predicted_dependencies"] = head_tag_labels
                output_dict["predicted_heads"] = head_indices
            else:
                assert False, "Unknown model type {}".format(self.task_config.params["model"])
        else:
            assert False, "Unknown task type {}".format(self.task_config.task_type)

        return output_dict

    def _parse(self,
               embedded_text_input: torch.Tensor,
               mask: torch.LongTensor,
               head_tags: torch.LongTensor = None,
               head_indices: torch.LongTensor = None,
               grammar_values: torch.LongTensor = None,
               lemma_indices: torch.LongTensor = None):

        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        # добавим измеремение, которое каждому выходу энкодера ставит в соответствие три его копии
        encoded_text_3 = encoded_text
        encoded_text_3 = torch.unsqueeze(encoded_text_3, 2)
        encoded_text_3 = encoded_text_3.repeat(1,1,3,1)
        # пропустим три копии вектора (с выхода энкодера) через lstm
        seq_len = encoded_text.size()[1]
        emb_div_val = encoded_text.size()[2]
        multi_triplets = torch.reshape(encoded_text_3, (-1, 3, emb_div_val))
        label_variants, _ = self.multilabeler_lstm(multi_triplets)
        batched_label_variants = torch.reshape(label_variants, (-1, seq_len, 3, emb_div_val))
#         # отладочный вывод
#         print("\n\n------------------------------------------------- ITLOG-BEGIN ------------------------------------------\n")
#         print( "ITLOG: encoded_text.size() = {}".format(encoded_text.size()) )
#         print( "ITLOG: encoded_text_3.size() = {}".format(encoded_text_3.size()) )
#         print( "ITLOG: multi_triplets.size() = {}".format(multi_triplets.size()) )
#         print( "ITLOG: label_variants.size() = {}".format(label_variants.size()) )
#         print( "ITLOG: batched_label_variants.size() = {}".format(batched_label_variants.size()) )
#         print("\n------------------------------------------------- ITLOG-END ------------------------------------------\n")

#        grammar_value_logits = self._gram_val_output(encoded_text)
        grammar_value_logits = self._gram_val_output(batched_label_variants)
#         print("\n\n------------------------------------------------- ITLOG-BEGIN ------------------------------------------\n")
#         print( "ITLOG: grammar_value_logits.size() = {}".format(grammar_value_logits.size()) )
#         print("\n------------------------------------------------- ITLOG-END ------------------------------------------\n")
#        grammar_value_logits = grammar_value_logits.select(2, 0)
        predicted_gram_vals = grammar_value_logits.argmax(-1)

#        lemma_logits = self._lemma_output(encoded_text)
        lemma_logits = self._lemma_output(batched_label_variants)
        predicted_lemmas = lemma_logits.argmax(-1)

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        token_mask = mask.float()
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)
        float_mask = mask.float()
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation,
                                           child_arc_representation)

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(head_tag_representation,
                                                                       child_tag_representation,
                                                                       attended_arcs,
                                                                       mask)
        else:
            predicted_heads, predicted_head_tags = self._mst_decode(head_tag_representation,
                                                                    child_tag_representation,
                                                                    attended_arcs,
                                                                    mask)
        if head_indices is not None and head_tags is not None:

            arc_nll, tag_nll = self._construct_loss(head_tag_representation=head_tag_representation,
                                                    child_tag_representation=child_tag_representation,
                                                    attended_arcs=attended_arcs,
                                                    head_indices=head_indices,
                                                    head_tags=head_tags,
                                                    mask=mask)
        else:
            arc_nll, tag_nll = self._construct_loss(head_tag_representation=head_tag_representation,
                                                    child_tag_representation=child_tag_representation,
                                                    attended_arcs=attended_arcs,
                                                    head_indices=predicted_heads.long(),
                                                    head_tags=predicted_head_tags.long(),
                                                    mask=mask)

        grammar_nll = torch.tensor(0.)
        if grammar_values is not None:
            token_mask_3 = token_mask
            token_mask_3 = torch.unsqueeze(token_mask_3, 2)
            token_mask_3 = token_mask_3.repeat(1,1,3)            
#             print("\n\n------------------------------------------------- ITLOG-BEGIN ------------------------------------------\n")
#             print( "ITLOG: token_mask.size = {}".format(token_mask.size()) )
#             print( "ITLOG: token_mask_3.size = {}".format(token_mask_3.size()) )
#             print( "ITLOG: token_mask_3 = {}".format(token_mask_3) )
#             print("\n------------------------------------------------- ITLOG-END ------------------------------------------\n")            
            grammar_nll = self._update_multiclass_prediction_metrics_3(
                logits=grammar_value_logits, targets=grammar_values,
                mask=token_mask_3, accuracy_metric=self._gram_val_prediction_accuracy
            )

        lemma_nll = torch.tensor(0.)
        if lemma_indices is not None:
            token_mask_3 = token_mask
            token_mask_3 = torch.unsqueeze(token_mask_3, 2)
            token_mask_3 = token_mask_3.repeat(1,1,3)            
            lemma_nll = self._update_multiclass_prediction_metrics_3(
                logits=lemma_logits, targets=lemma_indices,
                mask=token_mask_3, accuracy_metric=self._lemma_prediction_accuracy #, masked_index=self.lemmatize_helper.UNKNOWN_RULE_INDEX
            )

        output_dict = {
            "heads": predicted_heads,
            "head_tags": predicted_head_tags,
            "gram_vals": predicted_gram_vals,
            "lemmas": predicted_lemmas,
            "mask": mask,
            "arc_nll": arc_nll,
            "tag_nll": tag_nll,
            "grammar_nll": grammar_nll,
            "lemma_nll": lemma_nll,
        }

        return output_dict

    @staticmethod
    def _update_multiclass_prediction_metrics(logits, targets, mask, accuracy_metric, masked_index=None):
#         print("\n\n------------------------------------------------- ITLOG-BEGIN ------------------------------------------\n")
#         print( "ITLOG: logits.size() = {}".format(logits.size()) )
#         print( "ITLOG: targets.size() = {}".format(targets.size()) )
#         print( "ITLOG: mask = {}".format(mask) )
#         print( "ITLOG: targets = {}".format(targets) )
#         print("\n------------------------------------------------- ITLOG-END ------------------------------------------\n")

        accuracy_metric(logits, targets, mask)

        logits = logits.view(-1, logits.shape[-1])
        loss = F.cross_entropy(logits, targets.view(-1), reduction='none')
        if masked_index is not None:
            mask = mask * (targets != masked_index)
        loss_mask = mask.view(-1)
        return (loss * loss_mask).sum() / loss_mask.sum()

    @staticmethod
    def _update_multiclass_prediction_metrics_3(logits, targets, mask, accuracy_metric, masked_index=None):
#         print("\n\n------------------------------------------------- ITLOG-BEGIN ------------------------------------------\n")
#         print( "ITLOG: logits.size() = {}".format(logits.size()) )
#         print( "ITLOG: targets.size() = {}".format(targets.size()) )
#         print( "ITLOG: mask = {}".format(mask) )
#         print( "ITLOG: targets = {}".format(targets) )
#         print("\n------------------------------------------------- ITLOG-END ------------------------------------------\n")

        accuracy_metric(logits, targets, mask)

        # будем вычислять cross_entropy только для незамаскированных элементов тензора
#         non_masked_coords = torch.nonzero(mask)
#         print( "ITLOG: non_masked_coords = {}".format(non_masked_coords) )
        bmask = torch.unsqueeze(mask, -1)
        logits_m = torch.masked_select(logits, bmask.bool())
        logits_m = torch.reshape(logits_m, (-1, logits.shape[-1]))
#         print( "ITLOG: logits_m.size() = {}".format(logits_m.size()) )
#         print( "ITLOG: logits_m = {}".format(logits_m) )
        targets_m = torch.masked_select(targets, mask.bool())
#         print( "ITLOG: targets_m.size() = {}".format(targets_m.size()) )
#         print( "ITLOG: targets_m = {}".format(targets_m) )
        loss = F.cross_entropy(logits_m, targets_m, reduction='none')
#         if masked_index is not None:
#             mask = mask * (targets != masked_index)
#         loss_mask = mask.view(-1)
#         return (loss * loss_mask).sum() / loss_mask.sum()
#         print( "ITLOG: loss.size() = {}".format(loss.size()) )
#         print( "ITLOG: loss = {}".format(loss) )
        return loss.sum()

    def _construct_loss(self,
                        head_tag_representation: torch.Tensor,
                        child_tag_representation: torch.Tensor,
                        attended_arcs: torch.Tensor,
                        head_indices: torch.Tensor,
                        head_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = masked_log_softmax(attended_arcs,
                                                   mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation, child_tag_representation, head_indices)
        normalised_head_tag_logits = masked_log_softmax(head_tag_logits,
                                                        mask.unsqueeze(-1)) * float_mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        child_index = timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _greedy_decode(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       attended_arcs: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).to(dtype=torch.bool).unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation,
                                              child_tag_representation,
                                              heads)
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(self,
                    head_tag_representation: torch.Tensor,
                    child_tag_representation: torch.Tensor,
                    attended_arcs: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = (1 - mask.float()) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necesarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return torch.from_numpy(numpy.stack(heads)), torch.from_numpy(numpy.stack(head_tags))

    def _get_head_tags(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       head_indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        Returns
        -------
        head_tag_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(head_tag_representation)).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag_representations,
                                            child_tag_representation)
        return head_tag_logits

    def _get_mask_for_eval(self,
                           mask: torch.LongTensor,
                           pos_tags: torch.LongTensor) -> torch.LongTensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.

        Parameters
        ----------
        mask : ``torch.LongTensor``, required.
            The original mask.
        pos_tags : ``torch.LongTensor``, required.
            The pos tags for the sequence.

        Returns
        -------
        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self._pos_to_ignore:
            label_mask = pos_tags.eq(label).long()
            new_mask = new_mask * (1 - label_mask)
        return new_mask

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._attachment_scores.get_metric(reset)
        metrics['GramValAcc'] = self._gram_val_prediction_accuracy.get_metric(reset)
        metrics['LemmaAcc'] = self._lemma_prediction_accuracy.get_metric(reset)
        metrics['MeanAcc'] = (metrics['GramValAcc'] + metrics['LemmaAcc'] + metrics['LAS']) / 3.

        return metrics
