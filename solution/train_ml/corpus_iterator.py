# -*- coding: utf-8 -*-

import attr
from itertools import chain
from typing import Tuple, List


@attr.s(frozen=True)
class Token:
    token_no: str = attr.ib()
    text: str = attr.ib()
    lemma: str = attr.ib()
    pos_tag: str = attr.ib()
    grammar_value: str = attr.ib()
    head: int = attr.ib()
    head_tag: str = attr.ib()


class Sentence(object):
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._tm = []
        for token in self._tokens:
            if not ':' in token.token_no:
                self._tm.append( [token] )
            else:
                self._tm[-1].append(token)

    @property
    def words(self):
        return [token[0].text for token in self._tm]

    @property
    def pos_tags(self):
        if not self._tokens or self._tokens[0].pos_tag is None:  # если предложение пусто или колонка upos не заполнена (?)
            return None
        tmp = [ [ tv.pos_tag for tv in token] for token in self._tm ]
        tmp_ext = []
        for t in tmp:
            d = 3 - len(t)
            tmp_ext.append( t + ['EMPTY']*d )
        return tmp_ext

    @property
    def lemmas(self):
        if not self._tokens or self._tokens[0].lemma is None:
            return None
        #return [ [ tv.lemma for tv in token] for token in self._tm ]
        return [ token[0].lemma for token in self._tm]

    @property
    def grammar_values(self):
        if not self._tokens or self._tokens[0].grammar_value is None:
            return None
        #return [ [ tv.grammar_value for tv in token] for token in self._tm ]
        tmp = [ [ tv.grammar_value for tv in token] for token in self._tm ]
        tmp_ext = []
        for t in tmp:
            d = 3 - len(t)
            tmp_ext.append( t + ['EMPTY|_']*d )
        return tmp_ext

    @property
    def heads(self):
        if not self._tokens or self._tokens[0].head is None:
            return None
        return [token[0].head for token in self._tm]

    @property
    def head_tags(self):
        if not self._tokens or self._tokens[0].head_tag is None:
            return None
        return [token[0].head_tag for token in self._tm]

    def __len__(self):
        return len(self._tm)


class CorpusIterator:
    def __init__(self, path: str, separator: str='\t', token_col_index: int=1, lemma_col_index: int=2,
                 grammar_val_col_indices: Tuple=(3, 5), grammemes_separator: str='|',
                 head_col_index: int=6, head_tag_col_index: int=7,
                 skip_line_prefix: str='#', encoding: str='utf8'):
        """
        Creates iterator over the corpus in conll-like format:
        - each line contains token and its annotations (lemma and grammar value info) separated by ``separator``
        - sentences are separated by empty line
        :param path: path to corpus
        :param separator: separator between fields
        :param token_col_index: index of token field
        :param lemma_col_index: index of lemma field
        :param grammar_val_col_indices: indices of grammar value fields (e.g. POS and morphological tags)
        :param grammemes_separator: separator between grammemes (as in 'Case=Nom|Definite=Def|Gender=Com|Number=Sing')
        :param head_col_index: index of head field
        :param head_tag_col_index: index of head_tag field
        :param skip_line_prefix: prefix for comment lines
        :param encoding: encoding of the corpus file
        """
        self._path = path
        self._separator = separator
        self._token_col_index = token_col_index
        self._lemma_col_index = lemma_col_index
        self._grammar_val_col_indices = grammar_val_col_indices
        self._grammemes_separator = grammemes_separator
        self._head_col_index = head_col_index
        self._head_tag_col_index = head_tag_col_index
        self._skip_line_prefix = skip_line_prefix
        self._encoding = encoding

    def __enter__(self):
        self._file = open(self._path, encoding=self._encoding)
        return self

    def __exit__(self, type, value, traceback):
        self._file.close()

    def __iter__(self):
        return self

    def _read_token(self, line):
        fields = line.split(self._separator)
        
        token_no = fields[0]

        token_text = fields[self._token_col_index]
        lemma, pos_tag, grammar_value, head, head_tag = None, None, None, None, None

        if self._lemma_col_index is not None and self._lemma_col_index < len(fields):
            lemma = fields[self._lemma_col_index]

        if (self._grammar_val_col_indices is not None
            and all(index < len(fields) for index in self._grammar_val_col_indices)
        ):
            grammar_value = '|'.join(chain(*(sorted(fields[col_index].split(self._grammemes_separator))
                                                for col_index in self._grammar_val_col_indices)))

        if self._grammar_val_col_indices and self._grammar_val_col_indices[0] < len(fields):
            pos_tag = fields[self._grammar_val_col_indices[0]]

        if self._head_col_index is not None and self._head_col_index < len(fields):
            hv = fields[self._head_col_index]
            if hv != "_":
                head = int(hv)

        if self._head_tag_col_index is not None and self._head_tag_col_index < len(fields):
            head_tag = fields[self._head_tag_col_index]

        return Token(
            token_no=token_no,
            text=token_text,
            lemma=lemma,
            pos_tag=pos_tag,
            grammar_value=grammar_value,
            head=head,
            head_tag=head_tag
        )

    def __next__(self) -> Sentence:
        while True:
            sentence = []
            for line in self._file:
                line = line.rstrip()
                if line.startswith(self._skip_line_prefix):
                    continue
                if len(line) == 0:
                    break

                sentence.append(self._read_token(line))
            else:
                if not sentence:
                    raise StopIteration

            if sentence:
                return Sentence(sentence)
