import torch
import nltk

from nltk.parse.corenlp import CoreNLPDependencyParser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
parse, = dep_parser.raw_parse('What is the first Indian text to introduce iron?')
print(parse.nodes[0])
# parse = dep_parser.tag('The quick brown fox jumps over the lazy dog.'.split())
print(parse.to_conll(10))
