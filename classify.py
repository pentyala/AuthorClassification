#!/usr/bin/env python
from collections import defaultdict
from csv import DictReader, DictWriter
import nltk
import codecs
import sys
import gzip
from nltk import bigrams
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.tokenize import TreebankWordTokenizer
from string import punctuation

kTOKENIZER = TreebankWordTokenizer()

def ms(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    else:
        return word.lower()

class FeatureExtractor:

    def __init__(self):
        """
        You may want to add code here
        """
        self.s_words = []
        self.pronunciations = nltk.corpus.cmudict.dict()
        None


    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.
        """
        # Done implementing
        max_syl = 0
        if word not in self.pronunciations:
          return max_syl
        small_pron = []
        small_len = -1
        for pron in self.pronunciations[word]:
          if small_len == -1:
            small_pron = pron
            small_len = len(pron)
            continue
          if small_len > len(pron):
            small_pron = pron
            small_len = len(pron)
        c = 0
        for phone in small_pron:
          if phone[-1] == '1' or phone[-1] == '0' or phone[-1] == '2':
            c += 1
        return c

    def features(self, text):
        # Features is the word count dictionary.
        d = defaultdict(int)
        # for ch in text:
        #     if ch in punctuation:
        #         pron_count += 1
        tokens = kTOKENIZER.tokenize(text)
        l = list()
        pos_tags = []
        tot_len = 0
        st_words = 0
        for token in tokens:
            if token.endswith("st"):
                st_words += 1
            tot_len += len(token)
        d['avg_len'] = tot_len/len(tokens)
        d['tot_len'] = tot_len
        d['tokens'] = len(tokens)
        for token in tokens:
            if token in punctuation or token in sw.words('english'):
                l.append(token)
        for tok in l:
            tokens.pop(tokens.index(tok))
        syl = 0
        for tag in nltk.pos_tag(tokens):
            pos_tags.append(tag[1])
        # pairs = [ " ".join(pair) for pair in nltk.bigrams(pos_tags)]
        # for word in pairs:
        #     d[word] += 1
        d['pos'] = " ".join(pos_tags)
        d['end_pos'] = pos_tags[-1]
        if len(pos_tags) > 1:
            d['end_bi_pos'] = pos_tags[-1]+" "+pos_tags[-2]
        else:
            d['end_bi_pos'] = pos_tags[-1]
        # adjectives =[token for token, pos in nltk.pos_tag(nltk.word_tokenize(text)) if pos.startswith('IN')]
        special_count = 0
        pairs = [ " ".join(pair) for pair in nltk.bigrams(tokens)]
        for word in pairs:
            d[word] += 1
        pairs = [ " ".join(pair) for pair in nltk.trigrams(tokens)]
        for word in pairs:
            d[word] += 1
        for word in tokens:
            if word in punctuation or word in sw.words('english'):
                continue
            syl += self.num_syllables(word)
            if word in self.s_words:
                d[ms(word)] += 1
            else:
                d[ms(word)] += 1    
        return d
        
reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--trainfile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input train file")
    parser.add_argument("--testfile", "-t", nargs='?', type=argparse.FileType('r'), default=None, help="input test file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this fraction of total')
    args = parser.parse_args()
    trainfile = prepfile(args.trainfile, 'r')
    if args.testfile is not None:
        testfile = prepfile(args.testfile, 'r')
    else:
        testfile = None
    outfile = prepfile(args.outfile, 'w')

    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()
    
    # Read in training data
    train = DictReader(trainfile, delimiter='\t')
    
    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []
    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:         
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))

    sys.stderr.write("Training classifier ...\n")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
    total = len(dev_test)
    right = 0
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
    
    sys.stderr.write("Accuracy on dev: %f\n" % (float(right) / float(total)))

    if testfile is None:
        sys.stderr.write("No test file passed; stopping.\n")
    else:
        # Retrain on all data
        classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)

        # Read in test section
        test = {}
        for ii in DictReader(testfile, delimiter='\t'):
            test[ii['id']] = classifier.classify(fe.features(ii['text']))

        # Write predictions
        o = DictWriter(outfile, ['id', 'pred'])
        o.writeheader()
        for ii in sorted(test):
            o.writerow({'id': ii, 'pred': test[ii]})