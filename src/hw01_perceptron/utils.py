import random
from nltk import word_tokenize

def dot(dictA, dictB):
    return sum([dictA.get(tok) * dictB.get(tok, 0) for tok in dictA])

def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]

class DataInstance:
    def __init__(self, feature_counts, label):
        """ A data instance consists of a dictionary with feature counts (string -> int) and a label (True or False)."""
        self.feature_counts = feature_counts
        self.label = label

    @classmethod
    def from_list_of_feature_occurrences(cls, feature_list, label):
        """ Creates feature counts for all features in the list."""
        feature_counts = dict()
        for word in feature_list:
            count = feature_counts.get(word, 0)
            feature_counts[word] = count + 1
        return cls(feature_counts, label)

    @classmethod
    def from_text_file(cls, filename, label):
        with open(filename, 'r') as myfile:
            token_list = normalized_tokens(myfile.read().strip())
        return cls.from_list_of_feature_occurrences(token_list, label)

class Dataset:
    def __init__(self, instance_list):
        """ A data set is defined by a list of instances """
        self.instance_list = instance_list
        self.feature_set = set.union(*[set(inst.feature_counts.keys()) for inst in instance_list])


    def get_topn_features(self, n):
        """ This returns a set with the n most frequently occurring features (i.e. the features that are contained in most instances)."""
        all_feature = dict()
        for feature in self.feature_set:
            all_feature[feature] = 0
            for inst in self.instance_list:
                if feature in inst.feature_counts.keys():
                    all_feature[feature] += 1
        feature_tuple = sorted(all_feature.items(), key=lambda keyvalue: (keyvalue[1], keyvalue[0]), reverse = True) [:n]
        result = []
        for key, value in feature_tuple:
            result.append(key)
        return set(result)

    def set_feature_set(self, feature_set):
        """
        This restrics the feature set. Only features in the specified set all retained. All other feature are removed
        from all instances in the dataset AND from the feature set."""
        self.feature_set = self.feature_set.intersection(feature_set)
        for inst in self.instance_list:
            copy_dict = inst.feature_counts.copy()
            for feature in inst.feature_counts.keys():
                if feature not in feature_set:
                    del copy_dict[feature]
            inst.feature_counts = copy_dict

    def most_frequent_sense_accuracy(self):
        """ Computes the accuracy of always predicting the overall most frequent sense for all instances in the dataset. """
        label_count = dict()
        for inst in self.instance_list:
            if inst.label not in label_count.keys():
                label_count[inst.label] = 1
            else:
                label_count[inst.label] += 1
        max = 0
        all = 0
        for key, value in label_count.items():
            all += value
            max = value if value > max else max

        return max/all if all != 0 else 0.0

    def shuffle(self):
        """ Shuffles the dataset. Beneficial for some learning algorithms."""
        random.shuffle(self.instance_list)
