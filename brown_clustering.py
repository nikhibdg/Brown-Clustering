import os
import glob
import string
import pprint
import itertools
import operator
import sys
import math
from collections import OrderedDict
from numpy import array
import numpy as np
from scipy import spatial


class Preprocess:

    def preprocess(self, data):
        final_data = []
        for line in data:
            stripped_line = line.strip()
            line_list = stripped_line.split()
            if len(line_list) != 0:
                processed_line = []
                for token in line_list:
                    word_token = token.split('/')[0].lower()

                    processed_line.append(word_token.lower())
                lowercased_line = ' '.join(processed_line)
                final_data.append(lowercased_line)

        return final_data

class Process:

    def replace_by_unk(self, data):

        word_frequency = {}
        all_data = []
        punctuation = string.punctuation
        punctuation = punctuation + "''" + "``" + "--"

        for line in data:
            tokenized_list = line.split()
            for token in tokenized_list:
                if token not in punctuation:
                    if token not in word_frequency:
                        word_frequency[token] = 1
                    else:
                        word_frequency[token] += 1
        for line in data:
            tokenized_list = line.split()
            new_line = []
            for token in tokenized_list:
                if token in word_frequency:
                    if word_frequency[token] <= 0:
                        new_line.append("UNK")
                    else:
                        new_line.append(token)

            unk_replaced_line = ' '.join(new_line)
            all_data.append(unk_replaced_line)

        return all_data

    def sort_vocab(self, data):

        vocab_count = {}
        for line in data:
            tokenized_list = line.split()
            for token in tokenized_list:
                if token not in vocab_count:
                    vocab_count[token] = 1
                else:
                    vocab_count[token] += 1

        sorted_vocab_dict = OrderedDict(sorted(vocab_count.items(), key=lambda
        t:t[1], reverse=True))

        return sorted_vocab_dict


class Bigrams:

    def get_bigrams(self, data):
        bigrams_sentence = []
        for line in data:
            new_line = "START " + line + " STOP"

            line_list = new_line.split()
            for element in range(len(line_list)):
                if element < len(line_list) - 1:
                    bigrams_sentence.append((line_list[element], line_list[element + 1]))

        return bigrams_sentence


class Brown_Cluster:

    def get_bigram_counts(self, bigrams):

        bigram_counts = {}

        for bigram in bigrams:
            if bigram not in bigram_counts:
                bigram_counts[bigram] = 1
            else:
                bigram_counts[bigram] += 1

        return bigram_counts

    def get_top_clusters(self, cent_top_words):

        cluster = {}
        count = 0
        for key,value in cent_top_words:
            count += 1
            cluster[key] = count

        return cluster

    def get_bigram_clusters_count(self, cluster,i, j, cluster_bigram_counts):

        total_denom_count = 0

        for key, value in cluster_bigram_counts.items():
            total_denom_count += value

        return total_denom_count

    def get_quality_c(self, cluster, i, j, bigrams_sentence, bigram_counts,
    sorted_vocab_dict,cluster_bigram_counts, cluster_counts, N, i_word, j_word):

        i_cluster = i
        j_cluster = j

        trans = (i_cluster, j_cluster)
        if trans in cluster_bigram_counts:
            count_trans = cluster_bigram_counts[trans]
            trans_prob = count_trans / N

        else:
            trans_prob = 0

        n_i = cluster_counts[i_cluster]
        n_j = cluster_counts[j_cluster]

        #p_n = self.get_p_n(cluster_counts)
        p_n = N

        p_i = n_i/p_n

        p_j = n_j/p_n


        second_term = trans_prob/ (p_i * p_j)
        if second_term != 0:
            second_term_log = math.log(second_term)
        else:
            second_term_log = 0
        single_quality_entry = trans_prob * second_term_log

        return single_quality_entry


    def get_quality_corpus(self, cluster, bigrams_sentence, bigram_counts,
    sorted_vocab_dict,cluster_bigram_counts, cluster_counts, N):

        ci = 0
        cj = 0
        f = []
        final_quality = []
        for word, count in bigram_counts.items():
            if word[0] in cluster and word[1] in cluster:
                #if (cluster[word[0]], cluster[word[1]]) in cluster_bigram_counts:
                i = cluster[word[0]]
                j = cluster[word[1]]
                i_word = word[0]
                j_word = word[1]
                quality_c = self.get_quality_c(cluster, i, j, bigrams_sentence,
                bigram_counts, sorted_vocab_dict, cluster_bigram_counts,
            cluster_counts, N, i_word, j_word)
                final_quality.append(quality_c)
        result = sum(final_quality)

        return result

    def cluster_bigrams(self, cluster, bigrams_sentence, bigram_count,
    sorted_vocab_dict, i_value, j_value):
        cluster_bigram_counts = {}
        for key, value in bigram_counts.items():
            if key[0] in cluster and key[1] in cluster:

                if (cluster[key[0]], cluster[key[1]]) not in cluster_bigram_counts:
                    cluster_bigram_counts[(cluster[key[0]], cluster[key[1]])] = value
                else:
                    cluster_bigram_counts[(cluster[key[0]], cluster[key[1]])] += value


        return cluster_bigram_counts

    def get_cluster_counts(self, cluster, bigrams_sentence, bigram_counts,
    sorted_vocab_dict):

        cluster_count_dict = {}
        for key, value in sorted_vocab_dict.items():
            if key in cluster:
                if cluster[key] not in cluster_count_dict:
                    cluster_count_dict[cluster[key]] = value
                else:
                    cluster_count_dict[cluster[key]] += value

        return cluster_count_dict

    def get_merge_cluster(self, cluster, bigrams_sentence, bigram_counts,
    sorted_vocab_dict, N):

        total_quality = {}
        for i, i_value in cluster.items():
            for j, j_value in cluster.items():
                if i_value != j_value:
                    # Optimize
                    temp_cluster = {}
                    for k,v in cluster.items():
                        temp_cluster[k] = v

                    for k,v in cluster.items():
                        if v == j_value:
                            temp_cluster[k] = i_value
                        else:
                            temp_cluster[k] = v

                    cluster_bigram_counts = self.cluster_bigrams(temp_cluster,
                        bigrams_sentence, bigram_counts, sorted_vocab_dict, i_value
                        , j_value)

                    cluster_counts = self.get_cluster_counts(temp_cluster, bigrams_sentence, bigram_counts, sorted_vocab_dict)

                    quality_corpus = self.get_quality_corpus(temp_cluster,
                bigrams_sentence, bigram_counts, sorted_vocab_dict,
                cluster_bigram_counts, cluster_counts, N)
                    total_quality[(i,j)] = quality_corpus

        if len(total_quality) != 0:
            merge_clusters  = max(total_quality.items(), key=operator.itemgetter(1))[0]

            cluster_no  = total_quality[merge_clusters]
        else:
            merge_clusters = None

        return merge_clusters


if __name__ == "__main__":

    curr_path = os.path.dirname(os.path.realpath(__file__))

    all_data = []

    prp = Preprocess()
    #for new_file in brown_data:
    with open("subset_data.txt", "r") as fp:
        file_text = fp.readlines()
        preprocessed_data = prp.preprocess(file_text)
        for line in preprocessed_data:
            all_data.append(line)

    #all_data contains text with dummy POS tags removed and lowercased

    process = Process()

    # 3.1 Replacing count less than 10 as unk
    unk_data = process.replace_by_unk(all_data)

    # Sorting of vocabulary
    sorted_vocab_dict  = process.sort_vocab(unk_data)
    print(sorted_vocab_dict)

    v = []
    for line in unk_data:
        tokens = line.split()
        for token in tokens:
            v.append(token)

    N = len(v)

    # Uncomment to print the ranked vocabulary list
    #pprint.pprint(sorted_vocab_dict)

    ############################################################
    # 3.2 Adding of START and STOP symbols

    ############################################################

    bigrams = Bigrams()
    # Adding of START and STOP symbols and getting bigrams
    bigrams_sentence = bigrams.get_bigrams(unk_data)
    #print("Bigrams with START and stop symbols")
    #print(bigrams_sentence)

    ###############################################################

    # 3.3 Implementation of brown clustering

    ###############################################################

    brw = Brown_Cluster()
    bigram_counts = brw.get_bigram_counts(bigrams_sentence)

    cent_top_words = itertools.islice(sorted_vocab_dict.items(), 0, 8)

    cluster = brw.get_top_clusters(cent_top_words)

    length_vocab = len(sorted_vocab_dict)
    next_words = itertools.islice(sorted_vocab_dict.items(), 8, length_vocab)

    # Merging of next words into k clusters
    cluster_number = 9

    cluster_information = {}

    word_vector = {}

    for word in sorted_vocab_dict:
        word_vector[word] = ''

    print(len(sorted_vocab_dict))
    iters = 0
    for word, count in next_words:

        cluster[word] = cluster_number
        iters += 1

        two_merge_clusters = brw.get_merge_cluster(cluster, bigrams_sentence, bigram_counts, sorted_vocab_dict, N)

        print(two_merge_clusters)
        first_word = two_merge_clusters[0]
        second_word = two_merge_clusters[1]

        first_word_cluster = cluster[first_word]
        second_word_cluster = cluster[second_word]

        place_holder = {}
        for k,v in cluster.items():
            place_holder[k] = v

        if first_word_cluster < second_word_cluster:
            for k,v in place_holder.items():
                if v == first_word_cluster:
                    word_vector[k] = "0" + word_vector[k]
                else:
                    pass

            for k,v in place_holder.items():
                if v == second_word_cluster:
                    cluster[k] = first_word_cluster
                    word_vector[k] = "1" + word_vector[k]
                else:
                    pass
        else:
            for k,v in place_holder.items():
                if v == second_word_cluster:
                    word_vector[k] = "0" + word_vector[k]
                else:
                    pass

            for k,v in place_holder.items():
                if v == first_word_cluster:
                    cluster[k] = second_word_cluster
                    word_vector[k] = "1" + word_vector[k]
                else:
                    pass

        #sys.exit()
        cluster_number = cluster_number + 1

    with open("clusters.txt", "w") as fp:
        fp.write(str(cluster))

    with open("vectors.txt", "w") as fp:
        fp.write(str(word_vector))
    print("K clusters")
    pprint.pprint(cluster)
    print(word_vector)
    final_cluster = {}
    for word, count in cluster.items():
        if count not in final_cluster:
            final_cluster[count] = [word]
        else:
            word_list = final_cluster[count]
            word_list.append(word)
            final_cluster[count] = word_list


    new_cluster = {}
    for word, count in cluster.items():
        new_cluster[word] = count

    iters = 0

    for word, count in new_cluster.items():

        two_merge_clusters = brw.get_merge_cluster(cluster, bigrams_sentence, bigram_counts, sorted_vocab_dict, N)

        if two_merge_clusters is not None:
            print(two_merge_clusters)

            first_word = two_merge_clusters[0]
            second_word = two_merge_clusters[1]

            first_word_cluster = cluster[first_word]
            second_word_cluster = cluster[second_word]


            if first_word_cluster < second_word_cluster:
                for k,v in cluster.items():
                    if v == first_word_cluster:
                        word_vector[k] = "0" + word_vector[k]
                    else:
                        pass

                for k,v in cluster.items():
                    if v == second_word_cluster:
                        cluster[k] = first_word_cluster
                        word_vector[k] = "1" + word_vector[k]
                    else:
                        pass
            else:
                for k,v in cluster.items():
                    if v == second_word_cluster:
                        word_vector[k] = "0" + word_vector[k]
                    else:
                        pass

                for k,v in cluster.items():
                    if v == first_word_cluster:
                        cluster[k] = second_word_cluster
                        word_vector[k] = "1" + word_vector[k]
                    else:
                        pass


    print(cluster)
    print("Word strings")
    pprint.pprint(word_vector)

    ##########################################################

    # 3.4 Cosine distance

    ########################################################

    print("computing cosined")
    vector_length = []
    for word, vector in word_vector.items():
        vector_length.append(len(vector))

    max_length = max(vector_length)


    new_word_vector = {}

    for word, vector in word_vector.items():
        padding_length = max_length - len(vector)
        string_vector = "0"*padding_length + vector
        #string_vec_list = string_vector.split('|')
        #print(string_vec_list)

        w_vector = []
        for i in string_vector:
            w_vector.append(int(i))

        new_word_vector[word] = array(w_vector)


    # Computing cosine dist

    average_cosine = {}
    print(final_cluster)

    for cluster, words_list in final_cluster.items():
        d = len(words_list)
        cos_dist_cluster = np.zeros((d,d))
        for i in range(len(words_list)):
            for j in range(len(words_list)):
                vector_i = new_word_vector[words_list[i]]
                vector_j = new_word_vector[words_list[j]]
                cos_dist_cluster[i,j] = spatial.distance.cosine(vector_i, vector_j)
        average_cosine[cluster] = np.average(cos_dist_cluster)


    print("Avergae cosing per cluster")
    pprint.pprint(average_cosine)
    with open("average_cosine.txt", "w") as fp:
        fp.write(str(average_cosine))
