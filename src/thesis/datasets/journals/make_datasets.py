import os
import json
from thesis.utils.constants import DATA_PATH
from thesis.datasets.journals.read_dataset import load_jsonl

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')


def main(dataset='icpr_20'):

    S2ORC_PATH = 's2orc-journal'
    SESSION_PATH = 'sessions'

    PRINCIPAL_FIELD = 'title'
    NO_GROUP = None

    GROUP_FILE = 'groups'
    TITLE_FILE = 'titles'

    JOURNAL_PATH = os.path.join(DATA_PATH, S2ORC_PATH)
    SESSION_FULL = os.path.join(DATA_PATH, S2ORC_PATH, SESSION_PATH)

    def read_session_file(GROUP_FILE):
        lines = list()
        with open(os.path.join(SESSION_FULL, f'{dataset}_{GROUP_FILE}.txt')) as f_in:
            # All lines including the blank ones
            lines = [line.rstrip() for line in f_in]
            # Non-blank lines
            lines = [line for line in lines if line]

        return lines

    groups_lines = read_session_file(GROUP_FILE)
    titles_lines = read_session_file(TITLE_FILE)
    total = len(titles_lines)

    def lists_to_dict(lines, invert=False):
        d = dict()
        for line in lines:
            k, v = line.split(' : ')
            if not invert:
                d[k] = v
            else:
                d[v] = k
        return d

    groups_dict = lists_to_dict(groups_lines)
    titles_dict = lists_to_dict(titles_lines, invert=True)
    del groups_lines
    del titles_lines

    def assign_group_title(groups_dict, titles_dict):
        d = dict()
        for title_k, title_v in titles_dict.items():
            for group_k, group_v in groups_dict.items():
                title_to_check = title_v.split('-')[0]
                if group_k == title_to_check:
                    d[title_k] = group_v
        return d

    title_group = assign_group_title(groups_dict, titles_dict)
    del groups_dict
    del titles_dict

    def read_dataset_jsonl():
        DATASET_PATH = os.path.join(JOURNAL_PATH, f"{dataset}.jsonl")
        json_list = list()
        with open(DATASET_PATH, 'r') as j_in:
            for json_line in j_in.readlines():
                paper = json.loads(json_line)
                if paper[PRINCIPAL_FIELD]:
                    json_list.append(paper)

        return json_list

    dataset_list = read_dataset_jsonl()

    def get_most_similar(title_group, string):

        # Function to measure the similarity
        # between a sentence and lots of others
        # using cosine similarity.

        # tokenization
        X_list = word_tokenize(string)

        # sw contains the list of stopwords
        sw = stopwords.words('english')

        # remove stop words from the string
        X_set = {w for w in X_list if not w in sw}

        similarities = dict()
        for title, group in title_group.items():
            Y_list = word_tokenize(title)

            l1 = []
            l2 = []

            Y_set = {w for w in Y_list if not w in sw}

            # form a set containing keywords of both strings
            rvector = X_set.union(Y_set)
            for w in rvector:
                if w in X_set:
                    l1.append(1)  # create a vector
                else:
                    l1.append(0)
                if w in Y_set:
                    l2.append(1)
                else:
                    l2.append(0)
            c = 0

            # cosine formula
            for i in range(len(rvector)):
                c += l1[i]*l2[i]
            similarities[title] = (c / float((sum(l1)*sum(l2))**0.5), group)

        best = sorted(similarities.items(), key=lambda item: item[1]).pop()
        return best[0], best[1][0], best[1][1]

    def add_group_field(dataset_list, title_group):
        recovered = 0
        finished_jsonl = list()
        for json_line in dataset_list:
            finished_line = dict(json_line)
            # we check if the title in group assign
            # and the title of the paper match
            #
            # finished_line['group'] = title_group.get(finished_line[PRINCIPAL_FIELD], NO_GROUP)
            #
            # we need to approximate this
            t, s, g = get_most_similar(
                title_group, finished_line[PRINCIPAL_FIELD])
            finished_jsonl.append(finished_line)
            if s > 0.5:
                finished_line['group'] = g
                recovered += 1
            else:
                finished_line['group'] = NO_GROUP

        return finished_jsonl, recovered

    finished_jsonl, recovered = add_group_field(dataset_list, title_group)
    del dataset_list
    del title_group

    print(
        f"Recover paper's group for {recovered}/{total} papers on dataset {dataset}")

    def save_back_to_file(finished_jsonl):
        DATASET_PATH = os.path.join(JOURNAL_PATH, f"ex_{dataset}.jsonl")
        with open(DATASET_PATH, 'a') as j_out:
            for json_line in finished_jsonl:
                json.dump(json_line, j_out)
                j_out.write('\n')

        return True

    save_back_to_file(finished_jsonl)
    del finished_jsonl


if __name__ == '__main__':

    main('icpr_20')
    main('icdar_19')
