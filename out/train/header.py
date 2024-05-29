import sys


def extract_relations(data):
    triplets = data.split('▸')
    relations = set()
    for triplet in triplets:
        relation = triplet.split('|')[1].strip()
        relations.add(relation)
    return relations


def extract_triplets(input):
    triplets = input.split('▸')
    # create tuple of (x, relation, y)
    # strip to remove leading and trailing spaces
    triplets = [triplet.split('|') for triplet in triplets]
    return [tuple([x.strip() for x in triplet]) for triplet in triplets]


for data in sys.stdin:
    relations = extract_relations(data)
    triplets = extract_triplets(data)
