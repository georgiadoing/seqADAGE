'''
This script downloads pseudomonas KEGG or GO terms from TRIBE and writes them
into a file.

Usage:
    python get_annotations.py out_file target

    out_file: file path to the output file, a tab-delimited file with the first
              column storing geneset names and second column storing the genes
              in each geneset separated by comma.
    target:   specify the type of genesets, can be either "KEGG" or "GO"
'''
import requests
import sys
import csv

TRIBE_URL = 'http://tribe.greenelab.com'
limit = 2000  # this limits the number of items to return at one time


def get_terms(offset=None, target=None):
    '''
    this function queries the TRIBE web server to get KEGG or GO terms for
    pseudomonas.

    input:
        offset  get terms after the offset, used if the query limit is exceeded
        target  the type of geneset, either "KEGG" or "GO"
    return:
        a list of geneset dictionaries with the format (geneset title: [genes])
    '''
    # pseudonoas aeruginosa is organism number 9 in tribe
    parameters = {'show_tip': 'true', 'organism': 9, 'xrdb': 'Symbol',
                  'limit': limit, 'offset': offset}
    r = requests.get(TRIBE_URL + '/api/v1/geneset/?title__startswith='+target,
                     params=parameters)

    print(r.status_code)
    result = r.json()
    genesets = result['objects']
    simplified_genesets = []
    count = 0
    for geneset in genesets:
        count += 1
        if geneset['tip'] is None:  # skip it if the tip version is none
            continue
        # only download genesets within (5,100) size
        if (len(geneset['tip']['genes']) > 100) or\
           (len(geneset['tip']['genes']) < 5):
            continue
        else:
            new_geneset = {}
            new_geneset['title'] = geneset['title']
            new_geneset['genes'] = geneset['tip']['genes']
            simplified_genesets.append(new_geneset)
    if count == limit:
        print 'Need to increase the limit.'
    return simplified_genesets


if __name__ == '__main__':
    out_file = sys.argv[1]
    target = sys.argv[2]
    anno_terms = get_terms(target=target)

    with open(out_file, 'wb') as out_fh:
        writer = csv.writer(out_fh, delimiter='\t')
        for term in anno_terms:
            writer.writerow([term['title'], str(len(term['genes'])),
                             ';'.join([str(x) for x in term['genes']])])
