"""
Script to process a list of protein sequences present in input file and get its IDRs and pfam domains
"""
# Libraries to query D2P2 API and process the results
import requests
import argparse

# general utility libraries
import os

# Biopython related libraries for sequences analysis
from Bio import SeqIO


def get_pfam_domains_and_IDRs_in_proteins(input_file_path, output_file_path):
    """
    :param input_file_path: Path to .fasta file containing the full proteins sequences that we want to analyze
    :param output_file_path: Path to .tsv file that contains information about IDRs and pfam domains in the protein
    :return: None
    """
    # filename = "../data/raw_sequence_data/nuclear_proteins.fasta"

    # filename_write_data = "../data/raw_sequence_data/nuclear_proteins_pfam_domain_and_disorder.tsv"

    if not os.path.exists(os.path.dirname(output_file_path)):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    records = list(SeqIO.parse(input_file_path, "fasta"))

    output_data = {}

    with open(output_file_path, 'a+') as file_write:
        for item in records:

            id_current = str(item.id.split('|')[1])
            gene_name = str(item.id.split('|')[2])
            description = str(item.description.split('|')[-1])
            seq = str(item.seq)
            data = 'seqids=["{}"]'.format(id_current)
            request = requests.get('http://d2p2.pro/api/seqid', data)
            response = request.json()
            id_list = []

            if response[id_current]:

                count = 1;
                for pos in response[id_current][0][2]['disorder']['consranges']:
                    id = id_current + '_' + str(count)
                    output_data[id] = {}
                    output_data[id]['id'] = id_current
                    output_data[id]['seq'] = seq
                    output_data[id]['description'] = description
                    output_data[id]['idr_start'] = pos[0]
                    output_data[id]['idr_end'] = pos[1]
                    output_data[id]['gene_name'] = gene_name

                    id_list.append(id)
                    count = count + 1

                count_dom = 0

                if count == 1:
                    output_data[id_current] = {}
                    output_data[id_current]['id'] = id_current
                    output_data[id_current]['seq'] = seq
                    output_data[id_current]['description'] = description
                    output_data[id_current]['idr_start'] = -1
                    output_data[id_current]['idr_end'] = -1
                    output_data[id_current]['gene_name'] = gene_name

                    id_list.append(id_current)

                count_dom = 1

                id_domains = {}
                for pfams in (response[id_current][0][2]['structure']['pfam']):
                    pfam_name = pfams[2][0:7]

                    if (pfam_name.find('PF') > -1):
                        domain_id = id_current + '_pfam_' + str(count_dom)
                        id_domains[domain_id] = {}
                        id_domains[domain_id]['id'] = pfam_name
                        id_domains[domain_id]['start'] = int(pfams[7])
                        id_domains[domain_id]['end'] = int(pfams[8])
                        id_domains[domain_id]['escore'] = float(pfams[5])
                        id_domains[domain_id]['pfam_name'] = str(pfams[3])
                        id_domains[domain_id]['pfam_desc'] = str(pfams[4])
                        count_dom = count_dom + 1

                if count_dom == 1:
                    domain_id = id_current
                    id_domains[domain_id] = {}
                    id_domains[domain_id]['id'] = 'None'

                for ID in id_list:
                    output_data[ID]['pfam_list'] = []
                    for dom_id in list(id_domains.keys()):
                        str_to_write = ''
                        for key in id_domains[dom_id].keys():
                            str_to_write = str_to_write + str(id_domains[dom_id][key]) + '_'
                        output_data[ID]['pfam_list'].append(str_to_write)


            else:
                output_data[id_current] = {}
                output_data[id_current]['id'] = id_current
                output_data[id_current]['seq'] = seq
                output_data[id_current]['description'] = description
                output_data[id_current]['idr_start'] = -1
                output_data[id_current]['idr_end'] = -1
                output_data[id_current]['gene_name'] = gene_name
                output_data[id_current]['pfam_list'] = ['None']

                id_list.append(id_current)

            for ID in id_list:
                data_to_write = list(output_data[ID].values())
                file_write.write(str(ID) + '\t')
                for data in data_to_write:
                    if not isinstance(data, list):
                        file_write.write(str(data) + '\t')
                    else:
                        for pfam in data:
                            file_write.write(str(pfam) + '\t')

                file_write.write('\n')


if __name__ == "__main__":
    """
        Function is called when python code is run on command line and calls get_pfam_domains_and_IDRs_in_proteins()
        to initialize the simulation
    """
    parser = argparse.ArgumentParser(description='Take input and output files')
    parser.add_argument('--i', help="Path to input file", required=True)
    parser.add_argument('--o', help="Path to output file", required=True)
    args = parser.parse_args()
    get_pfam_domains_and_IDRs_in_proteins(args.i, args.o)


