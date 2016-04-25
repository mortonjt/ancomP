import argparse
import pandas as pd
import sys
from ancomP.stats.ancom import ancom

def run(otu_file, meta_file, voi, out_handle, alpha, permutations):
    otu_table = pd.read_table(otu_file,index_col=0)
    metadata  = pd.read_table(meta_file)
    cats = metadata[voi].as_matrix()

    sig_otus = ancom_cl(otu_table,cats,alpha,permutations)

    outhandle.write('\n'.join(sig_otus) + '\n')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=\
            'Performs ANCOM statistical test')
    parser.add_argument(\
        '--otu-table', type=str, required=True,
        help='Input classic OTU table (tab-delimited)')
    parser.add_argument(\
        '--meta-data', type=str, required=True,
        help='Input metadata file (tab-delimited)')
    parser.add_argument(\
        '--variable-of-interest', type=str, required=True,
        help="Variable to run ANCOM analysis on \
              contained within the metadata \
              (must be binary)")
    parser.add_argument(\
        '--output', type=str, required=False, default = None,
        help='Tab delimited output of significant OTUs')
    parser.add_argument(\
        '--alpha', type=int, required=False, default = 0.05,
        help='Signficance threshold')
    parser.add_argument(\
        '--permutations', type=str, required=False, default = 1000,
        help='Number of permutations to calculate exact p-value')
    args = parser.parse_args()

    if args.output != None:
        outhandle = open(args.output,'w')
    else:
        outhandle = sys.stdout

    run(args.otu_table,
        args.meta_data,
        args.variable_of_interest,
        outhandle,
        args.alpha,
        args.permutations)
