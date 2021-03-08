# -*- coding: utf-8 -*-
"""
Code related to the publication by Kaur and Jakob et al., 2021 (Nature).
OMPdb version needed to replicate our results: 01. Dec 2020 (UniProt release 2020_4).
More information on github @hiller-lab
"""

#%% Settings

## Input files

OMPdb_flat       = r"OMPdb/OMPdb.flat" # Complete database in Text format from http://www.ompdb.org/download.php
OMPdb_UniProt90  = r"OMPdb/OMPdb.90"   # Non-redundant sequence subset (90%) from http://www.ompdb.org/download.php
UniProt_speclist = r"UniProt/speclist.txt" # UniProtKB's controlled vocabulary of species from https://www.uniprot.org/docs/speclist

## Output files

bsignals_out = r"out/bsignals.fasta" # Either filepath as string (if output desired) or None (no output)
figures_out  = r"out/"                # Either dir path (with slash) as string (if output desired) or None (no output)

## Other settings

DAROBACTIN    = "WNWSKSF"    # Default is "WNWSKSF" (darobactin A)
RANDOM_SEED   = "darobactin" # Default is "darobactin", but can be anything
PRINT_SUMMARY = False        # Default is False


#%% Libraries and functions

import re
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from scipy.stats import percentileofscore

class motif():
    '''
    Class to generate PSSM, PSFM, PSCM.
    References:
     -  Schneider TD, Stephens RM. Sequence logos: a new way to display consensus sequences. Nucleic Acids Res.
        1990 Oct 25;18(20):6097-100. doi: 10.1093/nar/18.20.6097. PMID: 2172928; PMCID: PMC332411.
     -  https://en.wikipedia.org/wiki/Position_weight_matrix
    '''
    def __init__(self,
                 aligned_sequences,
                 alphabet=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
                 init_counts=None,
                 *args,
                 **kwargs):
        
      # Step 1: Load aligned sequences (list type)
        self.__sequences = aligned_sequences
        
      # Step 2: Compute letter counts, frequencies (= relative counts) and 'bits'
        self.__counts = self.__counts_func(alphabet) if init_counts is None else init_counts
        self.__frequencies = self.__frequencies_func()
        self.__bits = self.__bits_func(alphabet)
        
      # Step 3: Compute position-weighted matrix (PWM), aka position-specific scoring matrix (PSSM)
        self.__pwm = self.__pwm_func(*args, **kwargs)
        self.__consensus, self.__mv = self.__consensus_func()
    
    def __str__(self):
        return self.__consensus
    
    def __counts_func(self, alphabet):
        # Split strings into uppercase-character lists and convert into 2D numpy array
        sequences = np.asarray([list(s.upper()) for s in  self.__sequences])

        counts_list = []
        # For every position in the alignment, do...
        for col_index in range(0, min([len(s) for s in sequences])):
            # ...create a counts dictionary from the alphabet and assign the value '0' to each entry
            d = dict.fromkeys(alphabet, 0)
            # ...count how many times an entry (letter) is present
            uniques, counts = np.unique(sequences[:, col_index], return_counts=True)
            for i, u in enumerate(uniques):
                # ...replace the initial '0' value with the counts
                d[u] = float(counts[i])
            # ...append the counts dictionary to the counts_list    
            counts_list.append(d)
        
        return counts_list
    
    def __frequencies_func(self):
        return [{u : d[u]/sum(d.values()) for u in d.keys()} for d in self.__counts]
    
    def __bits_func(self, alphabet, e_func=lambda x: x*0): # See ref. Schneider & Stephens, 1990
        counts = self.__counts
        frequencies = self.__frequencies
        
        uncertainties = [] # Uncertainty H at each position of the consensus (the smaller the better)
        for d in frequencies:
            H = 0
            for u in d.keys():
                if d[u] > 0: H -= d[u]*np.log2(d[u])
            uncertainties.append(H)
            
        r_samples = [] # Amount of information at each position of the consensus (the larger the better)
        for i, u in enumerate(uncertainties):
            n = sum(counts[i].values())
            r = np.log2(len(alphabet))-(u+e_func(n))
            r_samples.append(r)
        
        bits_list = []
        for i, f in enumerate(frequencies):
            d = {}
            for u in f.keys():
                bits = f[u]*r_samples[i]
                d[u] = bits
            bits_list.append(d)
        
        return bits_list
    
    def __pwm_func(self, pseudocounts=0.5):
          
        pseudocounts = max(1e-3, pseudocounts) # safety
        
        pwm_list = []
        for d in self.__counts:
            totletters = len(d.keys())
            totcounts  = sum([val+pseudocounts for val in d.values()])
            # the simplest background model assumes that each letter appears equally frequently in the dataset:
            background = 1/totletters 
            e = {}
            for u in d.keys():
                e[u] = np.log2(((d[u]+pseudocounts)/totcounts) / background)
            pwm_list.append(e)
        
        return pwm_list
    
    def __consensus_func(self):
        cons = ""
        maxval = 0
        for pos in self.__pwm:
            aa, val = "", float("-inf")
            for letter, weight in pos.items():
                if weight > val:
                    aa = letter
                    val = weight
            cons += aa
            maxval += val
        return cons, maxval
    
    def __scan_func(self, sequence):
        result = []
        w = len(self.__pwm)
        for position in range(0, len(sequence)-w+1):
            fragment = sequence[position:position+w]
            score_sum = 0
            for position_fragment, letter in enumerate(fragment):
                # the score of a sequence can be calculated by adding (rather than multiplying) the relevant values at each position in the PWM.
                score_sum += self.__pwm[position_fragment][letter] 
            result.append((score_sum, position, fragment))
        return result
    
    def sequences(self, df=False):
        return pd.DataFrame(self.__sequences) if df is True else self.__sequences
    
    def consensus(self):
        return self.__consensus
    
    def consensusval(self):
        return self.__mv
    
    def length(self):
        return len(self.__consensus)
    
    def counts(self, df=False):
        return pd.DataFrame(self.__counts) if df is True else self.__counts
    
    def frequencies(self, df=False):
        return pd.DataFrame(self.__frequencies) if df is True else self.__frequencies
    
    def bits(self, df=False):
        return pd.DataFrame(self.__bits) if df is True else self.__bits
    
    def pwm(self, df=False):
        return pd.DataFrame(self.__pwm) if df is True else self.__pwm
    
    def pssm(self, *a, **kw):
        return self.pwm(*a, **kw)
    
    def scan(self, protein_sequence):
        return self.__scan_func(protein_sequence)
    
    def find(self, protein_sequence, score_threshold=float("-inf"), return_best=True):
        '''
        Uses the 'scan' function to return the best match
        '''
        scoreboard = sorted([x for x in self.scan(protein_sequence) if x[0] > score_threshold], key=lambda x: x[0], reverse=True)
        if len(scoreboard) == 0:
            return None
        elif return_best:
            spf = scoreboard[0]
            return float(spf[0])
        else:
            return scoreboard
    

def ids_from_fasta(fasta_filepath):
    ids = []
    with open(fasta_filepath, 'r') as f:
        line = f.readline()
        while line:
            if ">" in line:
                ids.append(line.lstrip(">").strip())
            line = f.readline()
    return ids

def parse_speclist(speclist_filepath, verbose=False):
    def line2stuff(line_):
        try:
            code_ = line_[0:5]
            kingdom_ = line_[6]
            taxon_ = line_[8:15]
            name_ = line_[19:].strip()
            if "V" in kingdom_: # trick for viruses named after their hosts
                species_ = name_.split("(")[0].strip()
            else:
                species_ = name_.split(" ")[0].strip()
            return code_, kingdom_, taxon_, name_, species_
        except:
            if verbose: print("Exception in 'parse_speclist'. Line:", line_)
            return None, None, None, None, None
    
    speclist = {}
    with open(speclist_filepath, 'r') as f:
        line = f.readline()
        while line and "_____" not in line:
            line = f.readline()
        line = f.readline()
        while line and "=====" not in line:
            _, kingdom, _, _, species = line2stuff(line)
            if kingdom is None or kingdom == " ":
                line = f.readline()
                continue
            speclist[species] = kingdom
            line = f.readline()
            
    return speclist

def last_M(topology):
    if ";" in topology:
        topology = topology.split(";")[-1].strip()
    flipped_topology = topology[::-1]
    try:
        flipped_Mf = flipped_topology.index("M")
        flipped_Mi = flipped_Mf+min(flipped_topology[flipped_Mf:].index("O"), flipped_topology[flipped_Mf:].index("I"))
        
        Mf = len(topology)-flipped_Mf
        Mi = len(topology)-flipped_Mi
    
    except:
        Mf = len(topology)
        Mi = 0
    
    return Mi, Mf, topology[Mi:Mf]

def convert_counts(seq, counts=None):
    if type(seq) == int:
        return convert_counts("".join(["X" for i in range(seq)]))
    if counts is None or type(counts) != list or counts == []:
        counts = []
        for aa in seq:
            cdict = {"P":0, "C":0, "H":0, "A":0, "N":0}
            if aa in "STNQY": # polar non-charged:
                cdict["P"] += 1
            if aa in "DEKRH": # charged
                cdict["C"] += 1
            if aa in "AVCPLIMWF": # hydrophobic non-polar
                cdict["H"] += 1
            if aa in "WFY": # aromatic
                cdict["A"] += 1
            if aa in "G": #neutral
                cdict["N"] += 1
            counts.append(cdict)
    elif len(seq) == len(counts):
        for i, aa in enumerate(seq):
            cdict = counts[i]
            if aa in "STNQY": # polar non-charged:
                cdict["P"] += 1
            if aa in "DEKRH": # charged
                cdict["C"] += 1
            if aa in "AVCPLIMWF": # hydrophobic non-polar
                cdict["H"] += 1
            if aa in "WFY": # aromatic
                cdict["A"] += 1
            if aa in "G": #neutral
                cdict["N"] += 1
    else:
        return None
    return counts

def convert_seq(seq):
    rules = {"A": "H",
             "C": "H",
             "D": "C",
             "E": "C",
             "F": "AH",
             "G": "N",
             "H": "C",
             "I": "H",
             "K": "C",
             "L": "H",
             "M": "H",
             "N": "P",
             "P": "H",
             "Q": "P",
             "R": "C",
             "S": "P",
             "T": "P",
             "V": "H",
             "W": "AH",
             "Y": "AP"}
    
    converted_bsignals = []
    
    def cnvrt(seq, lst=converted_bsignals, pos=0, rul=rules, conv=""):
        if len(conv) == len(seq):
            lst.append(conv)
        else:
            aatype = rul[seq[pos]]
            for c in aatype:
                cnvrt(seq, lst, pos+1, rul, conv+c) # recursive 
    
    cnvrt(seq, converted_bsignals)
    
    return converted_bsignals

            
#%% Generate beta-signals dataset

# 1) Parse OMPdb database

OMPdb = {}

with open(OMPdb_flat, 'r') as f:
    tmpdict = {"UNIPROT": ""}
    tmpstr = ""
    lastkey = ""
    line = f.readline()
    while line:
        if "//" in line:
            tmpdict = {k:v.rstrip("; ") for k, v in tmpdict.items()}
            OMPdb[tmpdict["UNIPROT"]] = tmpdict
            tmpdict = {"UNIPROT": ""}
            tmpstr = ""
            lastkey = ""
        else: #belongs to same entry
            key_, val = line[:14].strip(), line[14:].strip()
            if key_ != "":
                lastkey = key_
                tmpdict[key_] = val+"; "
            elif lastkey in tmpdict.keys():
                tmpdict[lastkey] += val
            else:
                tmpdict[lastkey] = val
                
        line = f.readline()

OMPdb_bad = {}  # dictionary of filtered-out entries


# 2) Remove redundants (>90% identical sequences) from OMPdb.

ids90 = ids_from_fasta(OMPdb_UniProt90)    # Get UniProt ids from the specified multi-FASTA file
k2pop = list(set(OMPdb.keys())-set(ids90)) # Get UniProt ids that are in the OMPdb, but NOT in the multi-FASTA file

for k in k2pop:                            # Remove ("pop") those entries with those ids from the OMPdb
    OMPdb_bad[k] = OMPdb.pop(k)
    
    
# 3) Retain only entries from kingdom bacteria.

speclist = parse_speclist(UniProt_speclist) # Parse list of organisms/species from specified file
k2pop    = []

for k, v in OMPdb.items(): # Check if organism of certain entry 'k' is a bacterium
    organism = v["ORGANISM"]
    species = organism.split(" ")[0]
    kingdom = None
    try:
        kingdom = speclist[species]
        if kingdom != "B":
            k2pop.append(k)
    except:
        if "bacter" not in organism:
            k2pop.append(k)

for k in k2pop: # Remove ("pop") those entries that are not from bacteria
    OMPdb_bad[k] = OMPdb.pop(k)


# 4) Remove entries with missing/unreliable precalculated topology.

k2pop = [] 

for k, v in OMPdb.items():
    if "TOPOLOGY" not in v.keys(): # 1) If topology keyword absent
        k2pop.append(k)
    elif len(v["TOPOLOGY"].split(";")[-1].strip()) < 10: # 2) If topology string absent or shorter than 10 aa
        k2pop.append(k)
    elif v["TOPOLOGY"].split(";")[-1].strip().count("M") < 8: # 3) If the intermembrane portion less than 8 aa
        k2pop.append(k)
    elif float(v["TOPOLOGY"].split(";")[-2][-5:-1]) < 80.0: # If reliability of the topology lower than 80%
        k2pop.append(k)

for k in k2pop:
    OMPdb_bad[k] = OMPdb.pop(k)


# 5) Remove entries that do not satisfy beta-signal specific requirements

tmplist = []
k2pop = []

for k, v in OMPdb.items():
    
    tillend = False # do not remove
    
    """
    # If you want to consider only entries from specific organism/species
    if "Escherichia coli" not in v["ORGANISM"]:
        k2pop.append(k)
        continue 
    """
    seq     = v["SEQUENCE"].split(";")[-1].strip()
    top     = v["TOPOLOGY"].split(";")[-1].strip()
    seqlen  = len(seq)
    Mi, Mf, _ = last_M(top)
    
    diff = 10-(Mf-Mi)  # Extend putative beta-signal sequence towards N-term, if length below 10 aa
    if diff > 0: Mi-=diff
    
    Mf0 = Mf # If the C-term of the putative beta-signal sequence does not match the C-term of the protein sequence ("true C-term"),
             # extend by up to 4 residues towards the true C-term until any aromatic is found. If more than 4 residues are required,
             # then extend to the true C-term (no matter how many residues).
    while Mf < seqlen and seq[Mi:Mf][-1] not in ["F","Y","W"]:
        if Mf-Mf0 >= 4:
            Mf = seqlen
            tillend = True
            break
        else:
            Mf += 1

    bsig = seq[Mi:Mf] # putative beta-signal sequence
    
    if len(bsig) == 0 or "X" in bsig:#or bsig[-1] not in ["F","Y","W"]:
        k2pop.append(k)
        continue # Jump directly to next loop iteration (skip current)
    
    # Regular expression from Paramsivam et al., 2012
    match = re.search("[^C][YFWKLHVITMADGRE][^C][YFWKLHVITMADGRE][^C][YFWKLHVITMADGRE][^C].[^C][YFWHILM]", bsig)
    if match is None:
        k2pop.append(k)
        continue # Jump directly to next loop iteration (skip current)
    elif tillend:
        Mi = match.start()
        Mf = match.end()
        bsig = bsig[Mi:Mf]
    
    bsig = bsig[-10:] # take last 10 residues only, like Paramsivam et al., 2012
    
    v["BSIG"] = bsig # add beta signal to OMPdb dictionary
    
    tmplist.append(">%s\n%s"%(k, bsig))
        
for k in k2pop:
    OMPdb_bad[k] = OMPdb.pop(k)
    
if bsignals_out is not None:
    f = open(bsignals_out, "w") # Write beta-signal sequences in the specified multi-FASTA file
    f.write("\n".join(tmplist))
    f.close()
    tmplist = []



#%% Analysis 1 (aa-type scoring)

bsignals7 = [v["BSIG"][-7:] for k, v in OMPdb.items()] # only consider last 7 characters
m7  = motif(bsignals7)

darobactin_score = m7.find(DAROBACTIN)

# 1) Generate scores for all beta-signals

scores = []
for bsig in bsignals7:
    best_score = m7.find(bsig)
    scores.append(best_score)

# 2) Generate scores for the "Random sequences"

random.seed(RANDOM_SEED) # Seed needed to replicate results

alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

rndsequences = []

rndsize = len(OMPdb) # Set size == OMPdb size. Generating all possible combinations would require too much memory

for i in range(rndsize):
    seq = "".join([random.choice(alphabet) for i in range(7)])
    rndsequences.append(seq)
    
rndscores = []
for bsig in rndsequences:
    best_score = m7.find(bsig)
    rndscores.append(best_score)
    
# 3) Generate scores for all darobactin combinations (permutations)

random.seed(RANDOM_SEED) # Seed needed to replicate results

cmbsequences = ["".join(c) for c in itertools.permutations(DAROBACTIN, len(DAROBACTIN))] # Possible combinations == 5040
cmbsequences = list(set(cmbsequences)) # Thereof unique combinations == 1260
    
cmbscores = []
for bsig in cmbsequences:
    best_score = m7.find(bsig)
    cmbscores.append(best_score)

# 4) Plot

# 4.1) Frequency logo

colorscheme = {
        "STNQ"    : [.10, .61, .99],
        "DEKRH"   : [.09, .60, .60],
        "AVCPLIM" : [.40, .02, .20],
        "WFY"     : [.99, .60, .25],
        "G"       : [.21, .40, .40]
        }

fig, ax = plt.subplots()
logomaker.Logo(df= m7.frequencies(df=True), ax = ax, color_scheme=colorscheme)
ax.set_ylabel("Frequency")
ax.set_title("Amino Acid Type Scoring")
plt.xticks(range(7), ("+7","+6","+5","+4","+3","+2","+1"))
plt.tight_layout()

if figures_out is not None:
    plt.savefig(figures_out+"1a.pdf")
    plt.close()
else:
    plt.show() # display e.g. in console

# 4.2) Histogram

binwidth = 1.0
b = np.arange(-50, 20 + binwidth, binwidth)
plt.hist(cmbscores, bins=b, color="darkolivegreen", alpha=0.5, density=True,
         label="Darobactin sequence permutations (%.1f%%)"%percentileofscore(cmbscores, darobactin_score))
plt.hist(rndscores, bins=b, color="firebrick", alpha=0.3, density=True,
         label="Random sequences (%.1f%%)"%percentileofscore(rndscores, darobactin_score))
plt.hist(scores,    bins=b, color="slategray", alpha=0.6, density=True,
         label=r"$\beta$-signal sequences (%.1f%%)"%percentileofscore(scores, darobactin_score))
vax =plt.gca()
plt.vlines(darobactin_score, 0, 1, transform=vax.get_xaxis_transform(), color="blue", label="Darobactin sequence")
plt.xlabel("Log-likelihood sum")
plt.ylabel("Density")
plt.ylim(0, 0.12)
plt.xlim(-50, 20)
plt.legend()
plt.tight_layout()

if figures_out is not None:
    plt.savefig(figures_out+"1b.pdf")
    plt.close()
else:
    plt.show()
    
# 5) Print summary to console (optional)
    
if PRINT_SUMMARY:
    print("\nStatistics for amino acid type scoring:\n")
    print("Beta-signal set size:\t", len(bsignals7))
    print("Rndseq      set size:\t", len(rndsequences))
    print("Cmbseq      set size:\t", len(cmbsequences))
    print("")
    print("Darobactin percentile in beta-signal set:\t",  percentileofscore(scores,    darobactin_score))
    print("Darobactin percentile in rndseq set:     \t",  percentileofscore(rndscores, darobactin_score))
    print("Darobactin percentile in cmbseq set:     \t",  percentileofscore(cmbscores, darobactin_score))



#%% Analysis 2 (aa-chemistry scoring)

counts = convert_counts(7) # initialize
for bsig in bsignals7:
    counts = convert_counts(bsig, counts)
    
m7c = motif([], alphabet=["H", "C", "P", "A", "N"], init_counts=counts) # initialize motif object from the counts, not sequences

darobactin_score_converted = max([m7c.find(bsig_converted) for bsig_converted in list(convert_seq(DAROBACTIN))])

use_best_score = True
if not use_best_score:
    random.seed(RANDOM_SEED) # Seed needed to replicate results


# 1) Generate scores for all beta-signals

scores2 = []
for bsig in bsignals7:
    if use_best_score:
        # convert beta-signal sequence to the one that gives the highest score among all possibilities (see 'rules')
        best_score = max([m7c.find(bsig_converted) for bsig_converted in list(convert_seq(bsig))])
        scores2.append(best_score)
    else:
        # convert random sequence to one among the possibilities, randomly (not necessarily the best, or worse) (see 'rules')
        bsig_converted = random.choice(convert_seq(bsig))
        rnd_score = m7c.find(bsig_converted)
        scores2.append(rnd_score)


# 2) Generate scores for the "Random sequences"
    
rndscores2 = []
for bsig in rndsequences:
    if use_best_score:
        best_score = max([m7c.find(bsig_converted) for bsig_converted in list(convert_seq(bsig))])
        rndscores2.append(best_score)
    else:
        bsig_converted = random.choice(convert_seq(bsig))
        rnd_score = m7c.find(bsig_converted)
        rndscores2.append(rnd_score)


# 3) Generate scores for all darobactin combinations (permutations)

cmbscores2 = []
for bsig in cmbsequences:
    if use_best_score:
        best_score = max([m7c.find(bsig_converted) for bsig_converted in list(convert_seq(bsig))])
        cmbscores2.append(best_score)
    else:
        bsig_converted = random.choice(convert_seq(bsig))
        rnd_score = m7c.find(bsig_converted)
        cmbscores2.append(rnd_score)

# 4) Plot

# 4.1) Frequency logo

colorscheme = {
        "P" : [.10, .61, .99],
        "C" : [.09, .60, .60],
        "H" : [.40, .02, .20],
        "A" : [.99, .60, .25],
        "N" : [.21, .40, .40]
        }

fig, ax = plt.subplots()
logomaker.Logo(df= m7c.frequencies(df=True), ax = ax, color_scheme=colorscheme)
ax.set_ylabel("Frequency")
ax.set_title("Amino Acid Chemistry Scoring")
plt.xticks(range(7), ("+7","+6","+5","+4","+3","+2","+1"))
plt.tight_layout()

if figures_out is not None:
    plt.savefig(figures_out+"2a.pdf")
    plt.close()
else:
    plt.show()

# 4.2) Histogram

binwidth = 0.5
b = np.arange(-20, 10 + binwidth, binwidth)
plt.hist(cmbscores2, bins=b, color="darkolivegreen", alpha=0.5, density=True,
         label="Darobactin sequence permutations (%.1f%%)"%percentileofscore(cmbscores2, darobactin_score_converted))
plt.hist(rndscores2, bins=b, color="firebrick", alpha=0.3, density=True,
         label="Random sequences (%.1f%%)"%percentileofscore(rndscores2, darobactin_score_converted))
plt.hist(scores2,    bins=b, color="slategray", alpha=0.6, density=True,
         label=r"$\beta$-signal sequences (%.1f%%)"%percentileofscore(scores2, darobactin_score_converted))
vax =plt.gca()
plt.vlines(darobactin_score_converted, 0, 1, transform=vax.get_xaxis_transform(), color="blue", label="Darobactin sequence")
plt.xlabel("Log-likelihood sum")
plt.ylabel("Density")
plt.ylim(0, 0.31)
plt.xlim(-20, 10)
plt.legend()
plt.tight_layout()

if figures_out is not None:
    plt.savefig(figures_out+"2b.pdf")
    plt.close()
else:
    plt.show()

# 5) Print summary

if PRINT_SUMMARY:
    print("\nStatistics for amino acid chemistry scoring:\n")
    print("Beta-signal set size:\t", len(bsignals7))
    print("Rndseq      set size:\t", len(rndsequences))
    print("Cmbseq      set size:\t", len(cmbsequences))
    print("")
    print("Darobactin percentile in beta-signal set:\t",  percentileofscore(scores2,    darobactin_score_converted))
    print("Darobactin percentile in rndseq set:     \t",  percentileofscore(rndscores2, darobactin_score_converted))
    print("Darobactin percentile in cmbseq set:     \t",  percentileofscore(cmbscores2, darobactin_score_converted))