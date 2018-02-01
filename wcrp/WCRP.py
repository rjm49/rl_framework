from collections import Counter

import pandas
import numpy
import scipy

tables = []
table_affs = []
all_affs = []
alpha = 10.0
bias = 0.6

#number of customers at table y with affiliation a
def n_a_y(table, aff):
    retval= table_affs[table].count(aff)
    # print("{} people at table {} with aff {}".format(retval, table, aff))
    return retval

def K_a_y(table, aff):
    if bias==1.0:
        return 1.0;
    nom = (1.0-bias)**(-n_a_y(table, aff))
    denom = numpy.sum( [ (1.0-bias)**(-n_a_y(table, _aff)) for _aff in all_affs ] )
    # print("K_a_y = {} / {}".format(nom,denom))
    return (nom/denom)

def prob_of_occd_table(table, n_tables, cust_aff, all_affs):
    nom = 1.0+ bias*(K_a_y(table, cust_aff)-1.0)
    denom = 1.0+ bias*(1.0/n_tables -1.0)
    ny = len(tables[table])
    print("prob of occd tab = {} * {} / {}".format(ny, nom, denom))
    return ( ny * nom / denom)

qtypes = pandas.read_csv("../../../isaac_data_files/atypes.csv", header=None, index_col=0)
for qix in range(qtypes.shape[0]):
    qrow = qtypes.iloc[qix,:]
    affil = "/".join(map(str,qrow[[2,3,4]]))
    if affil not in all_affs:
        all_affs.append(affil)
    print(qix, affil)

    print("tables are", tables)
    print("affiliations are", table_affs)
    if not tables: #first customer
        print("frist")
        tables.append([qix])
        table_affs.append([affil])
        print("sat {} with affil {}".format(qix, affil))
    else:
        print("new arrival to seat...")
        n_tables = len(tables)
        #already have customers
        p_new_table = [alpha*(1-bias)]
        #p_occ_tables = [ b/(alpha+qix) for b in [len(grp) for grp in at_table]]
        p_occ_tables = [prob_of_occd_table(t, n_tables, affil, all_affs) for t,tab in enumerate(tables)]
        ws = p_new_table + p_occ_tables # concat into a single list of weights
        ws = numpy.divide(ws, numpy.sum(ws))
        print(ws)
        choice = numpy.random.choice(len(ws), p=ws)
        if choice == 0: # new table
            tables.append([qix])
            table_affs.append([affil])
        else: #add to table
            tables[choice - 1].append(qix)
            table_affs[choice - 1].append(affil)

    print("tables->",tables)
    print("groupsizes->",[len(grp) for grp in tables])
    print("affs->",table_affs)

for tx, afflist in enumerate(table_affs):
    mode = max(afflist, key=Counter(afflist).get)
    mode_cnt = afflist.count(mode)
    print("{} : {} {} / {}".format(tx, mode, mode_cnt, len(afflist)))