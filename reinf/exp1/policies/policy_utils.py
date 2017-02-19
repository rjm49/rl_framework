import codecs
def state_as_str(S):
    Ss = "".join([ '1' if boo else '0' for boo in S ])
#         print(S, "-> ", Ss)
    return Ss

def statestr_as_tuple(ss):
    s = []
    ls=list(ss)
    while(ls):
        s.append(bool(ls.pop(0)))
    print(s)
    print(tuple(s))
    return tuple(s)

def save_policy(qvals, fname):
    pol_str = qvals_to_policy(qvals)
    f = codecs.open(fname,"w")
    f.write(pol_str)
    f.close()

def load_policy(fname, concepts=None):
    pol_str = codecs.open(fname, "r").readlines()
    if(concepts):
        # map values to concepts (by ID) if they are provided
        return policy_to_qvals(pol_str, concepts)
    else:
        #else just return the string
        return pol_str


def qvals_to_policy(qvals):
    '''
    This fn takes a dict of state-[action] weights (or Q-values) converts to flat format
    '''
    outstr = ""
    print(qvals)
    for state in sorted(qvals):        
        outstr += str(state_as_str(state)) + " "
        for action in sorted(qvals[state], key=lambda x: (x.id if x else -1)):
            act_weight = qvals[state][action]
            outstr = "{} {}:{:.4f}".format(outstr, (action.id if action else -1), act_weight)
        outstr += "\n"
    
    print(outstr)
    return outstr

def policy_to_qvals(pol, concepts):
    qvals = {}
    lines = pol.split("\n")
    for line in lines:
        state = line.split("  ")[0]
        acts = line.split("  ")[1]
        stup = statestr_as_tuple(state)
        act_d = {}
        for a in acts.split(" "):
            actid = a.split(":")[0]
            act = concepts[actid]
            wgt = a.split(":")[1]
            act_d[act]=wgt
        qvals[stup]=act_d
        