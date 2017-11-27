import numpy
class StudentSim():
    def __init__(self, predictor):
        self.p = predictor
        #self.s = None
        self.haveseen = set()
        #self.s = numpy.zeros(shape=swidth)

    def encodeinput(self, state, qenc):
        return state + qenc

    def passprob(self, uK, qenc):
        # take an encoded question and return a pass probability
        inp = numpy.append(uK.flatten(), qenc.flatten())#.reshape(1,-1)
        print(inp.shape)
        input("inp")
        probs = self.p.predict_proba(inp)
        #print(probs)
        return (probs[0][0])

    def doipass(self, A, uK, qenc):
        # take an encoded question and return true or false, stochastically
        if A in self.haveseen:
            #print("have seen!",A)
            return False
        else:
            self.haveseen.add(A)
            prob = self.passprob(uK, qenc)
            # if prob>=0.5:
            #     return True
            # else:
            #     return False
            att = numpy.random.normal()
            # print("ppass {}, attempt {}".format(prob, att))
            return att < prob

    def updatemystate(self, qenc, passed=False):
        #do the state update thing here
        #namely take the updateinfo tuple and decode it into
        #meaningful changes to self.s
        #specifically in the K33 model, we update the category with the new score
        assert len(numpy.nonzero(qenc))==1 # only one value shd be non zero
        catix = numpy.nonzero(qenc)[0]
        self.s[catix] = self.s[catix] + qenc[catix] # no learning rate specified yet!