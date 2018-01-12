import numpy
class StudentSim():
    def __init__(self, predictor, scaler):
        self.p = predictor
        self.scaler = scaler
        #self.s = None
        self.havedone = set()
        #self.s = numpy.zeros(shape=swidth)

    def encodeinput(self, state, qenc):
        return state + qenc

    # def passprob(self, uK, qenc):
    #     # take an encoded question and return a pass probability
    #     inp = numpy.append(uK.flatten(), qenc.flatten()).reshape(1,-1)
    #     print(inp.shape)
    #     print(inp)
    #     sinp = self.scaler.transform(inp)
    #     probs = self.p.predict(sinp) #_proba(sinp)
    #     # print(probs)
    #     #return (probs[0][0])
    #     return 1 if probs==0 else 0

    def doipass(self, A, X, qenc):
        # take an encoded question and return true or false, stochastically
        # print(X.shape)
        # print(qenc.shape)
        inp = numpy.append(X.flatten(), qenc.flatten())
        sinp = self.scaler.transform(inp.reshape(1, 68))
        p = self.p.predict_proba(sinp)  # _proba(sinp)
        print(p)

        if A in self.havedone:
            #print("have seen!",A)
            return False
        else:
            if numpy.random.rand() <= p[0,0]:
                self.havedone.add(A)
                return True
            else:
                return False
            #att = numpy.random.normal()
            # print("ppass {}, attempt {}".format(prob, att))
            #return att < prob

    def updatemystate(self, qenc, passed=False):
        #do the state update thing here
        #namely take the updateinfo tuple and decode it into
        #meaningful changes to self.s
        #specifically in the K33 model, we update the category with the new score
        assert len(numpy.nonzero(qenc))==1 # only one value shd be non zero
        catix = numpy.nonzero(qenc)[0]
        self.s[catix] = self.s[catix] + qenc[catix] # no learning rate specified yet!

    def encode_student(self, S,K):
        jnd = numpy.append(S.flatten(), K.flatten())
        return jnd