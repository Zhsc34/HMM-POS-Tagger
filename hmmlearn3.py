import sys
import numpy as np
import json 
import os.path
from operator import itemgetter

#Check if a file path has been entered
if len(sys.argv) == 1:
	print("Please enter a file path!")
	sys.exit()

#make sure the file entered is valid
if os.path.isfile(sys.argv[1]) == False:
	print("Please enter a valid file")
	sys.exit()

with open(sys.argv[1], 'r', encoding = 'utf-8') as fin:
	tagset = set([]) # set of tags
	tagcount = {} # tags and their corresponding numbers of occurrence in the training data
	vocab = set([]) # set of vocabulary
	wseq = [] # sequence of words (observations)
	tseq = [] # sequence of tags (states)
	endtag = {} # number of occurrences of each tag before sentence boundaries
	starttag = {} # number of occurrences of each tag at the start of sentences
	sentnum = 0 # number of sentences in the training data
	while True:
		td = fin.readline()
		if td == "":
			break
		else:
			td = td.split()
			for i in range(0, len(td)): 
				wtsequence = td[i]
				num = wtsequence.rfind('/')
				w = wtsequence[:num]
				t = wtsequence[num + 1:]
				wseq.append(w)
				tseq.append(t)
				vocab.add(w.lower()) # disregard any capitalization for this model
				tagset.add(t)
				if t in tagcount:
					tagcount[t] += 1
				else:
					tagcount[t] = 1
				if i == 0:
					if t in starttag:
						starttag[t] += 1
					else:
						starttag[t] = 1
				if i == len(td) - 1:
					if t in endtag:
						endtag[t] += 1
					else:
						endtag[t] = 1
			sentnum += 1
	tagset = np.array(list(tagset))
	vocab = np.array(list(vocab))
	wseq = np.array(wseq)
	tseq = np.array(tseq)
fin.closed

tp = np.zeros((len(tagset), len(tagset))) # transition probability
ep = np.zeros((len(tagset), len(vocab))) # emission probability
endp = [] # end tag probability
sp = [] # starting tag probability

tagtoindex = {}

for i in range(0, len(tagset)):
	tagtoindex[tagset[i]] = i

vocabtoindex = {}

for i in range(0, len(vocab)):
	vocabtoindex[vocab[i]] = i

for i in range(0, len(tseq) - 1):
	tp[tagtoindex[tseq[i]]][tagtoindex[tseq[i + 1]]] += 1
	ep[tagtoindex[tseq[i]]][vocabtoindex[wseq[i].lower()]] += 1

oc = []

for i in range(0, len(tagset)):
	tc = tagcount[tagset[i]]
	if len(np.argwhere(ep[i, :] == 1))/tc >= 0.05: # calculating open class; for simplicity, we assume that a tag is open if and only if more than 5% of its observations occurred only once
		oc.append(i)
	tp[i, :] = (tp[i, :] + 1)/(tc + 1 * len(tagset)) # using add-one smoothing for transition probabilities
	ep[i, :] /= tc
	if tagset[i] in endtag:
		endp.append((endtag[tagset[i]] + 1)/(sentnum + len(tagset)))
	else:
		endp.append(1/(sentnum + len(tagset)))
	if tagset[i] in starttag:
		sp.append((starttag[tagset[i]] + 1)/(sentnum + len(tagset)))
	else:
		sp.append(1/(sentnum + len(tagset)))


fout = open("./hmmmodel.txt", "w+", encoding = "utf-8")

fout.write("States (POS tags):\n")
json.dump(tagset.tolist(), fout)

fout.write("\n\nObservations (Seen words): \n")
json.dump(vocab.tolist(), fout, ensure_ascii = False)

fout.write("\n\nTransition Probabilities (probability on row i and column j corresponds to the transition probability from the ith state to the jth state in the list of states above):\n")
json.dump(tp.tolist(), fout)

fout.write("\n\nEmission Probabilities (probability on row i and column j corresponds to the emission probability of the ith state emitting jth vocabulary in the list of states and vocabulary above):\n")
json.dump(ep.tolist(), fout)

fout.write("\n\nTransition Probabilities for END tag (probability at index i corresponds to the probability of the ith tag in the tagset being at the end of a sentence): \n")
json.dump(endp, fout)

fout.write("\n\nTransition Probabilities for START tag (probability at index i corresponds to the probability of the ith tag in the tagset being at the start of a sentence): \n")
json.dump(sp, fout)

fout.write("\n\nSet of open class tags (tags that have significant number of unseen and rare words): \n")
json.dump(oc, fout)

fout.closed