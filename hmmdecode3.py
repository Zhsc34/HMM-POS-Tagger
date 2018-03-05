import sys
import numpy as np
import json 
import os.path

#suppress warnings on calculating log of 0
np.seterr(divide = 'ignore')

# check if a file path has been entered
if len(sys.argv) == 1:
	print("Please enter a file path!")
	sys.exit()

#  check if the file entered is valid
if os.path.isfile(sys.argv[1]) == False:
	print("Please enter a valid file")
	sys.exit()

with open("./hmmmodel.txt", 'r', encoding = 'utf-8') as model:
	model.readline()
	tagset = np.array(json.loads(model.readline()))
	model.readline()
	model.readline()
	vocab = np.array(json.loads(model.readline()))
	model.readline()
	model.readline()
	tp = np.array(np.log2(json.loads(model.readline())))
	model.readline()
	model.readline()
	ep = np.array(np.log2(json.loads(model.readline())))
	model.readline()
	model.readline()
	endp = np.array(np.log2(json.loads(model.readline())))
	model.readline()
	model.readline()
	sp = np.array(np.log2(json.loads(model.readline())))
	model.readline()
	model.readline()
	oc = np.array(json.loads(model.readline()))
model.closed

tagtoindex = {}
isopen = {}

for i in range(0, len(tagset)):
	tagtoindex[tagset[i]] = i
	if i in oc:
		isopen[i] = True
	else:
		isopen[i] = False


vocabtoindex = {}

for i in range(0, len(vocab)):
	vocabtoindex[vocab[i]] = i

tagsize = tagset.size
vocsize = vocab.size


with open("./hmmoutput.txt", 'w+', encoding = 'utf-8') as fout:
	with open(sys.argv[1], 'r', encoding = 'utf-8') as fin:
		while True:
			td = fin.readline()
			if td == "":
				break
			else:
				# initializing matrices
				tdorig = td.split()
				td = np.array([x.lower() for x in tdorig])
				tdsize = td.size
				viterbi = np.ndarray([tagsize, tdsize])
				backpointer = np.ndarray([tagsize, tdsize])

				# initialization of the first column
				isInVocab = td[0] in vocab
				if isInVocab:
					vocabIndex = vocabtoindex[td[0]]
				for i in range(0, tagsize):
					logProb = 0.0
					if isInVocab:
						logProb = ep[i, vocabIndex]
					else:
						if isopen[i] == False:
							logProb = np.NINF
					viterbi[i, 0] = sp[i] + logProb
				

				for T in range(1, tdsize):
					isInVocab = td[T] in vocab
					if isInVocab:
						vocabIndex = vocabtoindex[td[T]]
					for s in range(0, tagsize):
						prevTrans = tp[:, s] + viterbi[:, T - 1]
						maxidx = np.argmax(prevTrans)
						viterbi[s, T] = prevTrans[maxidx]
						backpointer[s, T] = maxidx
						logProb = 0.0
						if isInVocab:	
							logProb = ep[s, vocabIndex]
						else:
							if isopen[s] == False:
								logProb = np.NINF
						viterbi[s, T] += logProb

				# end of sentences and backtrack
				endprob = viterbi[:, tdsize - 1] + endp
				path = np.empty([tdsize])
				path[tdsize - 1] = np.argmax(endprob)
				for i in range(tdsize - 2, -1, -1):
					path[i] = backpointer[np.int(path[i + 1]), i + 1]
				tagpath = []
				for i in range(0, tdsize):
					fout.write(tdorig[i] + "/" + tagset[np.int(path[i])])
					if i != tdsize - 1:
						fout.write(" ")
				fout.write("\n")
	fin.closed
fout.closed