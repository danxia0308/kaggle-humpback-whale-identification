#!/usr/bin/env python3
import pickle

testpkl = pickle.loads(open("/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/metadata/p2h.pickle", "rb").read())

pickle.dump(testpkl, open("/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/metadata/p2h2.pickle","wb"), protocol=2)
