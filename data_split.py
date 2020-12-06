############################################################################
#creates json files for train, dev1, dev2, test in data folder in home dir
#requires train files to be in train folder in home dir
#requires test files to be in test folder in home dir
#requires data folder to be in home dir
############################################################################

from sklearn.model_selection import train_test_split
from utils import read_records
import json

train_data = read_records('train/')
test_data = read_records('test/')

# FIXING ANNOTATIONS IN TEST FILE IN ACCORDANCE WITH THE FOLLOWING INSTRUCTIONS
# 
# Please note that, as per the discussion in the Google Group
# (https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/BEXpfnwOR4A)
# the annotations in the training and test set should be modified so that
# CABG, PTCA, and PCI are used consistently as evidence of ischemia for the
# ADVANCED-CAD category.  In particular,  the files 140, 156, 205, 266, and 277 should have the
# annotation <ADVANCED-CAD met="met”>.
# However, the files included here for evaluation of challenge submissions do not and will not include these corrections,
# so they are consistent with the training data that was available before the submission
# deadline.

wronglyLabeled = ['140','156','205','266','277']
for file in wronglyLabeled:
    tags = test_data[file]['tags']
    tags['ADVANCED-CAD'] = 'met'
    test_data[file]['tags'] = tags
    
train_list = [(k,v) for k,v in train_data.items()]
test_list = [(k,v) for k,v in test_data.items()]

#adding 20 files from gold standard test set to train
test, train_extra = train_test_split(test_list, test_size=0.25, random_state=42)
train_list.extend(train_extra)

#splitting train into 10% dev1 and 10% dev2
train_1, dev2 = train_test_split(train_list, test_size=0.1,random_state=42)
train, dev1 = train_test_split(train_1, test_size=0.1,random_state=42)

#converting back to dicts
train_dict = {k:v for k,v in train}
dev1_dict = {k:v for k,v in dev1}
dev2_dict = {k:v for k,v in dev2}
test_dict = {k:v for k,v in test}

#saving json files
files = [train_dict,dev1_dict,dev2_dict,test_dict]
outputFiles = ['./data/train.txt', './data/dev1.txt', './data/dev2.txt', './data/test.txt']

for i in range(len(files)):
    with open(outputFiles[i],'w') as outfile:
        json.dump(files[i], outfile)
