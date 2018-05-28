import pandas as pd
import numpy as np
from collections import defaultdict

labels = pd.read_csv('OASIS3Labels.csv')
MRSesh = pd.read_csv('OASIS3MRSessions_Subject_Age.csv')
out = MRSesh[['MR ID', 'Subject', 'Age']]
#out['Age'] = MRSesh['Age'].astype(float)

labels['daysPostEntry'] = labels['ADRC_ADRCCLINICALDATA ID'].str.split('d').str.get(1)
labels['yearsPostEntry'] = pd.to_numeric(labels['daysPostEntry'], downcast='float') / 365.
labels = labels.assign(ageAtDiagnosis=labels[['ageAtEntry', 'yearsPostEntry']].sum(1))
subject2age2diag = defaultdict(dict)
for i, row in labels.iterrows():
    subject2age2diag[row['Subject']][row['ageAtDiagnosis']] = row['cdr']

#out['Age'] = out['Age'].astype(float)
cdr_list = []
for i, row in out.iterrows():
    MR_age = row['Age']
    age_diff_tuples = []
    for diag_age in subject2age2diag[row['Subject']]:
        age_diff_tuples.append((abs(float(diag_age) - float(MR_age)), subject2age2diag[row['Subject']][diag_age]))
    min_tuple = min(age_diff_tuples, key=lambda x: x[0])
    if min_tuple[0] < 3.:
        cdr_list.append(min_tuple[1])
    else:
        cdr_list.append(None)

cdr_col = pd.Series(cdr_list)
out.insert(loc=1, column='cdr', value=cdr_col)
clean_out = out[['MR ID', 'cdr']]
clean_out.to_csv('OASIS3_MRID2Label_052918.csv', index=False)