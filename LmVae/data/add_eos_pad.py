def add_eos():
    fw=open('sents_17w_eospad.csv','w')
    with open('sents_17w.csv','r') as f:
        for line in f.readlines():
            fw.write(line.strip('\n')+' <eos> <pad>\n')
            
add_eos()