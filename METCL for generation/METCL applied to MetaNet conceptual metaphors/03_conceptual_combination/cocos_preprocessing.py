# Builds a COCOS input file from the prototypes of two concepts in input

import cocos_config as cfg
import csv
import os


# returns a dict containing the typical properties read from the specified path
def getTypicalProperties(path_file) :    
    prop_dict = dict()

    with open(path_file) as f:
        for p in f.readlines():
            prop = p.split(':')
            value = float(prop[1].strip())
            prop_dict[prop[0]] = round(value, 3)

    return prop_dict

# returns a list containing the rigid properties read from the specified path
def getRigidProperties(path_file) :    
    prop_list = list()

    if os.path.exists(path_file):
        with open(path_file) as f:
            for p in f.readlines():
                prop_list.append(p.strip())

    return prop_list

# write the input file for COCOS
def write_cocos_file(head, modifier):
    with open(f'{cfg.COCOS_DIR}/{head}_{modifier}.txt', "w") as f:
        f.write(f'Title: {head}-{modifier}\n\n')
        f.write(f'Head Concept Name: {head}\n')
        f.write(f'Modifier Concept Name: {modifier}\n\n')
    
        # rigid properties
        for p in getRigidProperties(f'{cfg.RIGID_PROP_DIR}/{head}.txt'):
            f.write("head, " + p)
        f.write("\n\n")
        
        for p in getRigidProperties(f'{cfg.RIGID_PROP_DIR}/{modifier}.txt'):
            f.write("modifier, " + p)
        f.write("\n\n")
        
        # typical properties
        modifier_typ = getTypicalProperties(f'{cfg.TYPICAL_PROP_DIR}/{modifier}.txt')
        for p in modifier_typ:
            f.write(f'T(modifier), {p}, {modifier_typ[p]}\n')
        f.write("\n")
        
        head_typ = getTypicalProperties(f'{cfg.TYPICAL_PROP_DIR}/{head}.txt')
        for p in head_typ:
            f.write(f'T(head), {p}, {head_typ[p]}\n')
        f.write("\n")



def main():
    # remove old prototype files
    for filename in os.listdir(cfg.COCOS_DIR):
        file_path = os.path.join(cfg.COCOS_DIR, filename)
        try:
            os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    # for each metaphor in input, write file for the combination of source and target
    # NOTE: source concept is being used as modifier
    with open(cfg.CORPUS_FILE, newline='', encoding='utf-8') as corpus:
        reader = csv.reader(corpus, delimiter='\t', quotechar='"')
        for row in reader:
            source = row[0].lower().replace(" ", "_").replace("-", "_")
            target = row[1].lower().replace(" ", "_").replace("-", "_")
            write_cocos_file(head=target, modifier=source)


if __name__ == '__main__' : main()
