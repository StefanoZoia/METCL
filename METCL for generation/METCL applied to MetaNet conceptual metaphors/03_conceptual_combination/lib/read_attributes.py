# input file parsing class
class ReadAttributes:

    def __init__(self, path = 'input.txt'): 
        
        # save file lines on a list
        with open(path) as f:
            self.input_lines = f.readlines()

        # remove comments and void lines
        self.input_lines = [x.strip() for x in self.input_lines 
                            if x.strip() != '' and x.strip()[0] != '#']
        
        # save title, head name and modifier name
        self.title = self.input_lines[0].split(':')[1].strip()
        self.input_lines.pop(0)
        self.head_conc = self.input_lines[0].split(':')[1].strip()
        self.input_lines.pop(0)
        self.mod_conc = self.input_lines[0].split(':')[1].strip()
        self.input_lines.pop(0)

        # scan remaining lines building rigid and typical property lists
        self.typical_attrs = []
        self.attrs = []
        for line in self.input_lines:
            line = [k.strip() for k in line.split(',')]
            if len(line) == 3 and line[0][0] == 'T':
                # typical attr := (property, probability, belongs_to_head)
                self.typical_attrs.append(tuple([
                                                    line[1],
                                                    float(line[2]),
                                                    (True if line[0][2:-1] == 'head' else False)
                                                ]))
            if len(line) == 2:
                # rigid attr := (property, belongs_to_head)
                self.attrs.append(tuple([
                                            line[1],
                                            (True if line[0] == 'head' else False)
                                        ]))

