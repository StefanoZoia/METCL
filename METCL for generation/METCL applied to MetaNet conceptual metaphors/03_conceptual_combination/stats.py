import os
import cocos_config as cfg

def main():
    n_files = 0
    n_results = 0
    for filename in os.listdir(cfg.COCOS_DIR):
        file_path = os.path.join(cfg.COCOS_DIR, filename)
        with open(file_path) as f:
            input_lines = f.readlines()
            # remove comments and void lines
            input_lines = [x.strip() for x in input_lines 
                            if x.strip() != '' and x.strip()[0] != '#']
            
            n_files += 1
            file_has_result = False
            
            for line in input_lines:
                line = [k.strip() for k in line.split(':', maxsplit=1)]
                if line[0] == 'Result':
                    file_has_result = True
            
            if file_has_result:
                n_results += 1
    
    print(f'total combination files: {n_files}')
    print(f'total successful combinations: {n_results}')


if __name__ == '__main__' : main()