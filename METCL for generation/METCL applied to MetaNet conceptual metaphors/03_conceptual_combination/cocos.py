import math
import lib.read_attributes as ra
import lib.scenarios_table as st
import lib.scenarios_blocks as sb
import cocos_config as cfg
import os
import json


# translates a scenario from bits to properties
def scenario_to_properties(scenario, typical_props):
    properties_dict = dict()

    # for each typical property
    for i in range(len(typical_props)):
        # if the property is selected in scenario
        if scenario[i] == 1:
            # add property and probability to dict
            properties_dict[typical_props[i][0]] = typical_props[i][1]

    # add the scenario's score to the dict and return 
    properties_dict['@scenario_probability'] = scenario[-1]
    return properties_dict

# perfroms the combination of concepts
def cocos(filename, max_attrs = math.inf):

    # read input file
    input_data = ra.ReadAttributes(filename)

    # build table of non-trivial scenarios
    tab = st.Table(input_data, max_attrs)

    print(f'\n\nHead Concept: {input_data.head_conc}, Modifier Concept: {input_data.mod_conc}')

    # retrieve the best-scored consistent scenario(s)
    best = sb.best_block(tab)
    if best != []:
        print('\nRecommended scenario(s):')
        for scenario in best:
           print(f'  - {scenario_to_properties(scenario, input_data.typical_attrs)}')
           
        # write the result on the input file
        with open(filename, "a") as f:
            result = scenario_to_properties(best[0], input_data.typical_attrs)
            f.write("\nResult: " + json.dumps(result))
            f.write("\nScenario: " + str(best[0]))
    else:
        print('\nNO recommended scenarios!')


def main():
    for file in os.listdir(cfg.COCOS_DIR):
        cocos(os.path.join(cfg.COCOS_DIR, file), cfg.MAX_ATTRS)
    

if __name__ == '__main__' : main()