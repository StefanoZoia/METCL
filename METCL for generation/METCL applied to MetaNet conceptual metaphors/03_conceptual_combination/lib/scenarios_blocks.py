# returns the the best-scored valid scenarios (rows)
def best_block(table):
    
    res = []
    row_index = 0
    table_len = len(table.table)

    while res == [] and row_index < table_len:
        
        # build next block
        block = []
        max_prob = table.table[row_index][-1]

        while row_index < table_len and table.table[row_index][-1] == max_prob:
            block.append(table.table[row_index])
            row_index += 1

        # add to result the consistent scenarios in the block (if any)
        for scen in block:
            if table.consistent_scenario(scen):
                res.append(scen)
    return res  

